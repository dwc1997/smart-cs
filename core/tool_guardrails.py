"""Tool Guardrails

Track tool call signatures per turn and detect:
  - Exact loops: same tool+args repeated N times
  - No-progress cycles: different tools but same result pattern

Pure side-effect-free: only tracks and reports, never modifies state.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    """A single tool call observation."""
    tool_name: str
    args: dict[str, Any]
    result: str
    signature: str  # deterministic hash of (tool_name, sorted args)
    result_hash: str  # deterministic hash of result content


@dataclass
class GuardrailVerdict:
    """Returned after each check — describes what, if anything, to do."""
    ok: bool  # True → keep going, False → should stop
    warning: bool  # True → a soft warning was raised (still ok=True)
    reason: Optional[str] = None
    suggestion: Optional[str] = None


class ToolGuardrails:
    """Track tool calls within a turn and detect stuck loops.

    Usage:
        guardrails = ToolGuardrails()  # fresh per turn
        for tc in tool_calls:
            verdict = guardrails.check(tool_name, args, result)
            if not verdict.ok:
                # break out of loop
                break

    Configuration:
        warn_threshold:  issue a warning after this many identical calls
        hard_stop:       stop after this many identical calls
        result_cycle_len: window size for no-progress cycle detection
    """

    def __init__(
        self,
        warn_threshold: int = 2,
        hard_stop: int = 3,
        result_cycle_len: int = 4,
    ):
        if hard_stop < warn_threshold:
            raise ValueError("hard_stop must be >= warn_threshold")
        self.warn_threshold = warn_threshold
        self.hard_stop = hard_stop
        self.result_cycle_len = result_cycle_len

        self._records: list[ToolCallRecord] = []
        self._warnings_issued: set[str] = set()

    # ── public API ────────────────────────────────────────────────

    @property
    def call_count(self) -> int:
        return len(self._records)

    def check(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: str,
    ) -> GuardrailVerdict:
        """Check a tool call against guardrails.

        This is a pure side-effect-free call. It records the call and
        returns a verdict but never modifies external state.

        Returns:
            GuardrailVerdict with ok=True to continue, ok=False to stop.
        """
        record = self._make_record(tool_name, args, result)
        self._records.append(record)

        # 1. Exact loop detection: same signature repeated hard_stop times
        loop_count = self._count_consecutive(record.signature)
        if loop_count >= self.hard_stop:
            reason = (
                f"Tool '{tool_name}' called with identical arguments "
                f"{loop_count} times in a row — hard stop."
            )
            logger.warning(f"[guardrails] {reason}")
            return GuardrailVerdict(ok=False, warning=False, reason=reason)

        if loop_count >= self.warn_threshold:
            warn_key = f"loop:{record.signature}"
            if warn_key not in self._warnings_issued:
                self._warnings_issued.add(warn_key)
                reason = (
                    f"Tool '{tool_name}' called with identical arguments "
                    f"{loop_count} times — possible loop."
                )
                logger.warning(f"[guardrails] {reason}")
                return GuardrailVerdict(
                    ok=True,
                    warning=True,
                    reason=reason,
                    suggestion="Try a different approach or stop.",
                )

        # 2. No-progress cycle detection: different tools, same result pattern
        if len(self._records) >= self.result_cycle_len:
            recent_hashes = [
                r.result_hash for r in self._records[-self.result_cycle_len:]
            ]
            unique_signatures = {
                r.signature for r in self._records[-self.result_cycle_len:]
            }
            # Different tools but all results are identical
            if len(unique_signatures) > 1 and len(set(recent_hashes)) == 1:
                reason = (
                    f"Last {self.result_cycle_len} tool calls produced identical "
                    f"results despite using different arguments — no progress."
                )
                logger.warning(f"[guardrails] {reason}")
                return GuardrailVerdict(ok=False, warning=False, reason=reason)

            # Even same tools: if last N results are all identical AND
            # we've seen warn_threshold+ with same result pattern
            if len(set(recent_hashes)) == 1:
                total_same_result = sum(
                    1 for r in self._records if r.result_hash == recent_hashes[0]
                )
                if total_same_result >= self.hard_stop:
                    reason = (
                        f"Tool calls produced the same result {total_same_result} "
                        f"times — no progress detected."
                    )
                    logger.warning(f"[guardrails] {reason}")
                    return GuardrailVerdict(ok=False, warning=False, reason=reason)

        return GuardrailVerdict(ok=True, warning=False)

    def get_history(self) -> list[dict[str, Any]]:
        """Return the recorded tool call history (for debugging)."""
        return [
            {
                "tool": r.tool_name,
                "args": r.args,
                "result_hash": r.result_hash,
                "signature": r.signature,
            }
            for r in self._records
        ]

    def reset(self) -> None:
        """Reset state for a new turn."""
        self._records.clear()
        self._warnings_issued.clear()

    # ── internals ─────────────────────────────────────────────────

    @staticmethod
    def _make_record(
        tool_name: str,
        args: dict[str, Any],
        result: str,
    ) -> ToolCallRecord:
        sig_src = json.dumps(
            {"tool": tool_name, "args": args},
            sort_keys=True,
            ensure_ascii=False,
        )
        signature = hashlib.sha256(sig_src.encode()).hexdigest()[:16]

        # Normalize result: strip whitespace variations for robust comparison
        result_normalized = result.strip() if isinstance(result, str) else str(result)
        result_hash = hashlib.sha256(result_normalized.encode()).hexdigest()[:16]

        return ToolCallRecord(
            tool_name=tool_name,
            args=args,
            result=result,
            signature=signature,
            result_hash=result_hash,
        )

    def _count_consecutive(self, signature: str) -> int:
        """Count how many consecutive trailing records match *signature*."""
        count = 0
        for rec in reversed(self._records):
            if rec.signature == signature:
                count += 1
            else:
                break
        return count
