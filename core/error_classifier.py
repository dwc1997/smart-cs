"""Error Classifier

Classify API errors into categories with recovery actions.
Supports exponential backoff with jitter for retries.
"""

import asyncio
import logging
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Error categories for API failures."""
    RATE_LIMIT = "rate_limit"
    AUTH_ERROR = "auth_error"
    CONTEXT_OVERFLOW = "context_overflow"
    NETWORK_ERROR = "network_error"
    MODEL_ERROR = "model_error"
    UNKNOWN = "unknown"


class RecoveryAction(str, Enum):
    """Recovery actions for each error category."""
    RETRY = "retry"
    ROTATE_CREDENTIAL = "rotate_credential"
    COMPRESS_CONTEXT = "compress_context"
    FALLBACK_MODEL = "fallback_model"
    ABORT = "abort"


# Category → (recovery action, max retries, base delay seconds)
_CATEGORY_POLICY: dict[ErrorCategory, tuple[RecoveryAction, int, float]] = {
    ErrorCategory.RATE_LIMIT:      (RecoveryAction.RETRY,             5, 2.0),
    ErrorCategory.AUTH_ERROR:       (RecoveryAction.ROTATE_CREDENTIAL, 0, 0.0),
    ErrorCategory.CONTEXT_OVERFLOW: (RecoveryAction.COMPRESS_CONTEXT,  1, 1.0),
    ErrorCategory.NETWORK_ERROR:    (RecoveryAction.RETRY,             3, 1.0),
    ErrorCategory.MODEL_ERROR:      (RecoveryAction.FALLBACK_MODEL,    2, 1.0),
    ErrorCategory.UNKNOWN:          (RecoveryAction.ABORT,             0, 0.0),
}


@dataclass
class ErrorClassification:
    """Result of classifying an exception."""
    category: ErrorCategory
    action: RecoveryAction
    max_retries: int
    base_delay: float
    message: str
    original_exception: Optional[Exception] = None

    @property
    def should_retry(self) -> bool:
        return self.action == RecoveryAction.RETRY

    @property
    def should_abort(self) -> bool:
        return self.action == RecoveryAction.ABORT

    @property
    def should_compress(self) -> bool:
        return self.action == RecoveryAction.COMPRESS_CONTEXT


@dataclass
class RetryState:
    """Tracks retry attempts per error category within a call."""
    attempts: dict[str, int] = field(default_factory=dict)

    def attempt(self, category: ErrorCategory) -> int:
        count = self.attempts.get(category.value, 0) + 1
        self.attempts[category.value] = count
        return count

    def get(self, category: ErrorCategory) -> int:
        return self.attempts.get(category.value, 0)


class ErrorClassifier:
    """Classify API errors and determine recovery actions.

    Usage:
        classifier = ErrorClassifier()
        state = RetryState()

        try:
            result = await call_llm(...)
        except Exception as e:
            classification = classifier.classify(e)
            if classification.should_retry and state.attempt(classification.category) <= classification.max_retries:
                delay = classifier.backoff_delay(classification, state)
                await asyncio.sleep(delay)
                # retry ...
            else:
                raise  # or abort
    """

    # ── pattern matchers (applied to str(exception)) ──────────────

    _RATE_LIMIT_PATTERNS = [
        re.compile(r"rate.?limit", re.IGNORECASE),
        re.compile(r"429"),
        re.compile(r"too many requests", re.IGNORECASE),
        re.compile(r"quota exceeded", re.IGNORECASE),
        re.compile(r"retry.after", re.IGNORECASE),
    ]

    _AUTH_ERROR_PATTERNS = [
        re.compile(r"401"),
        re.compile(r"403"),
        re.compile(r"unauthorized", re.IGNORECASE),
        re.compile(r"invalid.*api.?key", re.IGNORECASE),
        re.compile(r"authentication", re.IGNORECASE),
        re.compile(r"permission denied", re.IGNORECASE),
    ]

    _CONTEXT_OVERFLOW_PATTERNS = [
        re.compile(r"context.*length", re.IGNORECASE),
        re.compile(r"token.*limit", re.IGNORECASE),
        re.compile(r"maximum.*context", re.IGNORECASE),
        re.compile(r"too.?long", re.IGNORECASE),
        re.compile(r"max.*token", re.IGNORECASE),
    ]

    _NETWORK_ERROR_PATTERNS = [
        re.compile(r"timeout", re.IGNORECASE),
        re.compile(r"connection", re.IGNORECASE),
        re.compile(r"network", re.IGNORECASE),
        re.compile(r"ECONNREFUSED|ECONNRESET|ETIMEDOUT", re.IGNORECASE),
        re.compile(r"502|503|504"),
        re.compile(r"server.*error", re.IGNORECASE),
    ]

    _MODEL_ERROR_PATTERNS = [
        re.compile(r"model.*not.*found", re.IGNORECASE),
        re.compile(r"model.*unavailable", re.IGNORECASE),
        re.compile(r"500"),
        re.compile(r"internal.*server", re.IGNORECASE),
    ]

    # ── public API ────────────────────────────────────────────────

    def classify(self, exc: Exception) -> ErrorClassification:
        """Classify an exception into an ErrorClassification."""
        text = str(exc)
        # Also check __cause__ and status_code attributes
        status = getattr(exc, "status_code", None)

        category = self._detect_category(text, status)
        action, max_retries, base_delay = _CATEGORY_POLICY[category]

        return ErrorClassification(
            category=category,
            action=action,
            max_retries=max_retries,
            base_delay=base_delay,
            message=text,
            original_exception=exc,
        )

    def backoff_delay(
        self,
        classification: ErrorClassification,
        retry_state: RetryState,
    ) -> float:
        """Compute exponential backoff with jitter.

        delay = base_delay * 2^(attempt-1) + random jitter [0, base_delay)
        """
        attempt = retry_state.get(classification.category)
        base = classification.base_delay
        delay = base * (2 ** (attempt - 1))
        jitter = random.uniform(0, base)
        return delay + jitter

    # ── internals ─────────────────────────────────────────────────

    def _detect_category(self, text: str, status: Optional[int] = None) -> ErrorCategory:
        """Detect error category from exception text and optional status code."""
        if status is not None:
            if status == 429:
                return ErrorCategory.RATE_LIMIT
            if status in (401, 403):
                return ErrorCategory.AUTH_ERROR
            if status == 502 or status == 503 or status == 504:
                return ErrorCategory.NETWORK_ERROR
            if status == 500:
                return ErrorCategory.MODEL_ERROR

        for pattern in self._RATE_LIMIT_PATTERNS:
            if pattern.search(text):
                return ErrorCategory.RATE_LIMIT
        for pattern in self._AUTH_ERROR_PATTERNS:
            if pattern.search(text):
                return ErrorCategory.AUTH_ERROR
        for pattern in self._CONTEXT_OVERFLOW_PATTERNS:
            if pattern.search(text):
                return ErrorCategory.CONTEXT_OVERFLOW
        for pattern in self._NETWORK_ERROR_PATTERNS:
            if pattern.search(text):
                return ErrorCategory.NETWORK_ERROR
        for pattern in self._MODEL_ERROR_PATTERNS:
            if pattern.search(text):
                return ErrorCategory.MODEL_ERROR

        return ErrorCategory.UNKNOWN
