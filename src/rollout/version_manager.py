# Version manager for rollout workers

import threading

class VersionManager:
    """Tracks the current policy version and notifies rollout workers.

    The orchestrator increments the version after each weight sync. Workers can
    query the manager to obtain the latest version number.
    """

    def __init__(self, initial_version: int = 0):
        self._version = initial_version
        self._lock = threading.Lock()

    def get_version(self) -> int:
        with self._lock:
            return self._version

    def bump_version(self) -> int:
        with self._lock:
            self._version += 1
            return self._version

    # ── Test-facing API ──────────────────────────────────────────────────────

    def current(self) -> int:
        """Return the current policy version."""
        return self.get_version()

    def bump(self) -> None:
        """Increment the policy version by 1."""
        self.bump_version()

    def staleness(self, version: int) -> int:
        """Return how many versions behind ``version`` is relative to current."""
        return self.current() - version
