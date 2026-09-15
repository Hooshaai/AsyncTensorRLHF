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
