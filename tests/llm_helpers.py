from __future__ import annotations

import threading

from macaw.coding.llm.backends import LLMResponse


class CountingBackend:
    """A model that answers by a rule of the prompt and seed, and counts the calls under way at once."""

    name = "fake"
    takes_seed = True

    def __init__(self, answer, *, gate: int = 1, delay=None, fail_at: int | None = None, context: int | None = None, cut_off: bool = False):
        self.answer = answer
        # calls after the first are held until this many are under way
        self.gate = gate
        self.delay = delay or (lambda user, seed: 0.0)
        # the call, counted from 1, that raises
        self.fail_at = fail_at
        self.error = RuntimeError("the server fell over")
        self.context = context
        self.cut_off = cut_off
        self.calls: list[tuple[str, int | None]] = []
        self.finished: list[tuple[str, int | None]] = []
        self.most = self.running = 0
        self._lock = threading.Lock()
        self._full = threading.Event()

    def context_length(self) -> int | None:
        return self.context

    def complete(self, system, user, *, seed=None, images=()):
        with self._lock:
            self.calls.append((user, seed))
            index = len(self.calls)
            self.running += 1
            self.most = max(self.most, self.running)
            if self.running >= self.gate:
                self._full.set()
        try:
            if index == self.fail_at:
                raise self.error
            if index > 1:
                self._full.wait(timeout=5)
            # an event's wait, as the tests turn time.sleep off
            threading.Event().wait(self.delay(user, seed))
            return LLMResponse(text=self.answer(user, seed), model="fake", cut_off=self.cut_off)
        finally:
            with self._lock:
                self.running -= 1
                self.finished.append((user, seed))
