#!/usr/bin/env python3
"""Deterministic tests for the controlled-capture orchestrator's accounting, state
transitions, single-instance lock, and repository-root portability.

These tests deliberately do NOT run the benchmark binary or the real soak. They stub the
environment primitives (soak, run_session) so every scenario is reproducible in
milliseconds, and they assert on the append-only state ledger and the counted-attempt
derivation -- the machinery that decides whether the three-attempt stopping rule fired.
This is the layer that failed on 2026-07-19 (two concurrent instances corrupting each
other's accounting); the frozen measurement logic is not exercised here and is unchanged.

Run: python3 tests/capture_orchestrator_test.py
"""
from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import time
import types
import unittest

MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "capture_scaling_baseline.py"


def load_module():
    """Load the orchestrator as a module without running main() (it is __main__-guarded)."""
    spec = importlib.util.spec_from_file_location("capture_scaling_baseline", MODULE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_time():
    """A time replacement whose sleep is instant but whose monotonic is real, so horizon
    arithmetic still advances and no test ever actually blocks."""
    return types.SimpleNamespace(sleep=lambda _s: None, monotonic=time.monotonic)


def canned_samples(n=10, value=12.0):
    return [value] * n


class CaptureAccountingTest(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.tmp = pathlib.Path(td.name)
        # Redirect all persistent state into the temp dir and neutralise real sleeps.
        self.mod.SCRATCH = self.tmp / "scratch"
        self.mod.REVIEW_DIR = self.tmp / "review"
        self.mod.SCRATCH.mkdir(parents=True, exist_ok=True)
        self.mod.time = fake_time()
        # A binary path that "exists" so main() gets past its build-guard.
        binpath = self.tmp / "dw_bench"
        binpath.write_text("#!/bin/sh\n")
        self.mod.BIN = binpath

    # ----- pure ledger accounting -------------------------------------------------

    def _ledger(self):
        return self.mod.read_ledger()

    def _counted(self, waiter_id=None):
        return self.mod.counted_attempt_run_ids(self._ledger(), waiter_id)

    def test_scenario1_soak_only_timeout_is_not_an_attempt(self):
        """A soak that never clears before any session begins is a waiting-only void."""
        self.mod.soak = lambda hold_s, label, max_wait_s: (False, canned_samples())
        outcome, sessions = self.mod.one_controlled_run(self.mod.SCRATCH / "run-a", "w1")
        self.assertEqual(outcome, "void_waiting")
        self.assertEqual(sessions, [])
        self.assertEqual(self._counted("w1"), set(), "waiting-only void must not count")
        # A VOID_ENVIRONMENT (waiting only) verdict file must be written.
        self.assertIn("waiting only", (self.mod.SCRATCH / "run-a" / "VERDICT").read_text())

    def test_scenario2_session_began_then_void(self):
        """Session 0 completes, then session 1's soak times out: a started attempt voids."""
        soak_results = iter([(True, canned_samples()), (False, canned_samples())])
        self.mod.soak = lambda *_a, **_k: next(soak_results)
        self.mod.run_session = lambda *_a, **_k: ("ok", {"context": {}})
        outcome, sessions = self.mod.one_controlled_run(self.mod.SCRATCH / "run-b", "w1")
        self.assertEqual(outcome, "void_started")
        self.assertEqual(self._counted("w1"), {"run-b"}, "session-began void counts as one attempt")

    def test_scenario3_retry_exhaustion(self):
        """First case's retries exhaust -> run_session returns non-ok -> counted void."""
        self.mod.soak = lambda *_a, **_k: (True, canned_samples())
        self.mod.run_session = lambda *_a, **_k: ("void_started", None)
        outcome, _ = self.mod.one_controlled_run(self.mod.SCRATCH / "run-c", "w1")
        self.assertEqual(outcome, "void_started")
        self.assertEqual(self._counted("w1"), {"run-c"})

    def test_scenario4_completed_numerical_rejection(self):
        """All four sessions complete but the numeric gate rejects: a counted attempt."""
        self.mod.soak = lambda *_a, **_k: (True, canned_samples())
        self.mod.run_session = lambda *_a, **_k: ("ok", {"context": {}})
        self.mod.evaluate = lambda sessions: "REJECT (session-to-session spread too high)"
        outcome, _ = self.mod.one_controlled_run(self.mod.SCRATCH / "run-d", "w1")
        self.assertEqual(outcome, "reject")
        self.assertEqual(self._counted("w1"), {"run-d"})

    def test_scenario5_accepted_protocol_is_terminal_not_counted(self):
        """An accepted run is terminal; it does not consume an attempt slot."""
        self.mod.soak = lambda *_a, **_k: (True, canned_samples())
        self.mod.run_session = lambda *_a, **_k: ("ok", {"context": {}})
        self.mod.evaluate = lambda sessions: "ACCEPTED"
        outcome, sessions = self.mod.one_controlled_run(self.mod.SCRATCH / "run-e", "w1")
        self.assertEqual(outcome, "accepted")
        self.assertEqual(len(sessions), self.mod.SESSIONS)
        self.assertEqual(self._counted("w1"), set(), "accepted run is not a counted attempt")

    def test_scenario6_termination_between_dir_and_session_never_counts(self):
        """A run whose process died after run_start but before any terminal state: the
        ledger has run_start only, so it is never counted."""
        self.mod.append_state("w1", "run_start", "run-f")
        # (process dies here -- no terminal record is ever written)
        self.assertEqual(self._counted("w1"), set())
        self.assertEqual(self._counted(None), set())

    def test_scenario7_restart_starts_clean_but_preserves_history(self):
        """A prior waiter's counted attempt stays in the ledger, but a restart's own
        accounting begins empty (waiter-scoped counting)."""
        self.mod.append_state("old", "run_start", "run-g")
        self.mod.append_state("old", "run_void_started", "run-g")
        self.assertEqual(self._counted("old"), {"run-g"})
        self.assertEqual(self._counted("new"), set(), "restart accounting starts clean")
        self.assertEqual(self._counted(None), {"run-g"}, "history is preserved for audit")

    def test_scenario8_no_double_counting_of_partial_or_replayed_records(self):
        """Duplicate/partial terminal records for one run count that run exactly once."""
        self.mod.append_state("w1", "run_start", "run-h")
        self.mod.append_state("w1", "run_void_started", "run-h")
        self.mod.append_state("w1", "run_void_started", "run-h")  # replayed / double write
        self.mod.append_state("w1", "run_start", "run-h")         # partial re-entry
        self.assertEqual(self._counted("w1"), {"run-h"})
        self.assertEqual(len(self._counted("w1")), 1)

    # ----- singleton lock ---------------------------------------------------------

    def test_singleton_lock_excludes_a_second_instance(self):
        first = self.mod.acquire_singleton_lock()
        self.assertIsNotNone(first, "first waiter must acquire the lock")
        second = self.mod.acquire_singleton_lock()
        self.assertIsNone(second, "a concurrent second waiter must be refused")
        first.close()
        third = self.mod.acquire_singleton_lock()
        self.assertIsNotNone(third, "lock is reusable once released")
        third.close()

    # ----- stopping-rule integration through main() -------------------------------

    def test_main_three_started_voids_fire_stopping_rule(self):
        """Three started-then-void attempts -> UNSUITABLE, with ledger count == 3."""
        self.mod.soak = lambda *_a, **_k: (True, canned_samples())
        self.mod.run_session = lambda *_a, **_k: ("void_started", None)
        rc = self.mod.main()
        self.assertEqual(rc, 1)
        verdict = (self.mod.SCRATCH / "FINAL_VERDICT").read_text()
        self.assertIn("UNSUITABLE_MOVE_TO_DEDICATED", verdict)
        # The waiter's own attempts (whatever its generated id) must number exactly three.
        recs = self._ledger()
        waiter_ids = {r["waiter_id"] for r in recs if r["state"] == "waiter_start"}
        self.assertEqual(len(waiter_ids), 1)
        (wid,) = waiter_ids
        self.assertEqual(len(self._counted(wid)), 3)

    def test_main_accepts_and_does_not_count_waiting_voids(self):
        """A waiting-only void precedes an accepted run: rc 0, zero counted attempts."""
        outcomes = iter(["void_waiting", "accepted"])

        def scripted(run_dir, waiter_id):
            run_dir.mkdir(parents=True, exist_ok=True)
            state = next(outcomes)
            self.mod.append_state(waiter_id, "run_start", run_dir.name)
            if state == "void_waiting":
                self.mod.append_state(waiter_id, "run_void_waiting", run_dir.name)
                return "void_waiting", []
            self.mod.append_state(waiter_id, "run_accepted", run_dir.name)
            return "accepted", [{"context": {}} for _ in range(self.mod.SESSIONS)]

        self.mod.one_controlled_run = scripted
        rc = self.mod.main()
        self.assertEqual(rc, 0)
        self.assertIn("ACCEPTED_PENDING_REVIEW", (self.mod.SCRATCH / "FINAL_VERDICT").read_text())
        recs = self._ledger()
        (wid,) = {r["waiter_id"] for r in recs if r["state"] == "waiter_start"}
        self.assertEqual(self._counted(wid), set())

    def test_main_refuses_when_another_instance_holds_the_lock(self):
        held = self.mod.acquire_singleton_lock()
        self.addCleanup(held.close)
        rc = self.mod.main()
        self.assertEqual(rc, 4, "main must exit 4 when the singleton lock is contended")

    # ----- portability ------------------------------------------------------------

    def test_repo_root_is_resolved_from_file_location(self):
        self.assertEqual(self.mod.REPO, MODULE_PATH.resolve().parents[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
