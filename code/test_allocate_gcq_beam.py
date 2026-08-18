import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from allocate_gcq_beam import (
    CACHE_FILE,
    TRACE_FILE,
    BeamAllocationError,
    BeamRun,
    Candidate,
    load_candidates,
)


def members_key(row):
    return frozenset(row["members"])


class BeamAllocatorTests(unittest.TestCase):
    def make_run(self, root, costs, cap, width=4, pareto=False):
        candidates = [Candidate(name, cost) for name, cost in costs]
        return BeamRun.initialize(
            root, candidates, cap, beam_width=width, pareto_prune=pareto
        )

    def score_plan(self, run, objective, *, reverse=False, partial=False):
        plan = run.plan()
        self.assertIsNotNone(plan)
        rows = list(plan["states"])
        if reverse:
            rows.reverse()
        results = {
            row["state_id"]: objective(frozenset(row["members"])) for row in rows
        }
        if partial and len(results) > 1:
            items = list(results.items())
            midpoint = len(items) // 2
            run.record(dict(items[:midpoint]))
            self.assertTrue(run.status_summary()["pending_state_ids"])
            run = BeamRun(run.run_dir)
            self.assertEqual(
                plan["artifact_name"], run.plan()["artifact_name"]
            )
            run.record(dict(items[midpoint:]))
        else:
            run.record(results)
        return run, plan

    def drive(self, run, objective, *, reverse=False, partial=False):
        while not run.is_complete:
            plan = run.plan()
            if plan is None:
                break
            rows = list(plan["states"])
            if reverse:
                rows.reverse()
            results = {
                row["state_id"]: objective(frozenset(row["members"]))
                for row in rows
            }
            if partial and len(results) > 1:
                items = list(results.items())
                run.record(dict(items[::2]))
                run = BeamRun(run.run_dir)
                self.assertEqual(
                    plan["artifact_name"], run.plan()["artifact_name"]
                )
                run.record(dict(items[1::2]))
            else:
                run.record(results)
        return run

    def final_members(self, run):
        final = run.status_summary()["final"]
        self.assertIsNotNone(final)
        return tuple(final["members"])

    def test_width_four_beats_greedy_on_conditional_interaction(self):
        scores = {
            frozenset(): 0.0,
            frozenset(("a",)): 5.0,
            frozenset(("b",)): 4.0,
            frozenset(("c",)): 4.0,
            frozenset(("a", "b")): 5.1,
            frozenset(("a", "c")): 5.2,
            frozenset(("b", "c")): 10.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            greedy = self.make_run(
                Path(tmp) / "greedy", [("c", 1), ("a", 2), ("b", 1)], 3, width=1
            )
            beam = self.make_run(
                Path(tmp) / "beam", [("c", 1), ("a", 2), ("b", 1)], 3, width=4
            )
            greedy = self.drive(greedy, scores.__getitem__)
            beam = self.drive(beam, scores.__getitem__)
            self.assertEqual(("a", "c"), self.final_members(greedy))
            self.assertEqual(("b", "c"), self.final_members(beam))
            self.assertEqual(5.2, greedy.status_summary()["final"]["score"])
            self.assertEqual(10.0, beam.status_summary()["final"]["score"])

    def test_exact_cap_is_never_exceeded_and_reports_budget_stop(self):
        def objective(members):
            return float(len(members))

        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp, [("a", 3), ("b", 4), ("too_big", 6)], 5)
            run = self.drive(run, objective)
            cache = json.loads((Path(tmp) / CACHE_FILE).read_text())
            self.assertTrue(cache["entries"])
            self.assertTrue(all(row["cost_bytes"] <= 5 for row in cache["entries"]))
            self.assertEqual("budget_exhausted", run.data["stop_reason"])
            self.assertLessEqual(run.status_summary()["final"]["cost_bytes"], 5)

    def test_smaller_cap_selection_uses_completed_primary_trace(self):
        def objective(members):
            return float(sum({"a": 3, "b": 2, "c": 1}[name] for name in members))

        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp, [("a", 3), ("b", 2), ("c", 1)], 5)
            with self.assertRaisesRegex(BeamAllocationError, "completed run"):
                run.select_scored_state(3)
            run = self.drive(run, objective)
            secondary = run.select_scored_state(3)
            self.assertEqual(3, secondary["selection_cap_bytes"])
            self.assertEqual(run.data["catalog_hash"], secondary["catalog_hash"])
            self.assertEqual(["a"], secondary["state"]["members"])
            self.assertLessEqual(secondary["state"]["cost_bytes"], 3)
            with self.assertRaisesRegex(BeamAllocationError, "exceeds"):
                run.select_scored_state(6)

    def test_strictly_nonpositive_marginals_stop_at_baseline(self):
        scores = {
            frozenset(): 1.0,
            frozenset(("a",)): 1.0,  # Equality is not a positive transition.
            frozenset(("b",)): 0.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp, [("a", 1), ("b", 1)], 1)
            run = self.drive(run, scores.__getitem__)
            self.assertEqual((), self.final_members(run))
            self.assertEqual("non_positive_marginal", run.data["stop_reason"])

    def test_tie_break_is_score_then_bytes_then_lexicographic_members(self):
        scores = {
            frozenset(): 0.0,
            frozenset(("a",)): 1.0,
            frozenset(("b",)): 1.0,
            frozenset(("c",)): 1.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp, [("c", 1), ("b", 1), ("a", 1)], 1, width=1)
            run = self.drive(run, scores.__getitem__, reverse=True)
            self.assertEqual(("a",), self.final_members(run))

    def test_record_order_and_partial_resume_do_not_change_result(self):
        scores = {
            frozenset(): 0.0,
            frozenset(("a",)): 3.0,
            frozenset(("b",)): 2.0,
            frozenset(("c",)): 1.0,
            frozenset(("a", "b")): 3.1,
            frozenset(("a", "c")): 3.2,
            frozenset(("b", "c")): 8.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            first = self.make_run(
                Path(tmp) / "first", [("a", 1), ("b", 1), ("c", 1)], 2
            )
            second = self.make_run(
                Path(tmp) / "second", [("c", 1), ("b", 1), ("a", 1)], 2
            )
            first = self.drive(first, scores.__getitem__)
            second = self.drive(
                second, scores.__getitem__, reverse=True, partial=True
            )
            self.assertEqual(self.final_members(first), self.final_members(second))
            self.assertEqual(
                first.status_summary()["final"]["score"],
                second.status_summary()["final"]["score"],
            )

    def test_child_state_is_deduplicated_across_multiple_parents(self):
        singleton_scores = {
            frozenset(): 0.0,
            frozenset(("a",)): 1.0,
            frozenset(("b",)): 1.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp, [("a", 1), ("b", 1)], 2)
            run, _ = self.score_plan(run, singleton_scores.__getitem__)  # baseline
            run, _ = self.score_plan(run, singleton_scores.__getitem__)  # singletons
            pair_plan = run.plan()
            self.assertEqual(1, len(pair_plan["states"]))
            pair = pair_plan["states"][0]
            self.assertEqual(["a", "b"], pair["members"])
            self.assertEqual(2, len(pair["parents"]))

    def test_pareto_pruning_removes_costlier_lower_scoring_state(self):
        scores = {
            frozenset(): 0.0,
            frozenset(("a",)): 2.0,
            frozenset(("b",)): 1.0,
            frozenset(("c",)): 3.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(
                tmp, [("a", 1), ("b", 2), ("c", 2)], 2, pareto=True
            )
            run, _ = self.score_plan(run, scores.__getitem__)  # baseline
            run, _ = self.score_plan(run, scores.__getitem__)  # singletons
            beam_members = {members_key(row) for row in run.status_summary()["beam"]}
            self.assertEqual(
                {frozenset(("a",)), frozenset(("c",))}, beam_members
            )

    def test_cache_trace_and_plan_are_valid_resumable_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp, [("a", 1), ("b", 1)], 2)
            baseline = run.plan()
            plan_path = Path(tmp) / baseline["artifact_name"]
            self.assertEqual(baseline, json.loads(plan_path.read_text()))
            baseline_id = baseline["states"][0]["state_id"]
            run.record({baseline_id: 0.0})
            expansion = run.plan()
            first = expansion["states"][0]
            run.record({first["state_id"]: 1.0})

            resumed = BeamRun(tmp)
            self.assertEqual(
                expansion["artifact_name"], resumed.plan()["artifact_name"]
            )
            remaining = {
                row["state_id"]: 1.0
                for row in expansion["states"]
                if row["state_id"] != first["state_id"]
            }
            resumed.record(remaining)
            json.loads((Path(tmp) / CACHE_FILE).read_text())
            trace_rows = [
                json.loads(line)
                for line in (Path(tmp) / TRACE_FILE).read_text().splitlines()
            ]
            self.assertEqual(
                list(range(len(trace_rows))),
                [row["event_index"] for row in trace_rows],
            )

    def test_validation_rejects_noninteger_costs_and_conflicting_scores(self):
        with self.assertRaises(BeamAllocationError):
            Candidate("bad", 1.5)
        with self.assertRaises(BeamAllocationError):
            Candidate("bad", True)
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp, [("a", 1)], 1)
            plan = run.plan()
            state_id = plan["states"][0]["state_id"]
            run.record({state_id: 0.0})
            expansion = run.plan()
            child_id = expansion["states"][0]["state_id"]
            run.record({child_id: 1.0})
            # The completed expansion has no active plan, so stale/conflicting
            # writes cannot silently mutate the cache.
            with self.assertRaises(BeamAllocationError):
                run.record({child_id: 2.0})

    def test_shortlist_module_name_alias_and_bound_context_reach_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = Path(tmp) / "shortlist.json"
            catalog_path.write_text(json.dumps({
                "candidates": [{"module_name": "projection", "delta_bytes": 7}]
            }))
            candidates, embedded = load_candidates(catalog_path)
            self.assertIsNone(embedded)
            self.assertEqual([Candidate("projection", 7)], candidates)
            context = "ab" * 32
            run = BeamRun.initialize(
                Path(tmp) / "run", candidates, 7, context_sha256=context
            )
            plan = run.plan()
            self.assertEqual(context, plan["context_sha256"])
            self.assertEqual(context, run.status_summary()["context_sha256"])

    def test_tampered_frozen_shortlist_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shortlist.json"
            value = {
                "schema_version": 1,
                "top_k": 1,
                "candidates": [{"module_name": "a", "delta_bytes": 1}],
            }
            value["shortlist_sha256"] = hashlib.sha256(
                (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
            ).hexdigest()
            path.write_text(json.dumps(value))
            self.assertEqual([Candidate("a", 1)], load_candidates(path)[0])
            value["candidates"][0]["delta_bytes"] = 2
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(BeamAllocationError, "content hash"):
                load_candidates(path)


if __name__ == "__main__":
    unittest.main()
