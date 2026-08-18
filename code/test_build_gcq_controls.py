import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from allocate_gcq_beam import Candidate
from build_gcq_controls import (
    ControlBuildError,
    ExactSubsetCounter,
    InfeasibleTargetError,
    POLICIES,
    build_controls_manifest,
    load_fixed_members,
    load_score_map,
    main,
    sha256_file,
    solve_exact_knapsack,
    subset_cost,
    write_manifest_exclusive,
)


CONTEXT_SHA256 = "a" * 64


def repeated_score_maps(scores):
    return {policy: dict(scores) for policy in POLICIES}


class MatchedControlTests(unittest.TestCase):
    def test_shortlist_is_directly_usable_as_additive_gcq_score_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shortlist.json"
            path.write_text(json.dumps({"candidates": [
                {"module_name": "a", "repair_macro": 0.3, "delta_bytes": 1},
                {"module_name": "b", "repair_macro": 0.2, "delta_bytes": 1},
            ]}))
            self.assertEqual({"a": 0.3, "b": 0.2}, load_score_map(path))

    def test_historical_substrings_expand_to_exact_projection_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp) / "coarse.json"
            spec.write_text(json.dumps({"bits": 8, "substrings": ["layers.10.self_attn"]}))
            names = [
                "model.language_model.layers.10.self_attn.q_proj",
                "model.language_model.layers.10.self_attn.k_proj",
                "model.language_model.layers.10.mlp.up_proj",
            ]
            self.assertEqual(
                sorted(names[:2]),
                load_fixed_members(spec, candidate_names=names),
            )
            spec.write_text(json.dumps({"substrings": ["layers.99"]}))
            with self.assertRaisesRegex(ControlBuildError, "match no projection"):
                load_fixed_members(spec, candidate_names=names)

    def test_exact_knapsack_finds_global_optimum_not_single_best_item(self):
        candidates = [Candidate("a", 2), Candidate("b", 1), Candidate("c", 1)]
        scores = {"a": 5.0, "b": 3.0, "c": 3.0}
        solution = solve_exact_knapsack(candidates, scores, 2)
        self.assertEqual(("b", "c"), solution.members)
        self.assertEqual(2, solution.cost_bytes)
        self.assertEqual("6.0", str(solution.objective))

    def test_knapsack_tie_uses_lexicographically_smallest_members(self):
        candidates = [Candidate("z", 1), Candidate("a", 1), Candidate("m", 1)]
        scores = {"z": 2.0, "a": 2.0, "m": 2.0}
        solution = solve_exact_knapsack(candidates, scores, 1)
        self.assertEqual(("a",), solution.members)

    def test_uniform_unranking_is_bijection_over_exact_cost_support(self):
        candidates = [Candidate("a", 1), Candidate("b", 1), Candidate("c", 1)]
        counter = ExactSubsetCounter(candidates, 2)
        self.assertEqual(3, counter.support_size)
        support = {counter.unrank(rank) for rank in range(counter.support_size)}
        self.assertEqual(
            {("a", "b"), ("a", "c"), ("b", "c")}, support
        )
        self.assertTrue(
            all(subset_cost(candidates, members) == 2 for members in support)
        )

    def test_uniform_sampling_is_seed_deterministic_and_has_support(self):
        candidates = [Candidate("a", 1), Candidate("b", 1), Candidate("c", 1)]
        first = ExactSubsetCounter(candidates, 2)
        second = ExactSubsetCounter(reversed(candidates), 2)
        seeds = list(range(64))
        samples_one = [first.sample(seed) for seed in seeds]
        samples_two = [second.sample(seed) for seed in seeds]
        self.assertEqual(samples_one, samples_two)
        observed = {members for _, members in samples_one}
        self.assertEqual(
            {("a", "b"), ("a", "c"), ("b", "c")}, observed
        )

    def test_infeasible_exact_target_fails_loudly(self):
        candidates = [Candidate("a", 2), Candidate("b", 4)]
        scores = {"a": 1.0, "b": 2.0}
        with self.assertRaises(InfeasibleTargetError):
            solve_exact_knapsack(candidates, scores, 3)
        with self.assertRaises(InfeasibleTargetError):
            ExactSubsetCounter(candidates, 3)

    def test_manifest_matches_every_control_byte_and_preserves_all_seeds(self):
        candidates = [Candidate("a", 2), Candidate("b", 1), Candidate("c", 1)]
        scores = {"a": 5.0, "b": 3.0, "c": 3.0}
        seeds = [11, 7, 29, 3]
        manifest = build_controls_manifest(
            candidates,
            2,
            repeated_score_maps(scores),
            seeds,
            CONTEXT_SHA256,
            coarse_members=["b"],
        )
        self.assertEqual(CONTEXT_SHA256, manifest["protocol_context_sha256"])
        for row in manifest["score_driven_controls"].values():
            self.assertEqual(2, row["cost_bytes"])
            self.assertTrue(row["matches_target"])
        random_block = manifest["random_controls"]
        self.assertEqual(seeds, random_block["frozen_seeds"])
        self.assertEqual(seeds, [row["seed"] for row in random_block["samples"]])
        self.assertTrue(random_block["no_best_seed_selection"])
        self.assertTrue(
            all(row["cost_bytes"] == 2 for row in random_block["samples"])
        )
        coarse = manifest["coarse_historical_control"]
        self.assertEqual(1, coarse["actual_cost_bytes"])
        self.assertFalse(coarse["matches_target"])

    def test_historical_coarse_can_use_the_full_bank_outside_shortlist(self):
        shortlist = [Candidate("a", 1), Candidate("b", 1)]
        full = [*shortlist, Candidate("outside", 3)]
        scores = {"a": 2.0, "b": 1.0}
        manifest = build_controls_manifest(
            shortlist,
            1,
            repeated_score_maps(scores),
            [1],
            CONTEXT_SHA256,
            coarse_members=["outside"],
            coarse_candidates=full,
        )
        coarse = manifest["coarse_historical_control"]
        self.assertEqual(["outside"], coarse["members"])
        self.assertEqual(3, coarse["actual_cost_bytes"])
        self.assertFalse(coarse["matches_target"])

    def test_each_policy_uses_its_own_score_map(self):
        candidates = [Candidate("a", 1), Candidate("b", 1), Candidate("c", 1)]
        maps = {
            "additive_gcq": {"a": 3.0, "b": 2.0, "c": 1.0},
            "vqa_driven": {"a": 1.0, "b": 3.0, "c": 2.0},
            "maba_style_additive": {"a": 2.0, "b": 1.0, "c": 3.0},
        }
        manifest = build_controls_manifest(
            candidates, 1, maps, [0], CONTEXT_SHA256
        )
        controls = manifest["score_driven_controls"]
        self.assertEqual(["a"], controls["additive_gcq"]["members"])
        self.assertEqual(["b"], controls["vqa_driven"]["members"])
        self.assertEqual(["c"], controls["maba_style_additive"]["members"])

    def test_manifest_write_is_atomic_exclusive_and_round_trips(self):
        manifest = {
            "schema_version": 1,
            "protocol_context_sha256": CONTEXT_SHA256,
            "value": [1, 2, 3],
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "controls.json"
            digest = write_manifest_exclusive(output, manifest)
            self.assertEqual(manifest, json.loads(output.read_text()))
            self.assertEqual(64, len(digest))
            with self.assertRaisesRegex(ControlBuildError, "refusing to overwrite"):
                write_manifest_exclusive(output, manifest)
            self.assertEqual(["controls.json"], sorted(path.name for path in Path(tmp).iterdir()))

    def test_cli_binds_context_and_all_sources_into_single_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = root / "candidates.json"
            scores = root / "scores.json"
            context = root / "protocol.json"
            output = root / "controls.json"
            candidates.write_text(
                json.dumps(
                    {
                        "budget_bytes": 2,
                        "candidates": [
                            {"name": "a", "delta_bytes": 1},
                            {"name": "b", "delta_bytes": 1},
                            {"name": "c", "delta_bytes": 2},
                        ],
                    }
                )
            )
            scores.write_text(json.dumps({"a": 2.0, "b": 1.0, "c": 2.5}))
            context.write_text(json.dumps({"frozen": True}))
            argv = [
                "--candidates",
                str(candidates),
                "--target-bytes",
                "2",
                "--gcq-scores",
                str(scores),
                "--vqa-scores",
                str(scores),
                "--maba-scores",
                str(scores),
                "--random-seeds",
                "1",
                "2",
                "--protocol-context",
                str(context),
                "--out",
                str(output),
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(0, main(argv))
            manifest = json.loads(output.read_text())
            self.assertEqual(
                sha256_file(context), manifest["protocol_context_sha256"]
            )
            self.assertEqual(
                {1, 2},
                {row["seed"] for row in manifest["random_controls"]["samples"]},
            )
            self.assertEqual(
                {
                    "additive_gcq_scores",
                    "maba_style_additive_scores",
                    "projection_catalog",
                    "protocol_context",
                    "vqa_driven_scores",
                },
                set(manifest["source_sha256"]),
            )
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ControlBuildError, "refusing to overwrite"):
                    main(argv)

    def test_validation_rejects_duplicate_seeds_and_score_catalog_mismatch(self):
        candidates = [Candidate("a", 1), Candidate("b", 1)]
        scores = {"a": 1.0, "b": 2.0}
        with self.assertRaisesRegex(ControlBuildError, "seeds must be unique"):
            build_controls_manifest(
                candidates,
                1,
                repeated_score_maps(scores),
                [3, 3],
                CONTEXT_SHA256,
            )
        bad_maps = repeated_score_maps(scores)
        bad_maps["vqa_driven"] = {"a": 1.0}
        with self.assertRaisesRegex(ControlBuildError, "mismatch"):
            build_controls_manifest(
                candidates, 1, bad_maps, [3], CONTEXT_SHA256
            )


if __name__ == "__main__":
    unittest.main()
