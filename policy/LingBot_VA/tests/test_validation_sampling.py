import unittest

from policy.LingBot_VA.lingbot_va.wan_va.dataset.validation_sampling import (
    stratified_validation_selection,
)


class _Child:
    def __init__(self, metas):
        self.new_metas = metas

    def __len__(self):
        return len(self.new_metas)


class _Multi:
    def __init__(self, children):
        self._datasets = children

    def __len__(self):
        return sum(map(len, self._datasets))


class ValidationSamplingTest(unittest.TestCase):
    def setUp(self):
        metas = []
        for role in ("task-a/robot-a", "task-a/robot-b", "task-b/robot-a", "task-b/robot-b"):
            metas.extend({"episode_index": ep, "tasks": [role]} for ep in range(10))
        self.dataset = _Multi([_Child(metas[:20]), _Child(metas[20:])])

    def test_requested_count_is_raised_to_cover_every_role(self):
        indices, rows = stratified_validation_selection(self.dataset, 2)
        self.assertEqual(len(indices), 4)
        self.assertEqual({row["tasks"][0] for row in rows}, {
            "task-a/robot-a", "task-a/robot-b", "task-b/robot-a", "task-b/robot-b",
        })

    def test_selection_is_deterministic_and_can_add_samples(self):
        first = stratified_validation_selection(self.dataset, 8)
        second = stratified_validation_selection(self.dataset, 8)
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 8)


if __name__ == "__main__":
    unittest.main()
