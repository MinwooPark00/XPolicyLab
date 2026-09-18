import unittest

import numpy as np

from policy.LingBot_VA.rollout_contract import cache_action_chunk, cache_keyframes


class RolloutContractTest(unittest.TestCase):
    def setUp(self):
        # Each chronological action is its own index, repeated over channels.
        chronological = np.arange(80, dtype=np.float32)[:, None] + np.arange(3)[None, :] * 1000
        self.raw = chronological.reshape(4, 20, 3).transpose(2, 0, 1)

    def test_first_update_keeps_condition_and_executed_frame(self):
        selected = cache_action_chunk(
            self.raw,
            skipped_steps=20,
            executed_steps=20,
            action_per_frame=20,
            include_skipped_prefix=True,
        )
        self.assertEqual(selected.shape, (3, 2, 20))
        np.testing.assert_array_equal(selected, self.raw[:, :2])

    def test_later_update_discards_unexecuted_prediction(self):
        selected = cache_action_chunk(
            self.raw[:, :2],
            skipped_steps=0,
            executed_steps=20,
            action_per_frame=20,
            include_skipped_prefix=False,
        )
        self.assertEqual(selected.shape, (3, 1, 20))
        np.testing.assert_array_equal(selected, self.raw[:, :1])

    def test_partial_action_frame_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete action/video frames"):
            cache_action_chunk(
                self.raw,
                skipped_steps=0,
                executed_steps=24,
                action_per_frame=20,
                include_skipped_prefix=False,
            )

    def test_keyframes_include_post_final_action_observation(self):
        frames = cache_keyframes(
            list(range(19)),
            20,
            executed_steps=20,
            keyframe_stride=5,
        )
        self.assertEqual(frames, [4, 9, 14, 20])


if __name__ == "__main__":
    unittest.main()
