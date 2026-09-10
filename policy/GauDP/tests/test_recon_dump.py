import json

import pytest

np = pytest.importorskip("numpy")

from XPolicyLab.policy.GauDP.gaudp import recon_dump


def test_fractions_are_sorted_deduplicated_and_bounded():
    assert recon_dump.parse_fractions("0.5, 0, 0.5,0.25") == (0.0, 0.25, 0.5)
    with pytest.raises(ValueError):
        recon_dump.parse_fractions("1.0")
    with pytest.raises(ValueError):
        recon_dump.parse_fractions("")


def test_keyframe_positions_index_the_concatenated_episodes():
    # Two validation episodes out of a longer dataset: the dataset concatenates
    # them, so episode 7 starts at position 0 and episode 9 at position 100.
    keyframes = recon_dump.select_keyframes(
        [(300, 400), (900, 1100)],
        fractions=(0.0, 0.5),
        episode_ids=[7, 9],
    )
    assert [(kf.position, kf.episode_id, kf.frame) for kf in keyframes] == [
        (0, 7, 0),
        (50, 7, 50),
        (100, 9, 0),
        (200, 9, 100),
    ]


def test_short_episodes_neither_collide_nor_run_past_the_end():
    keyframes = recon_dump.select_keyframes([(0, 2)], fractions=(0.0, 0.25, 0.5, 0.75))
    # 4 fractions, 2 frames: the duplicates collapse and nothing lands on frame 2.
    assert [kf.frame for kf in keyframes] == [0, 1]


def test_empty_episodes_are_skipped_without_shifting_later_positions():
    keyframes = recon_dump.select_keyframes([(0, 0), (5, 15)], fractions=(0.0,), episode_ids=[3, 4])
    assert [(kf.position, kf.episode_id) for kf in keyframes] == [(0, 4)]


def test_ground_truth_and_render_share_one_depth_scale():
    depth = np.full((2, 2), 4.0, dtype=np.float32)
    depth[0, 0] = 0.0  # invalid
    grey = recon_dump._depth_to_uint8(depth, near=2.0, far=6.0)
    assert grey.shape == (2, 2, 3)
    assert grey[0, 0, 0] == 0  # invalid pixels are the only black
    assert grey[1, 1, 0] == pytest.approx(20 + 0.5 * 235, abs=1)
    # The same metric depth must map to the same grey in either column.
    assert (recon_dump._depth_to_uint8(depth, near=2.0, far=6.0) == grey).all()


def test_grid_is_views_by_four_tiles_plus_a_label_strip():
    views, height, width = 2, 8, 6
    images = np.zeros((views, 3, height, width), dtype=np.float32)
    depth = np.ones((views, height, width), dtype=np.float32)
    grid = recon_dump.render_grid(
        images=images,
        depth=depth,
        rendered_rgb=images,
        rendered_depth=depth,
        near=0.5,
        far=2.0,
        labels=False,
    )
    assert grid.shape == (views * height, 4 * width, 3)
    labelled = recon_dump.render_grid(
        images=images,
        depth=depth,
        rendered_rgb=images,
        rendered_depth=depth,
        near=0.5,
        far=2.0,
    )
    assert labelled.shape[0] == views * height + recon_dump._LABEL_HEIGHT


def test_saved_keyframe_records_its_own_psnr(tmp_path):
    pytest.importorskip("PIL")
    views, height, width = 2, 4, 4
    images = np.zeros((views, 3, height, width), dtype=np.float32)
    rendered = np.full((views, 3, height, width), 0.1, dtype=np.float32)
    depth = np.ones((views, height, width), dtype=np.float32)
    keyframe = recon_dump.KeyFrame(position=12, episode_id=3, frame=12, length=48)
    record = recon_dump.save_keyframe(
        tmp_path,
        keyframe,
        images=images,
        depth=depth,
        rendered_rgb=rendered,
        rendered_depth=depth,
        near=0.5,
        far=2.0,
    )
    assert (tmp_path / "ep0003_f00012.png").is_file()
    assert record["file"] == "ep0003_f00012.png"
    assert record["fraction"] == pytest.approx(0.25)
    assert record["rgb_mse"] == pytest.approx(0.01)
    assert record["psnr"] == pytest.approx(20.0)

    manifest = recon_dump.write_manifest(tmp_path, [record])
    assert [json.loads(line) for line in manifest.read_text().splitlines()] == [record]
