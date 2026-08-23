import numpy as np

import openpi.shared.normalize as normalize


def test_normalize_update():
    arr = np.arange(12).reshape(4, 3)  # 4 vectors of length 3

    stats = normalize.RunningStats()
    for i in range(len(arr)):
        stats.update(arr[i : i + 1])  # Update with one vector at a time
    results = stats.get_statistics()

    assert np.allclose(results.mean, np.mean(arr, axis=0))
    assert np.allclose(results.std, np.std(arr, axis=0))


def test_serialize_deserialize():
    stats = normalize.RunningStats()
    stats.update(np.arange(12).reshape(4, 3))  # 4 vectors of length 3

    norm_stats = {"test": stats.get_statistics()}
    norm_stats2 = normalize.deserialize_json(normalize.serialize_json(norm_stats))
    assert np.allclose(norm_stats["test"].mean, norm_stats2["test"].mean)
    assert np.allclose(norm_stats["test"].std, norm_stats2["test"].std)


def test_multiple_batch_dimensions():
    # Test with multiple batch dimensions: (2, 3, 4) where 4 is vector dimension
    batch_shape = (2, 3, 4)
    arr = np.random.rand(*batch_shape)

    stats = normalize.RunningStats()
    stats.update(arr)  # Should handle (2, 3, 4) -> reshape to (6, 4)
    results = stats.get_statistics()

    # Flatten batch dimensions and compute expected stats
    flattened = arr.reshape(-1, arr.shape[-1])  # (6, 4)
    expected_mean = np.mean(flattened, axis=0)
    expected_std = np.std(flattened, axis=0)

    assert np.allclose(results.mean, expected_mean)
    assert np.allclose(results.std, expected_std)


# ------------------------------------------------------------------------
"""Quantile stats must stay finite on dimensions that barely move.

`Normalize._normalize_quantile` divides by `q99 - q01 + 1e-6`. Robot datasets
routinely carry a dimension that holds one value for almost every frame -- a
gripper or hand command that is zero except during a grasp, a base height that
never changes -- and for those both percentiles land on the same value. The
division then scales that dimension's rare real samples by up to 1e6. Nothing
raises: the blow-up hits well under 0.1% of samples and Adam bounds the step, so
the run trains to completion and simply learns nothing on those dimensions.

See `RunningStats._widen_degenerate_quantiles`.
"""

def _accumulate(x: np.ndarray, batch: int = 1000) -> "RunningStats":
    stats = normalize.RunningStats()
    for start in range(0, len(x), batch):
        stats.update(x[start : start + batch])
    return stats


def _normalized(x: np.ndarray, stats) -> np.ndarray:
    return (x - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2 - 1


def _columns(n: int = 20_000) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.stack(
        [
            rng.normal(size=n),  # ordinary continuous dimension
            # 1% of frames carry a command; the rest are exactly zero, so both
            # percentiles collapse onto 0 while the std is carried by the tail.
            np.where(rng.random(n) < 0.01, rng.choice([-0.33, 0.5], n), 0.0),
            np.full(n, 0.72),  # never moves at all
        ],
        axis=1,
    )


def test_degenerate_dimensions_stay_bounded():
    x = _columns()
    z = _normalized(x, _accumulate(x).get_statistics())

    # The sparse dimension is rescaled to its observed range, so everything that
    # was ever seen lands inside [-1, 1]; the constant one normalizes to ~0.
    assert np.abs(z[:, 1]).max() <= 1.0 + 1e-6
    assert np.abs(z[:, 2]).max() < 0.1


def test_ordinary_dimension_keeps_quantile_scaling():
    """The guard must not fire on a dimension that has a scale.

    For any distribution with real spread the 1-99% range is wide compared to
    one standard deviation -- 4.65 sigma for a Gaussian -- so `q99 - q01 < std`
    is only true once the mass has collapsed onto a point.
    """
    x = _columns()
    stats = _accumulate(x).get_statistics()

    # ~2.33 sigma each side for a standard normal.
    assert 2.0 < stats.q99[0] < 2.7
    assert -2.7 < stats.q01[0] < -2.0
    z = _normalized(x, stats)
    # Outliers past the 99th percentile still exceed 1, which is the point of
    # quantile normalization -- they are just not scaled by 1e6.
    assert 1.0 < np.abs(z[:, 0]).max() < 5.0


def test_guard_is_what_keeps_it_bounded():
    """The bound above is the guard's doing, not the data's.

    How far the raw percentiles blow up depends on how wide the histogram bins
    are, which depends on the dimension's min and max: on MHBench's real action
    columns it reaches 1e6, on this synthetic one a few thousand. Either way it
    is orders of magnitude, so assert the ratio rather than an absolute number.
    """
    x = _columns()
    stats = _accumulate(x)
    q01, q99 = stats._compute_quantiles([0.01, 0.99])  # noqa: SLF001 - the unguarded values
    unguarded = np.abs((x[:, 1] - q01[1]) / (q99[1] - q01[1] + 1e-6) * 2 - 1).max()
    guarded = np.abs(_normalized(x, stats.get_statistics())[:, 1]).max()
    assert unguarded > 1000 * guarded
