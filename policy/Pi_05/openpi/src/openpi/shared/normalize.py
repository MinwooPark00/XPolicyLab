import json
import logging
import pathlib

import numpy as np
import numpydantic
import pydantic

logger = logging.getLogger(__name__)

MIN_QUANTILE_WIDTH = 1e-4
"""Absolute floor for `q99 - q01`, for a dimension that never varies at all.

`Normalize._normalize_quantile` adds only 1e-6 to the denominator, so a constant
dimension would otherwise scale by ~1e6. See
`RunningStats._widen_degenerate_quantiles`.
"""


@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None  # 1st quantile
    q99: numpydantic.NDArray | None = None  # 99th quantile


class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    def update(self, batch: np.ndarray) -> None:
        """
        Update the running statistics with a batch of vectors.

        Args:
            vectors (np.ndarray): An array where all dimensions except the last are batch dimensions.
        """
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    def get_statistics(self, *, min_quantile_width: float = MIN_QUANTILE_WIDTH) -> NormStats:
        """
        Compute and return the statistics of the vectors processed so far.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        q01, q99 = self._widen_degenerate_quantiles(q01, q99, stddev, min_quantile_width)
        return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99)

    def _widen_degenerate_quantiles(self, q01, q99, stddev, min_width: float):
        """Keep quantile normalization finite on dimensions that barely move.

        `Normalize._normalize_quantile` divides by `q99 - q01 + 1e-6`. A
        dimension that holds one value for almost every frame -- a command that
        is zero except during the few steps of a grasp, a base height that never
        changes -- has both percentiles land on that value, and the division then
        multiplies its rare non-zero samples by up to 1e6. Training does not
        crash: the blow-up hits well under 0.1% of samples and Adam's 1/sqrt(v)
        bounds the step. It simply learns nothing on those dimensions, which on a
        humanoid are the grasp and locomotion commands.

        The test is `q99 - q01 < stddev`, not a fixed threshold. Percentiles are
        a *scale*, and for any distribution that has one they are wide compared
        to the standard deviation -- 4.65 sigma for a Gaussian. It is only when
        the mass collapses onto a point, leaving the standard deviation to be
        carried entirely by rare excursions, that the range falls below one
        sigma. On MHBench the separation is total: the flagged dimensions score
        exactly 0.0 and the rest never fall below 1.4.

        Such a dimension's honest scale is the range it was actually seen to
        take, so use [min, max], which maps every observed sample into [-1, 1].
        A dimension with no range at all has no scale either, so widen it
        symmetrically -- that normalizes it to exactly 0 rather than dividing by
        ~1e-6.
        """
        degenerate = (q99 - q01) < stddev
        if np.any(degenerate):
            q01 = np.where(degenerate, self._min, q01)
            q99 = np.where(degenerate, self._max, q99)

        constant = (q99 - q01) < min_width
        if np.any(constant):
            q01 = np.where(constant, q01 - min_width / 2, q01)
            q99 = np.where(constant, q99 + min_width / 2, q99)

        if np.any(degenerate) or np.any(constant):
            logger.warning(
                "Rescaled %d of %d quantile ranges to [min, max] because they were narrower than one "
                "standard deviation (indices %s); %d of those have no range at all. Left alone they "
                "would normalize to ~1e6.",
                int(degenerate.sum()),
                degenerate.size,
                np.flatnonzero(degenerate).tolist(),
                int(constant.sum()),
            )
        return q01, q99

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize the running statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the running statistics from a JSON string."""
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """Save the normalization stats to a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())
