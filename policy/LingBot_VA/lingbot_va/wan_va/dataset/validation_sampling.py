"""Deterministic task/role coverage for LingBot validation."""

from __future__ import annotations


def _records(dataset):
    """Yield ``(global_index, metadata)`` without loading episode tensors."""
    if hasattr(dataset, "_datasets"):
        offset = 0
        for child in dataset._datasets:
            for local_index, metadata in enumerate(child.new_metas):
                yield offset + local_index, metadata
            offset += len(child)
        return
    for index, metadata in enumerate(dataset.new_metas):
        yield index, metadata


def stratified_validation_selection(dataset, requested_samples: int):
    """Select at least one episode for every unique task/role instruction.

    CoHuB flattens each two-robot demonstration into two role episodes. Their
    role-specific instructions are the stable labels available in the LeRobot
    metadata, so grouping on ``tasks`` covers all eight tasks and both roles
    without relying on dataset ordering. The middle episode in each group is
    chosen to avoid making validation synonymous with either chronological
    endpoint. Extra requested samples are spread deterministically over the
    remaining held-out episodes.
    """
    groups = {}
    metadata_by_index = {}
    for index, metadata in _records(dataset):
        key = tuple(metadata.get("tasks") or ("<missing instruction>",))
        groups.setdefault(key, []).append(index)
        metadata_by_index[index] = metadata
    if not groups:
        return [], []

    selected = {indices[len(indices) // 2] for indices in groups.values()}
    target = min(len(dataset), max(int(requested_samples), len(selected)))
    remaining = [index for index in range(len(dataset)) if index not in selected]
    need = target - len(selected)
    if need:
        positions = ([len(remaining) // 2] if need == 1 else [
            round(i * (len(remaining) - 1) / (need - 1)) for i in range(need)
        ])
        selected.update(remaining[position] for position in positions)

    indices = sorted(selected)
    records = [{
        "dataset_index": index,
        "episode_index": metadata_by_index[index].get("episode_index"),
        "tasks": list(metadata_by_index[index].get("tasks") or []),
    } for index in indices]
    return indices, records
