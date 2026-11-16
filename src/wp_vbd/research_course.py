"""Explicit replay frequency for newly added data in a fresh research course."""


def sampling_rows(rows, policy=None):
    """Retain every existing row, then repeat only declared added datasets.

    Repeated samples are optimizer visits, never additional datasets or demos.
    Courses without this explicit policy keep their original exact ordering.
    """
    if policy is None:
        return rows
    if (not isinstance(policy, dict) or
            set(policy) != {'schema', 'new_dataset_ids', 'repeats'} or
            policy['schema'] != 'fresh_research_replay_v1'):
        raise ValueError('Explicit fresh research replay policy required')
    by_id = {row['dataset_id']: row for row in rows}
    added = policy['new_dataset_ids']
    repeats = policy['repeats']
    if (len(by_id) != len(rows) or not isinstance(added, list) or not added or
            any(not isinstance(key, str) for key in added) or
            len(set(added)) != len(added) or not set(added) <= set(by_id) or
            not isinstance(repeats, dict) or set(repeats) != set(added)):
        raise ValueError('Replay must identify distinct existing added datasets')
    if any(type(value) is not int or value < 1 for value in repeats.values()):
        raise ValueError('Positive integer replay frequencies required')
    result = list(rows)
    for key in added:
        result.extend([by_id[key]] * (repeats[key] - 1))
    return result
