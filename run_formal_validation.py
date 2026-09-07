#!/usr/bin/env python3
"""Pandas datetime-unit compatibility wrapper for formal v2 validation."""
import pandas as pd
import validate_v2_hypothesis as v

_orig_merge_asof = v.pd.merge_asof


def _merge_asof_ns(left, right, *args, **kwargs):
    left = left.copy()
    right = right.copy()
    left_on = kwargs.get("left_on")
    right_on = kwargs.get("right_on")
    if left_on:
        left[left_on] = pd.to_datetime(left[left_on]).astype("datetime64[ns]")
    if right_on:
        right[right_on] = pd.to_datetime(right[right_on]).astype("datetime64[ns]")
    return _orig_merge_asof(left, right, *args, **kwargs)


v.pd.merge_asof = _merge_asof_ns
v.main()
