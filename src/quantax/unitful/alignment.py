from __future__ import annotations

from quantax.core.utils import handle_different_scales
from quantax.unitful.unitful import Unitful, can_optimize_scale


def align_scales(
    u1: Unitful,
    u2: Unitful,
) -> tuple[Unitful, Unitful]:
    if u1.unit != u2.unit:
        raise Exception("Cannot align arrays with different units")

    # if scales are already aligned, do nothing
    if u1.scale == u2.scale:
        return u1, u2

    # non physical ArrayLikes need to keep scale 0
    force_zero_scale = False
    if not can_optimize_scale(u1):
        assert u1.scale == 0
        force_zero_scale = True
    if not can_optimize_scale(u2):
        assert u2.scale == 0
        force_zero_scale = True
    # calculate new scale
    if force_zero_scale:
        new_scale, factor1, factor2 = 0, 10**u1.scale, 10**u2.scale
    else:
        new_scale, factor1, factor2 = handle_different_scales(
            u1.scale,
            u2.scale,
        )
    # update unitfuls
    if new_scale != u1.scale:
        u1 = Unitful(
            val=u1.val * factor1,
            unit=u1.unit,
            scale=new_scale,
            optimize_scale=False,
        )
    if new_scale != u2.scale:
        u2 = Unitful(
            val=u2.val * factor2,
            unit=u2.unit,
            scale=new_scale,
            optimize_scale=False,
            # static_arr=None if u2.static_arr is None else u2.static_arr * factor2,
        )
    return u1, u2
