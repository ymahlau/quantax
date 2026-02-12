from quantax import Unit, Unitful
from quantax.core.utils import handle_different_scales
from quantax.unitful.unitful import can_optimize_scale


def align_scales(
    u1: Unitful,
    u2: Unitful,
) -> tuple[Unitful, Unitful]:
    if u1.unit.dim != u2.unit.dim:
        raise Exception("Cannot align arrays with different units")
    # non physical ArrayLikes need to keep scale 0
    force_zero_scale = False
    if not u1.optimize_scale or not can_optimize_scale(u1):
        assert u1.unit.scale == 0
        force_zero_scale = True
    if not u2.optimize_scale or not can_optimize_scale(u2):
        assert u2.unit.scale == 0
        force_zero_scale = True
    # calculate new scale
    if force_zero_scale:
        new_scale, factor1, factor2 = 0, 10**u1.unit.scale, 10**u2.unit.scale
    else:
        new_scale, factor1, factor2 = handle_different_scales(
            u1.unit.scale,
            u2.unit.scale,
        )
    # update unitfuls
    if new_scale != u1.unit.scale:
        u1 = Unitful(
            val=u1.val * factor1,
            unit=Unit(scale=new_scale, dim=u1.unit.dim),
            optimize_scale=False,
            # static_arr=None if u1.static_arr is None else u1.static_arr * factor1,
        )
    if new_scale != u2.unit.scale:
        u2 = Unitful(
            val=u2.val * factor2,
            unit=Unit(scale=new_scale, dim=u2.unit.dim),
            optimize_scale=False,
            # static_arr=None if u2.static_arr is None else u2.static_arr * factor2,
        )
    return u1, u2
