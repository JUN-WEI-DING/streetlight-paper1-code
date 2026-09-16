"""Paper 1 life-cycle assessment helpers."""

__all__ = ["compute_lca_at_knee"]


def compute_lca_at_knee(pv_factor: float, batt_factor: float) -> dict:
    from streetlight.lca.hardware import compute_lca_at_knee as _compute_lca_at_knee

    return _compute_lca_at_knee(pv_factor, batt_factor)
