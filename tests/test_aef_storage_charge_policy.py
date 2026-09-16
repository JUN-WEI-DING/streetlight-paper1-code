import pandas as pd
import pytest

from streetlight.aef import AEFPipeline


def _values(value, n):
    return list(value) if isinstance(value, (list, tuple, pd.Series)) else [value] * n


def _frame(index, *, aef, gen=100.0, bess=0.0, phs=0.0, **flows):
    n = len(index)
    data = {
        "AEF": _values(aef, n),
        "Total_Gen (MWh)": _values(gen, n),
        "BESS": _values(bess, n),
        "PHS": _values(phs, n),
    }
    for name, value in flows.items():
        data[name] = _values(value, n)
    return pd.DataFrame(data, index=index)


def test_base_mix_helper_returns_disconnected_island_local_aef():
    idx = pd.date_range("2024-01-01", periods=1, freq="10min")
    region_data = {
        "north": _frame(idx, aef=0.4, F_CN=0.0),
        "central": _frame(idx, aef=0.6, F_NC=0.0, F_SC=0.0, F_EC=0.0),
        "south": _frame(idx, aef=0.8, F_CS=0.0),
        "east": _frame(idx, aef=0.2, F_CE=0.0),
        "island_penghu": _frame(idx, aef=0.91),
        "island_kinmen": _frame(idx, aef=0.92),
        "island_lienchiang": _frame(idx, aef=0.93),
    }

    result = AEFPipeline().flow_calculator.compute_base_mix_aef(idx[0], region_data)

    assert result["IP"] == pytest.approx(0.91)
    assert result["IK"] == pytest.approx(0.92)
    assert result["IL"] == pytest.approx(0.93)


def test_both_storage_charge_keeps_grid_serving_aef():
    idx = pd.date_range("2024-01-01", periods=1, freq="10min")
    region_data = {
        "north": _frame(idx, aef=0.4, F_CN=10.0),
        "central": _frame(idx, aef=0.6, bess=-10.0, phs=-20.0, F_NC=0.0, F_SC=0.0, F_EC=0.0),
        "south": _frame(idx, aef=0.8, F_CS=20.0),
        "east": _frame(idx, aef=0.2, F_CE=5.0),
        "island_penghu": _frame(idx, aef=0.9),
    }

    result, _ = AEFPipeline().run(region_data=region_data)

    assert result["central"].loc[idx[0], "FLOW_UNIT_FINAL_AEF"] == pytest.approx(0.6)
    assert result["north"].loc[idx[0], "FLOW_UNIT_FINAL_AEF"] == pytest.approx(
        (0.4 * 100.0 + 0.6 * 10.0) / 110.0
    )
    assert result["south"].loc[idx[0], "FLOW_UNIT_FINAL_AEF"] == pytest.approx(
        (0.8 * 100.0 + 0.6 * 20.0) / 120.0
    )
    assert result["east"].loc[idx[0], "FLOW_UNIT_FINAL_AEF"] == pytest.approx(
        (0.2 * 100.0 + 0.6 * 5.0) / 105.0
    )
    assert result["north"]["FLOW_UNIT_FINAL_AEF"].notna().all()
    assert result["central"]["FLOW_UNIT_FINAL_AEF"].notna().all()


def test_bess_charge_during_phs_discharge_keeps_grid_serving_aef():
    idx = pd.date_range("2024-01-01", periods=2, freq="10min")
    region_data = {
        "north": _frame(idx, aef=0.4, F_CN=[0.0, 10.0]),
        "central": _frame(
            idx,
            aef=0.6,
            bess=[0.0, -5.0],
            phs=[-50.0, 10.0],
            F_NC=[0.0, 0.0],
            F_SC=[0.0, 5.0],
            F_EC=[0.0, 0.0],
        ),
        "south": _frame(idx, aef=0.8, F_CS=[0.0, 20.0]),
        "east": _frame(idx, aef=0.2, F_CE=[0.0, 5.0]),
        "island_penghu": _frame(idx, aef=0.9),
    }

    result, _ = AEFPipeline().run(region_data=region_data)

    for region in ("north", "central", "south", "east"):
        assert result[region]["FLOW_UNIT_FINAL_AEF"].notna().all()
        assert result[region]["UNIT_SRC_AEF"].notna().all()


def test_disconnected_island_local_aef_fills_all_timesteps_after_partial_storage_write():
    idx = pd.date_range("2024-01-01", periods=2, freq="10min")
    region_data = {
        "north": _frame(idx, aef=0.4, F_CN=0.0),
        "central": _frame(
            idx,
            aef=0.6,
            bess=[-10.0, 0.0],
            phs=[-20.0, 0.0],
            F_NC=0.0,
            F_SC=0.0,
            F_EC=0.0,
        ),
        "south": _frame(idx, aef=0.8, F_CS=0.0),
        "east": _frame(idx, aef=0.2, F_CE=0.0),
        "island_penghu": _frame(idx, aef=0.9),
    }

    result, _ = AEFPipeline().run(region_data=region_data)

    assert result["island_penghu"]["FLOW_UNIT_FINAL_AEF"].notna().all()
    assert result["island_penghu"]["Flow_AEF_base"].notna().all()
    assert result["island_penghu"]["Flow_UNIT_AEF_base"].notna().all()
    assert result["island_penghu"]["UNIT_SRC_AEF"].notna().all()
    assert (
        result["island_penghu"]["FLOW_UNIT_FINAL_AEF"]
        == result["island_penghu"]["AEF"]
    ).all()


def test_disconnected_island_zero_generation_timestep_keeps_local_aef_proxy():
    idx = pd.date_range("2024-01-01", periods=2, freq="10min")
    region_data = {
        "north": _frame(idx, aef=0.4, F_CN=0.0),
        "central": _frame(idx, aef=0.6, F_NC=0.0, F_SC=0.0, F_EC=0.0),
        "south": _frame(idx, aef=0.8, F_CS=0.0),
        "east": _frame(idx, aef=0.2, F_CE=0.0),
        "island_lienchiang": _frame(
            idx,
            aef=[0.88, 0.0],
            gen=[10.0, 0.0],
        ),
    }

    result, _ = AEFPipeline().run(region_data=region_data)

    assert result["island_lienchiang"].loc[idx[1], "AEF"] == pytest.approx(0.88)
    assert result["island_lienchiang"].loc[
        idx[1], "FLOW_UNIT_FINAL_AEF"
    ] == pytest.approx(0.88)
