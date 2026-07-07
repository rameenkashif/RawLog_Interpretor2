"""
test_petrophysics.py
---------------------
Unit tests for every calculation in app/petrophysics.py (brief section 3).

Run with:
    cd backend
    pytest tests/test_petrophysics.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from app import petrophysics as pp


# -----------------------------------------------------------------------------
# 3.1 VSH
# -----------------------------------------------------------------------------
class TestVSH:
    def test_clean_sand_has_low_vsh(self, synthetic_well_df, config):
        vsh = pp.compute_vsh(synthetic_well_df, config)
        clean_zone_vsh = vsh.iloc[0:60]
        assert clean_zone_vsh.mean() < 0.15

    def test_shale_zone_has_high_vsh(self, synthetic_well_df, config):
        vsh = pp.compute_vsh(synthetic_well_df, config)
        shale_zone_vsh = vsh.iloc[60:120]
        assert shale_zone_vsh.mean() > 0.7

    def test_vsh_is_clipped_to_0_1(self, synthetic_well_df, config):
        vsh = pp.compute_vsh(synthetic_well_df, config)
        assert vsh.min() >= 0.0
        assert vsh.max() <= 1.0

    def test_tertiary_method_gives_different_result(self, synthetic_well_df, config):
        older_cfg = {**config, "vsh": {**config["vsh"], "method": "older"}}
        tertiary_cfg = {**config, "vsh": {**config["vsh"], "method": "tertiary"}}
        vsh_older = pp.compute_vsh(synthetic_well_df, older_cfg)
        vsh_tertiary = pp.compute_vsh(synthetic_well_df, tertiary_cfg)
        assert not np.allclose(vsh_older, vsh_tertiary)

    def test_manual_formula_matches_older_larionov(self, config):
        df = pd.DataFrame({"GR": [20.0, 40.0, 60.0, 80.0, 100.0]})
        cfg = {
            **config,
            "vsh": {
                **config["vsh"],
                "use_percentiles": False,
                "gr_clean_override": 20.0,
                "gr_shale_override": 100.0,
                "method": "older",
            },
        }
        vsh = pp.compute_vsh(df, cfg)
        igr = (df["GR"] - 20.0) / (100.0 - 20.0)
        expected = np.clip(0.33 * (2 ** (2 * igr) - 1), 0, 1)
        np.testing.assert_allclose(vsh.to_numpy(), expected.to_numpy(), rtol=1e-6)


# -----------------------------------------------------------------------------
# 3.2 PHIT
# -----------------------------------------------------------------------------
class TestPHIT:
    def test_formula_matches_manual_calculation(self, config):
        df = pd.DataFrame({"RHOB": [2.65, 2.4, 2.0, 1.0]})
        phit = pp.compute_phit(df, config)
        expected = (2.65 - df["RHOB"]) / (2.65 - 1.0)
        np.testing.assert_allclose(
            phit.to_numpy(), np.clip(expected, 0, 0.5), rtol=1e-6
        )

    def test_matrix_density_at_rhob_gives_zero_porosity(self, config):
        df = pd.DataFrame({"RHOB": [2.65]})
        phit = pp.compute_phit(df, config)
        assert phit.iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_limestone_matrix_override(self, config):
        cfg = {**config, "phit": {**config["phit"], "rhob_matrix": 2.71}}
        df = pd.DataFrame({"RHOB": [2.3]})
        phit = pp.compute_phit(df, cfg)
        expected = (2.71 - 2.3) / (2.71 - 1.0)
        assert phit.iloc[0] == pytest.approx(expected, rel=1e-6)

    def test_high_porosity_sand_zone_greater_than_shale_zone(
        self, synthetic_well_df, config
    ):
        phit = pp.compute_phit(synthetic_well_df, config)
        assert phit.iloc[0:60].mean() > phit.iloc[60:120].mean()


# -----------------------------------------------------------------------------
# 3.3 PHIE
# -----------------------------------------------------------------------------
class TestPHIE:
    def test_phie_never_exceeds_phit(self, synthetic_well_df, config):
        vsh = pp.compute_vsh(synthetic_well_df, config)
        phit = pp.compute_phit(synthetic_well_df, config)
        phie = pp.compute_phie(synthetic_well_df, vsh, phit, config)
        assert (phie <= phit + 1e-9).all()

    def test_phie_non_negative(self, synthetic_well_df, config):
        vsh = pp.compute_vsh(synthetic_well_df, config)
        phit = pp.compute_phit(synthetic_well_df, config)
        phie = pp.compute_phie(synthetic_well_df, vsh, phit, config)
        assert (phie >= 0).all()

    def test_zero_vsh_means_phie_equals_phit(self, config):
        df = pd.DataFrame({"RHOB": [2.3, 2.2]})
        phit = pp.compute_phit(df, config)
        vsh = pd.Series([0.0, 0.0])
        phie = pp.compute_phie(df, vsh, phit, config)
        np.testing.assert_allclose(phie.to_numpy(), phit.to_numpy(), rtol=1e-6)

    def test_density_neutron_crosscheck_shape(self, synthetic_well_df, config):
        phit = pp.compute_phit(synthetic_well_df, config)
        phie_dn = pp.compute_phie_density_neutron(synthetic_well_df, phit, config)
        assert len(phie_dn) == len(synthetic_well_df)
        assert (phie_dn >= 0).all()
        assert (phie_dn <= 0.5).all()


# -----------------------------------------------------------------------------
# 3.4 SWE (Archie)
# -----------------------------------------------------------------------------
class TestSWE:
    def test_hydrocarbon_zone_has_lower_sw_than_wet_zone(
        self, synthetic_well_df, config
    ):
        vsh = pp.compute_vsh(synthetic_well_df, config)
        phit = pp.compute_phit(synthetic_well_df, config)
        phie = pp.compute_phie(synthetic_well_df, vsh, phit, config)
        swe = pp.compute_swe(synthetic_well_df, phie, config)

        wet_zone_sw = swe.iloc[0:60].mean()
        hc_zone_sw = swe.iloc[120:200].mean()
        assert hc_zone_sw < wet_zone_sw

    def test_swe_clipped_to_0_1(self, synthetic_well_df, config):
        vsh = pp.compute_vsh(synthetic_well_df, config)
        phit = pp.compute_phit(synthetic_well_df, config)
        phie = pp.compute_phie(synthetic_well_df, vsh, phit, config)
        swe = pp.compute_swe(synthetic_well_df, phie, config)
        assert swe.min() >= 0.0
        assert swe.max() <= 1.0

    def test_manual_archie_formula(self, config):
        df = pd.DataFrame({"RESISTIVITY": [50.0]})
        phie = pd.Series([0.2])
        cfg = {**config, "swe": {"a": 1.0, "m": 2.0, "n": 2.0, "rw": 0.05}}
        swe = pp.compute_swe(df, phie, cfg)
        expected = ((1.0 * 0.05) / (0.2**2 * 50.0)) ** (1 / 2.0)
        assert swe.iloc[0] == pytest.approx(min(expected, 1.0), rel=1e-6)

    def test_zero_porosity_gives_sw_of_one(self, config):
        df = pd.DataFrame({"RESISTIVITY": [50.0]})
        phie = pd.Series([0.0])
        swe = pp.compute_swe(df, phie, config)
        assert swe.iloc[0] == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# 3.5 DPTM
# -----------------------------------------------------------------------------
class TestDPTM:
    def test_dptm_is_monotonically_increasing(
        self, synthetic_well_df, config, step_depth
    ):
        dptm = pp.compute_dptm(synthetic_well_df, config, step_depth)
        diffs = np.diff(dptm.to_numpy())
        assert (diffs >= 0).all()

    def test_dptm_disabled_returns_nan(self, synthetic_well_df, config, step_depth):
        cfg = {**config, "dptm": {"enabled": False}}
        dptm = pp.compute_dptm(synthetic_well_df, cfg, step_depth)
        assert dptm.isna().all()

    def test_manual_first_increment(self, config):
        df = pd.DataFrame({"DT": [100.0, 100.0]})
        step = 0.1524
        dptm = pp.compute_dptm(df, config, step)
        expected_increment = 100.0 * step / (2 * 3.28084 * 1e6)
        assert dptm.iloc[0] == pytest.approx(expected_increment, rel=1e-6)


# -----------------------------------------------------------------------------
# 3.6 MD / TVD
# -----------------------------------------------------------------------------
class TestMDTVD:
    def test_vertical_well_md_equals_tvd_equals_dept(self, synthetic_well_df):
        result = pp.compute_md_tvd(synthetic_well_df, deviation_survey=None)
        np.testing.assert_array_equal(
            result["MD"].to_numpy(), synthetic_well_df["DEPT"].to_numpy()
        )
        np.testing.assert_array_equal(
            result["TVD"].to_numpy(), synthetic_well_df["DEPT"].to_numpy()
        )

    def test_deviation_survey_raises_not_implemented(self, synthetic_well_df):
        survey = pd.DataFrame({"MD": [1000], "INCL": [5], "AZIM": [30]})
        with pytest.raises(NotImplementedError):
            pp.compute_md_tvd(synthetic_well_df, deviation_survey=survey)


# -----------------------------------------------------------------------------
# 3.7 PERM_TIXIER
# -----------------------------------------------------------------------------
class TestPermTixier:
    def test_higher_phie_gives_higher_perm(self, config):
        df = pd.DataFrame({"dummy": [0, 0]})
        phie = pd.Series([0.05, 0.25])
        swe = pd.Series([0.5, 0.3])
        cfg = {**config, "perm_tixier": {"swirr": 0.25}}
        k = pp.compute_perm_tixier(df, phie, swe, cfg)
        assert k.iloc[1] > k.iloc[0]

    def test_manual_formula(self, config):
        df = pd.DataFrame({"dummy": [0]})
        phie = pd.Series([0.2])
        swe = pd.Series([0.3])
        cfg = {**config, "perm_tixier": {"swirr": 0.25}}
        k = pp.compute_perm_tixier(df, phie, swe, cfg)
        expected = 250.0 * ((0.2**3) / 0.25) ** 2
        assert k.iloc[0] == pytest.approx(expected, rel=1e-6)

    def test_auto_swirr_estimation_from_cleanest_interval(
        self, synthetic_well_df, config
    ):
        vsh = pp.compute_vsh(synthetic_well_df, config)
        phit = pp.compute_phit(synthetic_well_df, config)
        phie = pp.compute_phie(synthetic_well_df, vsh, phit, config)
        swe = pp.compute_swe(synthetic_well_df, phie, config)
        cfg = {**config, "perm_tixier": {"swirr": None, "swirr_default": 0.25}}
        k = pp.compute_perm_tixier(synthetic_well_df, phie, swe, cfg)
        assert (k >= 0).all()
        assert len(k) == len(synthetic_well_df)


# -----------------------------------------------------------------------------
# 3.8 CORE_PERM_PRED
# -----------------------------------------------------------------------------
class TestCorePermPred:
    def test_train_and_predict_round_trip(self, synthetic_well_df, config):
        interp = pp.run_full_interpretation(
            synthetic_well_df, config, step_depth=0.1524
        )
        model = pp.train_core_perm_model(interp, config)
        pred = pp.predict_core_perm(interp, model)

        assert len(pred) == len(interp)
        assert (pred.dropna() >= 0).all()

    def test_save_and_load_model(self, synthetic_well_df, config, tmp_path):
        interp = pp.run_full_interpretation(
            synthetic_well_df, config, step_depth=0.1524
        )
        model = pp.train_core_perm_model(interp, config)

        model_path = tmp_path / "core_perm_model.joblib"
        pp.save_model(model, model_path)
        loaded = pp.load_model(model_path)

        pred_original = pp.predict_core_perm(interp, model)
        pred_loaded = pp.predict_core_perm(interp, loaded)
        np.testing.assert_allclose(
            pred_original.dropna().to_numpy(),
            pred_loaded.dropna().to_numpy(),
            rtol=1e-6,
        )


# -----------------------------------------------------------------------------
# 3.9 VVOLC
# -----------------------------------------------------------------------------
class TestVVolc:
    def test_high_density_low_neutron_low_gr_flagged(self, config):
        df = pd.DataFrame(
            {
                "RHOB": [2.8, 2.2],
                "NPHI": [0.05, 0.25],
                "GR": [10.0, 100.0],
            }
        )
        vsh = pd.Series([0.0, 0.9])
        vvolc = pp.compute_vvolc(df, vsh, config)
        assert vvolc.iloc[0] > 0
        assert vvolc.iloc[1] == 0

    def test_vvolc_bounded_0_1(self, synthetic_well_df, config):
        vsh = pp.compute_vsh(synthetic_well_df, config)
        vvolc = pp.compute_vvolc(synthetic_well_df, vsh, config)
        assert vvolc.min() >= 0.0
        assert vvolc.max() <= 1.0


# -----------------------------------------------------------------------------
# 3.10 ZONES
# -----------------------------------------------------------------------------
class TestZones:
    def test_hydrocarbon_sand_classified_as_pay(self, synthetic_well_df, config):
        interp = pp.run_full_interpretation(
            synthetic_well_df, config, step_depth=0.1524
        )
        hc_zone_codes = interp["ZONES"].iloc[120:200]
        # Most of the clean hydrocarbon sand should be classified as Pay
        assert (hc_zone_codes == pp.ZONE_PAY).mean() > 0.5

    def test_shale_classified_as_non_reservoir(self, synthetic_well_df, config):
        interp = pp.run_full_interpretation(
            synthetic_well_df, config, step_depth=0.1524
        )
        shale_zone_codes = interp["ZONES"].iloc[60:120]
        assert (shale_zone_codes == pp.ZONE_NON_RESERVOIR).mean() > 0.8

    def test_zone_labels_match_codes(self, synthetic_well_df, config):
        vsh = pp.compute_vsh(synthetic_well_df, config)
        phit = pp.compute_phit(synthetic_well_df, config)
        phie = pp.compute_phie(synthetic_well_df, vsh, phit, config)
        swe = pp.compute_swe(synthetic_well_df, phie, config)
        zones_df = pp.compute_zones(synthetic_well_df, vsh, phie, swe, config)

        for code, label in zip(zones_df["ZONES"], zones_df["ZONES_LABEL"]):
            assert pp.ZONE_LABELS[code] == label

    def test_custom_cutoffs_change_classification(self, synthetic_well_df, config):
        strict_cfg = {
            **config,
            "zones": {
                "vsh_max": 0.05,  # unrealistically strict -> almost nothing qualifies
                "phie_min": 0.3,
                "swe_reservoir_max": 0.1,
                "swe_pay_max": 0.05,
            },
        }
        interp_default = pp.run_full_interpretation(
            synthetic_well_df, config, step_depth=0.1524
        )
        interp_strict = pp.run_full_interpretation(
            synthetic_well_df, strict_cfg, step_depth=0.1524
        )

        pay_default = (interp_default["ZONES"] == pp.ZONE_PAY).sum()
        pay_strict = (interp_strict["ZONES"] == pp.ZONE_PAY).sum()
        assert pay_strict < pay_default


# -----------------------------------------------------------------------------
# Full pipeline orchestration
# -----------------------------------------------------------------------------
class TestFullInterpretation:
    def test_all_expected_columns_present(self, synthetic_well_df, config):
        interp = pp.run_full_interpretation(
            synthetic_well_df, config, step_depth=0.1524
        )
        expected_cols = {
            "DEPT",
            "GR",
            "RESISTIVITY",
            "RHOB",
            "NPHI",
            "DT",
            "MD",
            "TVD",
            "VSH",
            "PHIT",
            "PHIE",
            "PHIE_DN",
            "SWE",
            "DPTM",
            "PERM_TIXIER",
            "VVOLC",
            "ZONES",
            "ZONES_LABEL",
        }
        assert expected_cols.issubset(set(interp.columns))

    def test_no_unexpected_nans_in_core_curves(self, synthetic_well_df, config):
        interp = pp.run_full_interpretation(
            synthetic_well_df, config, step_depth=0.1524
        )
        for col in ["VSH", "PHIT", "PHIE", "SWE", "PERM_TIXIER", "ZONES"]:
            assert not interp[col].isna().any(), f"{col} has unexpected NaNs"

    def test_pipeline_with_core_perm_model(self, synthetic_well_df, config):
        interp = pp.run_full_interpretation(
            synthetic_well_df, config, step_depth=0.1524
        )
        model = pp.train_core_perm_model(interp, config)
        interp_with_model = pp.run_full_interpretation(
            synthetic_well_df, config, step_depth=0.1524, core_perm_model=model
        )
        assert "CORE_PERM_PRED" in interp_with_model.columns
