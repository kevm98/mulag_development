"""Automated tests for AAM SystemFactor pipeline."""

import os
import sys
import numpy as np
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AAM_SystemFactor.data.factor_targets import (
    compute_targets, NOMINAL_TARGET, TARGET_NAMES, NOMINAL_EPS_AIR,
    NOMINAL_EPS_VOL, NOMINAL_CD, NOMINAL_PSET, W_SEAL_SCALE, Vd_nom,
)
from AAM_SystemFactor.data.cylinder_windows import (
    build_feature_matrix, build_windows, FEATURE_NAMES, N_FEATURES,
)
from AAM_SystemFactor.data.normalization import FeatureScaler, TargetScaler
from AAM_SystemFactor.data.split_runs import (
    split_runs_deterministic, verify_no_leakage,
)
from AAM_SystemFactor.data.matlab_v73 import compute_ext_vel_causal


class TestFactorTargets(unittest.TestCase):
    """Test factor target computation (Sections 6.1-6.5)."""

    def test_nominal_target(self):
        params = {
            'eps_air': 0.005, 'eps_vol': 0.02,
            'delta_spool': 0.0, 'Pset_mult': 1.0,
            'w_seal': np.zeros(4),
        }
        t = compute_targets(params)
        np.testing.assert_allclose(t, NOMINAL_TARGET, atol=1e-10)

    def test_output_shape(self):
        t = compute_targets({'eps_air': 0.01, 'eps_vol': 0.03,
                             'delta_spool': 0.1, 'Pset_mult': 1.05,
                             'w_seal': np.array([0.1, 0.2, 0.3, 0.4])})
        self.assertEqual(t.shape, (7,))

    def test_compliance_formula(self):
        # e_comp = log(Vd_nom / Vd_i)
        eps_air, eps_vol = 0.005, 0.02
        Vd_i = 5e-4 * (1 + eps_air + eps_vol)
        expected = np.log(Vd_nom / Vd_i)
        t = compute_targets({'eps_air': eps_air, 'eps_vol': eps_vol})
        self.assertAlmostEqual(t[0], expected, places=10)

    def test_flow_formula(self):
        t = compute_targets({'delta_spool': 0.15})
        self.assertAlmostEqual(t[1], 0.15)

    def test_pressure_formula(self):
        t = compute_targets({'Pset_mult': 1.05})
        self.assertAlmostEqual(t[2], 0.05)

    def test_seal_formula(self):
        w = np.array([0.1, 0.2, 0.3, 0.4])
        t = compute_targets({'w_seal': w})
        expected = 1.0 + w / W_SEAL_SCALE
        np.testing.assert_allclose(t[3:7], expected)

    def test_beta_fixed_excluded(self):
        """Verify beta_eff is NOT part of the target (Section 6.1)."""
        self.assertNotIn('beta_eff', TARGET_NAMES)
        # Verify e_comp only uses eps_air, eps_vol
        t = compute_targets({'eps_air': 0.005, 'eps_vol': 0.02})
        self.assertAlmostEqual(t[0], 0.0)

    def test_no_duplicate_factors(self):
        """Verify delta_spool and Cd are not both targets (Section 6.2)."""
        self.assertEqual(len(TARGET_NAMES), 7)

    def test_seven_dimensional_target(self):
        t = compute_targets({})
        self.assertEqual(len(t), 7)


class TestCylinderWindows(unittest.TestCase):
    """Test window construction (Sections 5, 7)."""

    def setUp(self):
        self.T_feat = 50
        self.run_data = {
            'valve_cmd': np.random.randn(5, self.T_feat) * 0.5,
            'extension': np.random.rand(4, self.T_feat) * 0.5,
            'ext_vel': np.random.randn(4, self.T_feat) * 0.1,
            'dP': np.random.randn(4, self.T_feat) * 1e7,
            'force': np.random.randn(4, self.T_feat) * 1e6,
            'dt': 0.01,
        }

    def test_feature_matrix_shape(self):
        fm = build_feature_matrix(self.run_data)
        self.assertEqual(fm.shape, (self.T_feat, 20))

    def test_feature_ordering(self):
        """Verify exact input ordering (Section 5)."""
        fm = build_feature_matrix(self.run_data)
        # Columns 0-3: ValveCmd 1..4
        np.testing.assert_allclose(fm[:, 0], self.run_data['valve_cmd'][0, :])
        np.testing.assert_allclose(fm[:, 3], self.run_data['valve_cmd'][3, :])
        # Columns 4-7: Extension 1..4
        np.testing.assert_allclose(fm[:, 4], self.run_data['extension'][0, :])
        # Columns 16-19: Force 1..4
        np.testing.assert_allclose(fm[:, 16], self.run_data['force'][0, :])

    def test_build_windows_shape(self):
        fm = build_feature_matrix(self.run_data)
        T = 10
        windows, indices = build_windows(fm, T)
        expected = self.T_feat - T + 1
        self.assertEqual(len(windows), expected)
        self.assertEqual(windows.shape, (expected, T, 20))
        self.assertEqual(len(indices), expected)

    def test_input_shape_is_B_T_20(self):
        fm = build_feature_matrix(self.run_data)
        windows, _ = build_windows(fm, 10)
        self.assertEqual(windows.ndim, 3)
        self.assertEqual(windows.shape[1], 10)
        self.assertEqual(windows.shape[2], 20)

    def test_causal_velocity(self):
        """Test backward difference velocity computation."""
        ext = np.array([[0.0, 0.1, 0.15, 0.25, 0.30]])
        dt = 0.01
        vel = compute_ext_vel_causal(ext, dt)
        expected = np.array([[10.0, 10.0, 5.0, 10.0, 5.0]])
        np.testing.assert_allclose(vel, expected)

    def test_fifth_valve_excluded(self):
        """Verify only cylinders 1-4 are used (Section 5)."""
        self.assertEqual(N_FEATURES, 20)
        self.assertEqual(len(FEATURE_NAMES), 20)
        self.assertNotIn('ValveCmd_5', FEATURE_NAMES)


class TestNormalization(unittest.TestCase):
    """Test normalization (Section 8)."""

    def test_fit_transform(self):
        X = np.random.randn(100, 20) * 5 + 10
        scaler = FeatureScaler()
        X_scaled = scaler.fit_transform(X)
        self.assertAlmostEqual(X_scaled.mean(), 0.0, delta=0.1)
        self.assertAlmostEqual(X_scaled.std(), 1.0, delta=0.1)

    def test_round_trip(self):
        X = np.random.randn(100, 7)
        scaler = TargetScaler()
        X_scaled = scaler.fit_transform(X)
        X_back = scaler.inverse_transform(X_scaled)
        np.testing.assert_allclose(X, X_back, atol=1e-6)

    def test_training_only_fit(self):
        """Verify validation data is not used to fit scalers (Section 8)."""
        X_train = np.random.randn(50, 20) * 5 + 10
        X_val = np.random.randn(50, 20) * 8 + 30
        scaler = FeatureScaler()
        scaler.fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_val_s = scaler.transform(X_val)
        self.assertAlmostEqual(X_train_s.mean(), 0.0, delta=0.2)
        # Val should NOT be perfectly standardized
        self.assertTrue(abs(X_val_s.mean()) > 0.1)


class TestSplitRuns(unittest.TestCase):
    """Test run-level splitting (Section 7)."""

    def test_split_shape(self):
        n = 100
        train_m, val_m, test_m, assignments = split_runs_deterministic(
            run_ids=[f'r{i}' for i in range(n)],
            run_types=['Nominal'] * n,
            source_types=['closed_loop'] * n,
            n_samples=[100] * n,
            train_frac=0.80, val_frac=0.10, seed=42,
        )
        self.assertEqual(len(assignments), n)
        self.assertGreaterEqual(train_m.sum(), 1)
        self.assertGreaterEqual(val_m.sum(), 1)
        self.assertGreaterEqual(test_m.sum(), 1)

    def test_no_leakage(self):
        train_ids = set(['r1', 'r2'])
        val_ids = set(['r3'])
        test_ids = set(['r4'])
        self.assertTrue(verify_no_leakage(train_ids, val_ids, test_ids))

    def test_leakage_detected(self):
        train_ids = set(['r1', 'r2'])
        val_ids = set(['r2'])
        test_ids = set(['r3'])
        with self.assertRaises(AssertionError):
            verify_no_leakage(train_ids, val_ids, test_ids)


class TestSourceLabels(unittest.TestCase):
    """Test source-type tracking."""

    def test_source_labels(self):
        run_ids = ['ol1', 'ol2', 'ol3', 'ol4', 'cl1', 'cl2', 'cl3']
        source_types = ['open_loop'] * 4 + ['closed_loop'] * 3
        train_m, val_m, test_m, _ = split_runs_deterministic(
            run_ids, ['N'] * 7, source_types, [100] * 7, seed=42,
        )
        # Both sources should appear in train
        self.assertIn('open_loop', np.array(source_types)[train_m])
        self.assertIn('closed_loop', np.array(source_types)[train_m])


class TestFiniteOutputs(unittest.TestCase):
    """Test that models produce finite outputs."""

    def test_nominal_finite(self):
        from AAM_SystemFactor.models.baselines import NominalBaseline
        nb = NominalBaseline()
        pred = nb.predict(np.zeros((5, 10, 20)))
        self.assertTrue(np.all(np.isfinite(pred)))

    def test_aggregate_mlp_finite(self):
        import torch
        from AAM_SystemFactor.models.baselines import AggregateMLP
        model = AggregateMLP()
        model.eval()
        with torch.no_grad():
            x = torch.randn(4, 10, 20)
            out = model(x)
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_lstm_finite(self):
        import torch
        from AAM_SystemFactor.models.aam_lstm import AAMLSTM
        model = AAMLSTM()
        model.eval()
        with torch.no_grad():
            x = torch.randn(4, 10, 20)
            out = model(x)
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_lstm_output_shape(self):
        import torch
        from AAM_SystemFactor.models.aam_lstm import AAMLSTM
        model = AAMLSTM()
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(4, 10, 20))
        self.assertEqual(out.shape, (4, 7))

    def test_lstm_params_count(self):
        import torch
        from AAM_SystemFactor.models.aam_lstm import AAMLSTM
        model = AAMLSTM(input_size=20, hidden_size=64, num_layers=2, output_size=7)
        n_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(n_params, 1000)
        self.assertLess(n_params, 500000)


if __name__ == '__main__':
    unittest.main(verbosity=2)
