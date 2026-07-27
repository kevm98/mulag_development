"""Compute seven physically meaningful hydraulic system factors from run parameters."""

import numpy as np
from typing import Dict

# Physical constants
BETA_0 = 0.7e9
NOMINAL_Vd = 5e-4
NOMINAL_EPS_AIR = 0.005
NOMINAL_EPS_VOL = 0.020
NOMINAL_CD = 0.7
NOMINAL_PSET = 27.6e6
W_SEAL_SCALE = 0.25

Vd_nom = NOMINAL_Vd * (1 + NOMINAL_EPS_AIR + NOMINAL_EPS_VOL)

NOMINAL_TARGET = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

TARGET_NAMES = [
    'e_comp',
    'e_flow',
    'e_pressure',
    'e_seal_1',
    'e_seal_2',
    'e_seal_3',
    'e_seal_4',
]


def compute_targets(params: Dict) -> np.ndarray:
    """Compute the 7-D factor target vector from run parameters.

    Args:
        params: dict with keys eps_air, eps_vol, delta_spool,
                Pset_mult, w_seal (array of 4).

    Returns:
        e: (7,) numpy array [e_comp, e_flow, e_pressure, e_seal_1..4]
    """
    eps_air = float(params.get('eps_air', NOMINAL_EPS_AIR))
    eps_vol = float(params.get('eps_vol', NOMINAL_EPS_VOL))
    delta_spool = float(params.get('delta_spool', 0.0))
    Pset_mult = float(params.get('Pset_mult', 1.0))
    w_seal = np.asarray(params.get('w_seal', np.zeros(4))).flatten()

    # 6.1 Compliance: e_comp = log(Vd_nom / Vd_i)
    Vd_i = NOMINAL_Vd * (1.0 + eps_air + eps_vol)
    Vd_i = np.clip(Vd_i, NOMINAL_Vd, 1e-3)
    e_comp = np.log(Vd_nom / Vd_i)

    # 6.2 Valve flow: e_flow = Cd_i / 0.7 - 1 = delta_spool
    e_flow = delta_spool

    # 6.3 Relief pressure: e_pressure = Pset_i / 27.6e6 - 1 = Pset_mult - 1
    e_pressure = Pset_mult - 1.0

    # 6.4 Seal factors: e_seal_j = w_seal_j / 0.25
    e_seal = 1.0 + w_seal / W_SEAL_SCALE

    return np.array([
        e_comp, e_flow, e_pressure,
        e_seal[0], e_seal[1], e_seal[2], e_seal[3],
    ])


def compute_all_targets(all_params: list) -> np.ndarray:
    """Compute targets for a list of param dicts.

    Returns:
        targets: (N, 7) array
    """
    return np.array([compute_targets(p) for p in all_params])

