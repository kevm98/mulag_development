"""MATLAB v7.3 / HDF5 decoder for Simscape cylinder datasets."""

import h5py
import numpy as np
from typing import Any, Dict


def dereference(f: h5py.File, ref: h5py.h5r.Reference) -> Any:
    """Dereference an HDF5 object reference."""
    return f[ref]


def decode_char_array(obj: h5py.Dataset) -> str:
    """Decode a MATLAB char array stored as an HDF5 dataset."""
    data = obj[:].flatten()
    return ''.join(chr(int(c)) for c in data)


def extract_run_type(params_group: h5py.Group) -> str:
    """Extract the RunType string from a Params group."""
    rt_ds = params_group['RunType']
    return decode_char_array(rt_ds)


def load_open_loop_run(run_path: str) -> dict:
    """Load a single open-loop per-run MAT file."""
    with h5py.File(run_path, 'r') as f:
        rd = f['RunData']
        valve_cmd = rd['ValveCmd'][:]
        extension = rd['Extension'][:]
        ext_vel = rd['ExtVel'][:]
        dp = rd['dP'][:]
        force = rd['Force'][:]
        joint_pos = rd['JointPos'][:]
        joint_vel = rd['JointVel'][:]
        time = rd['Time'][:].squeeze()
        dt = float(time[1] - time[0])
        n_transitions = len(time) - 1
        params_group = rd['Params']
        params = _parse_params(params_group)
        run_type = extract_run_type(params_group)

    return {
        'valve_cmd': valve_cmd,
        'extension': extension,
        'ext_vel': ext_vel,
        'dP': dp,
        'force': force,
        'joint_pos': joint_pos,
        'joint_vel': joint_vel,
        'time': time,
        'dt': dt,
        'params': params,
        'run_type': run_type,
        'n_transitions': n_transitions,
    }


def load_closed_loop_run(cl_mat_path: str, run_index: int) -> dict:
    """Load a single closed-loop run from the curriculum MAT file."""
    with h5py.File(cl_mat_path, 'r') as f:
        data_ds = f['Dataset/Data']
        ref = data_ds[run_index, 0]
        obj = f[ref]
        valve_cmd = obj['ValveCmd'][:]
        extension = obj['Extension'][:]
        dp = obj['dP'][:]
        force = obj['Force'][:]
        joint_pos = obj['JointPos'][:]
        joint_vel = obj['JointVel'][:]
        time = obj['Time'][:].squeeze()
        dt = float(time[1] - time[0])
        n_transitions = len(time) - 1
        params_group = obj['Params']
        params = _parse_params(params_group)
        run_type = extract_run_type(params_group)

    return {
        'valve_cmd': valve_cmd,
        'extension': extension,
        'ext_vel': None,
        'dP': dp,
        'force': force,
        'joint_pos': joint_pos,
        'joint_vel': joint_vel,
        'time': time,
        'dt': dt,
        'params': params,
        'run_type': run_type,
        'n_transitions': n_transitions,
    }


def _parse_params(params_group: h5py.Group) -> Dict[str, Any]:
    """Extract scalar and vector parameters from a Params HDF5 group."""
    result = {}
    for key in params_group.keys():
        val = params_group[key]
        try:
            if hasattr(val, 'shape') and val.shape is not None:
                arr = val[:]
                if arr.size == 1:
                    result[key] = float(arr.item())
                else:
                    result[key] = arr.squeeze()
            else:
                result[key] = val
        except Exception:
            result[key] = None
    return result


def compute_ext_vel_causal(extension: np.ndarray, dt: float) -> np.ndarray:
    """Compute causal extension velocity via backward difference.

    xdot_t = (x_t - x_{t-1}) / dt  for t >= 1
    xdot_0 = xdot_1 (forward fallback)
    """
    ext_vel = np.zeros_like(extension)
    if extension.shape[1] >= 2:
        ext_vel[:, 1:] = (extension[:, 1:] - extension[:, :-1]) / dt
        ext_vel[:, 0] = ext_vel[:, 1]
    return ext_vel
