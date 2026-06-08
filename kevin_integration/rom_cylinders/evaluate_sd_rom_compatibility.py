from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

try:
    from rom_data_contract import TARGET_FEATURE_NAMES
except ImportError:
    from kevin_integration.rom_cylinders.rom_data_contract import TARGET_FEATURE_NAMES


ROM_DIR = Path(__file__).resolve().parent
DEFAULT_SD_CHECKPOINT = ROM_DIR.parent / "models" / "sd.pth"
RESULTS_DIR = ROM_DIR / "results"


SD_INPUT_ORDER = (
    "q_1",
    "q_2",
    "q_3",
    "q_4",
    "q_5",
    "qdot_1",
    "qdot_2",
    "qdot_3",
    "qdot_4",
    "qdot_5",
    "dP_1",
    "dP_2",
    "dP_3",
    "dP_4",
    "Fnet_1",
    "Fnet_2",
    "Fnet_3",
    "Fnet_4",
    "ValveCmd_1",
    "ValveCmd_2",
    "ValveCmd_3",
    "ValveCmd_4",
)

SD_DECODED_STATE_ORDER = (
    "q_next_1",
    "q_next_2",
    "q_next_3",
    "q_next_4",
    "q_next_5",
    "qdot_next_1",
    "qdot_next_2",
    "qdot_next_3",
    "qdot_next_4",
    "qdot_next_5",
    "dP_next_1",
    "dP_next_2",
    "dP_next_3",
    "dP_next_4",
    "Fnet_next_1",
    "Fnet_next_2",
    "Fnet_next_3",
    "Fnet_next_4",
)


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        meta = {key: value for key, value in payload.items() if key != "model_state_dict"}
    elif isinstance(payload, dict):
        state_dict = payload
        meta = {}
    else:
        return {"exists": True, "path": str(path), "error": f"Unsupported checkpoint payload {type(payload)}"}

    input_dim = None
    output_dim = None
    hidden_dim = None
    num_layers = 0
    if "dynamic_encoder.weight_ih_l0" in state_dict:
        weight = state_dict["dynamic_encoder.weight_ih_l0"]
        input_dim = int(weight.shape[1])
        hidden_dim = int(weight.shape[0] // 4)
        while f"dynamic_encoder.weight_ih_l{num_layers}" in state_dict:
            num_layers += 1
    if "regressor.3.weight" in state_dict:
        output_dim = int(state_dict["regressor.3.weight"].shape[0])
    elif "regressor.2.weight" in state_dict:
        output_dim = int(state_dict["regressor.2.weight"].shape[0])
    return {
        "exists": True,
        "path": str(path),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers or None,
        "target_mode": str(meta.get("target_mode", "") or ""),
        "window_size": meta.get("history_len", meta.get("window_size", meta.get("sd_window_size"))),
        "has_feature_normalizer": "feat_mean" in meta and "feat_std" in meta,
        "has_delta_normalizer": "y_delta_mean" in meta and "y_delta_std" in meta,
        "metadata_keys": sorted(str(key) for key in meta.keys()),
    }


def build_report(sd_checkpoint: Path) -> dict[str, Any]:
    checkpoint = inspect_checkpoint(sd_checkpoint)
    return {
        "sd_checkpoint": checkpoint,
        "sd_input_order": SD_INPUT_ORDER,
        "sd_decoded_state_order": SD_DECODED_STATE_ORDER,
        "sd_decoded_slices": {
            "q_next": [0, 5],
            "qdot_next": [5, 10],
            "dP_next": [10, 14],
            "Fnet_next": [14, 18],
        },
        "rom_target_order": TARGET_FEATURE_NAMES,
        "rom_target_slices": {
            "DeltaExtension_next": [0, 4],
            "DeltaDP_next": [4, 8],
        },
        "practical_pipeline_options": {
            "Option A": "ROM predicts DeltaExtension only; SD/observer provides dP state.",
            "Option B": "ROM predicts DeltaExtension and DeltaDP; SD remains a passive consistency checker.",
            "Option C": "ROM predicts DeltaExtension and DeltaDP; SD provides residual/blended correction.",
        },
        "recommended_first_option": "Option A",
        "compatibility_summary": [
            "SD input order is [q(5), qdot(5), dP(4), Fnet(4), valve_cmd(4)].",
            "SD decoded output order is [q_next(5), qdot_next(5), dP_next(4), Fnet_next(4)].",
            "ROM target order is [DeltaExtension_next(4), DeltaDP_next(4)].",
            "To compare SD with ROM pressure targets, decode SD state first and compare dP_next - current_dP to ROM DeltaDP_next.",
            "SD does not predict cylinder DeltaExtension; it cannot replace the learned cylinder ROM motion target.",
            "The practical first pipeline should use a ROM extension-only target while pressure prediction remains unvalidated.",
            "Do not use SD as ROM feedback or input until units, normalization, timestep, sign convention, and target_mode are validated on synchronized held-out samples.",
        ],
        "recommended_later_roles": [
            "passive observer",
            "dP/Fnet comparison channel",
            "optional Force/Fnet feature provider for a Force-augmented ROM",
            "small blended correction only after offline compatibility validation",
        ],
        "decision": "Use Option A as the first offline selection target; do not merge SD into ROM training or Isaac ROM integration yet.",
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    ckpt = report["sd_checkpoint"]
    lines = ["# SD / ROM Compatibility Note", ""]
    lines.append("## SD Checkpoint")
    lines.append("")
    for key, value in ckpt.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Channel Orders")
    lines.append("")
    lines.append("- SD input: `" + ", ".join(report["sd_input_order"]) + "`")
    lines.append("- SD decoded state: `" + ", ".join(report["sd_decoded_state_order"]) + "`")
    lines.append("- ROM target: `" + ", ".join(report["rom_target_order"]) + "`")
    lines.append("")
    lines.append("## Compatibility Summary")
    lines.append("")
    for item in report["compatibility_summary"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Practical Pipeline Decision")
    lines.append("")
    for key, value in report["practical_pipeline_options"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append(f"- Recommended first option: `{report['recommended_first_option']}`")
    lines.append("")
    lines.append("## Later Roles")
    lines.append("")
    for item in report["recommended_later_roles"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append(f"Decision: **{report['decision']}**")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Document SD/ROM compatibility without wiring SD into ROM training.")
    parser.add_argument("--sd_checkpoint", type=Path, default=DEFAULT_SD_CHECKPOINT)
    parser.add_argument("--output_dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.sd_checkpoint)
    json_path = args.output_dir / "sd_rom_compatibility.json"
    md_path = args.output_dir / "sd_rom_compatibility.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
