from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_rom_b_gui_player_uses_robust_task_and_current_artifacts() -> None:
    source = (ROOT / "PPO2_ROM_B_PhysX/scripts/play.py").read_text()
    assert "RRLAB-Mulag-PPO2-ROM-B-Robust-Policy-PLAY-v0" in source
    assert "model_53_ppo2_rom_b_final.pt" in source
    assert "plant_reachable_trajectory_library_v3_circle_c7_retime.json" in source
    assert 'cfg.rom_b_mode = "random" if use_variable_plant else "nominal"' in source
    assert "cfg.clearance_frame_consistent = True" in source
    assert "cfg.truck_full_pose_fixture = True" in source
    assert '"[ROM-B PLAY] truck fixture=FIXED CHASSIS' in source
    assert "truck_rotation_error_deg" in source
    assert "cfg.residual_scale_m = 0.002" in source
    assert "cfg.cylinder4_residual_scale_m = 0.00025" in source
    assert "class LiveArmPlots" in source
    assert "class ViewportTrajectoryLine" in source
    assert 'print("[ROM-B PLAY] viewport trajectory enabled: blue=reference"' in source
    assert 'f"correlated-health h={float(a.fixed_h):.3f}"' in source
    assert '"independent-random h=n/a"' in source
    assert '"nominal h=1.000"' in source
    assert "raw.cfg.episode_length_s = float(raw._complete_cycle_time)" in source
    assert 'f"[ROM-B PLAY] episode=A->B->A duration="' in source
    assert 'default=1,\n    help="Number of A->B->A episodes' in source
    assert 'f"episode={episode:03d}/{a.episodes:03d} "' in source
    assert '"timeout/round-trip" if bool(raw.reset_time_outs[0])' in source
    assert "def begin_episode(" in source
    assert 'f"ROM-B episode {episode}/{total_episodes} | "' in source
    assert 'f"plant={plant} | factors={factor_text}"' in source
