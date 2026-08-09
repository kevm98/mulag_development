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


def test_switching_oracle_screen_is_paired_isolated_and_noncertifiable() -> None:
    source = (ROOT / "PPO2_ROM_B_PhysX/scripts/screen_switching_oracle.py").read_text()
    assert '"model_53_ppo2_rom_b_final.pt"' in source
    assert '"model_49_ppo2_rom_b_final.pt"' in source
    assert '("romb", a.baseline_checkpoint)' in source
    assert '("oracle", a.oracle_checkpoint)' in source
    assert '"process_isolation": True' in source
    assert '"diagnostic_only": True' in source
    assert 'raw._command_cylinder_rom.true_factors.copy_(target)' in source
    assert 'choices=("health", "independent")' in source
    assert 'generator.manual_seed(500000 + int(a.seed))' in source
    assert 'target[:, 0] = -0.125 + unit[:, 0]' in source
    assert '"target_factors"' in source
    assert 'elapsed >= float(raw._one_way_s)' in source
    assert '"post_switch_rmse_m"' in source


def test_switching_oracle_matrix_is_resumable_and_controls_no_change() -> None:
    source = (
        ROOT / "PPO2_ROM_B_PhysX/scripts/screen_switching_oracle_matrix.py"
    ).read_text()
    assert 'default="0.4,0.7,1.0,1.2"' in source
    assert 'default="17,29,41"' in source
    assert 'if 1.0 not in target_hs:' in source
    assert "if _valid_condition(path, seed, target_h):" in source
    assert '"acceleration_interaction_mps2": gap - control_gap' in source
    assert '"positive_interaction_seeds"' in source
    assert '"unique_metric_outcomes"' in source
    assert '"independent_seed_replication"' in source
    assert '"diagnostic_only": True' in source


def test_independent_switch_screen_uses_distinct_factor_vectors_and_control() -> None:
    source = (
        ROOT / "PPO2_ROM_B_PhysX/scripts/screen_switching_oracle_independent.py"
    ).read_text()
    assert 'default="17,29,41"' in source
    assert 'run(control_path, mode="health", seed=seeds[0])' in source
    assert 'run(path, mode="independent", seed=seed)' in source
    assert 'if len(factor_vectors) != len(seeds):' in source
    assert '"distinct_target_factor_vectors": len(factor_vectors)' in source
    assert '"controlled_acceleration_interaction_mps2"' in source
    assert '"diagnostic_only": True' in source
