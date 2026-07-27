# Project todos

## DeepSeek agent behavior improvements
- [x] Create AGENTS.md with strong agent rules
- [x] Create memory.md with project context
- [x] Create todo.md for tracking
- [x] Set up deepseek-v4-pro.config.toml with workspace-write permissions
- [x] Create permanent setup in ~/rrlab_mulag (main repo)
- [x] Verify Moon Bridge routing (correct)
- [ ] Test DeepSeek autonomy with real coding tasks
- [ ] Document working patterns in memory.md for future sessions

## Ongoing observations
- Moon Bridge routing: Correct; agent hesitation is governance/permission-related
- DeepSeek V4 Pro: Has 1M context and agent capability; needs strong AGENTS.md + explicit task framing
- Safety: Workspace-write sandbox balances autonomy with safety (no full-access needed)

## PPO2 ROM residual policy

- [x] Replace template phase with local reference preview while preserving the 41-D contract
- [x] Build and gate five exact plus ten conservative procedural trajectories
- [x] Fix the 350 exact + 150 mixed iteration training profile and compact metrics
- [x] Record canonical ROM, trajectory, ownership, runtime, and promotion contracts
- [x] User ran the initial PPO profile; it was deliberately stopped after model 200
- [x] Stop the initial run after model 200 and compare model 100/model 200 for smoothness
- [x] Select model 100 as the conservative fine-tuning base
- [x] User ran the 150-iteration fixed-LR smoothness fine-tune from model 100
- [x] Complete smoothness fine-tune and freeze model 249 as the accepted controller
- [x] Add the quiet one-command five-station visualizer
- [x] Implement separate 42-observation/4-action constrained cylinder-4 experiment
- [ ] User: train the four-cylinder experiment and validate against accepted model 249
- [ ] Evaluate `model_499.pt` on all five shapes for three seeds against zero-policy baseline
- [ ] Promote only if the tracking, acceleration, head, clearance, and truck-lock gates pass
- [ ] Run the promoted checkpoint through the five-station park/work/relocate supervisor
