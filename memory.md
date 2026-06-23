# Project memory

## Environment
- Main Python env: `/home/rrlab/miniconda3/envs/env_isaaclab/bin/python`
- IsaacLab path: `~/IsaacLab`
- RRLAB project path: `~/rrlab_mulag`
- Codex model: DeepSeek V4 Pro (with Moon Bridge routing)

## Known constraints
- Laptop GPU: RTX 3060 Laptop, 6 GB VRAM
- Large IsaacLab runs can OOM; use small num_envs and disable cameras/terrain for debugging
- DeepSeek V4 Pro needs Moon Bridge/LiteLLM for routing (Responses API → DeepSeek bridge)

## Working commands
- Python import check: `/home/rrlab/miniconda3/envs/env_isaaclab/bin/python test_import_mulag.py`
- VS Code Copilot CLI: `gh copilot`
- Activate conda: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab`

## DeepSeek integration status
- Routing: VS Code extension injects `modelProvider = "moonbridge"` for deepseek-v4-pro
- Moon Bridge: Handles Codex Responses API → DeepSeek translation
- Config: `~/.config/codex-deepseek/deepseek-v4-pro.config.toml` (workspace-write sandbox)
- Writable paths: `/home/rrlab/rrlab_mulag`, `/home/rrlab/rrlab_mulag.worktrees`, `/home/rrlab/IsaacLab`, `/tmp`

## Verified working patterns
- AGENTS.md loaded by Codex/VS Code before task execution
- Agent autonomy: No asking before read-only operations, grep, Python checks
- Terminal execution: All commands (python/bash/grep/git) run without approval gate
- Memory system: `<<` reads memory.md before task; `>> <text>` saves to memory.md
- Explicit task framing: "Do not ask permission" → agent behaves as executor, not advisor
