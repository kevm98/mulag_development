# DeepSeek V4 Pro Setup for Codex

This document explains the permanent setup for DeepSeek V4 Pro agent behavior improvements.

## Files in this repository

### AGENTS.md
Strong autonomy and safety rules for agent behavior. Codex loads this automatically when it's in the repository root.

**Key rules:**
- Do not ask before read-only operations (grep, Python checks, tests)
- Do not ask before file edits or terminal runs
- Keep safe boundaries: no sudo, destructive git, force-push
- Explicit debugging workflow: inspect → search → change → verify

### memory.md
Persistent project context. Updated manually with:
- Environment paths and constraints
- Working commands
- Integration status (DeepSeek routing, Moon Bridge, config location)
- Verified working patterns from successful tasks

Use `<< <task>` in VS Code to have the agent read this before acting.
Use `>> <fact>` to record new learnings.

### todo.md
Progress tracking for DeepSeek improvements and ongoing observations.

### .codex-config/deepseek-v4-pro.config.toml
Backup of the system-wide config. If the system config at `~/.config/codex-deepseek/deepseek-v4-pro.config.toml` is lost:

```bash
cp /home/rrlab/rrlab_mulag/.codex-config/deepseek-v4-pro.config.toml \
   ~/.config/codex-deepseek/deepseek-v4-pro.config.toml
```

## System-wide configuration

**File:** `~/.config/codex-deepseek/deepseek-v4-pro.config.toml`

```toml
approval_policy = "on-request"
sandbox_mode = "workspace-write"

[sandbox_workspace_write]
network_access = false
writable_roots = [
  "/home/rrlab/rrlab_mulag",
  "/home/rrlab/rrlab_mulag.worktrees",
  "/home/rrlab/IsaacLab",
  "/tmp"
]
```

**What this does:**
- `approval_policy = "on-request"`: No constant permission nagging, but asks before risky operations
- `sandbox_mode = "workspace-write"`: Can edit files and run commands in project directories
- `network_access = false`: No external API calls or downloads
- `writable_roots`: Safe list of directories the agent can modify

**Why not full-access?** Workspace-write provides autonomy while protecting your system. Full-access is rarely needed and removes safety boundaries.

## How to verify the setup

1. **Check AGENTS.md is loaded:**
   ```bash
   cat ~/rrlab_mulag/AGENTS.md | head -5
   ```

2. **Check config exists:**
   ```bash
   cat ~/.config/codex-deepseek/deepseek-v4-pro.config.toml
   ```

3. **Test in VS Code:**
   - Select DeepSeek V4 Pro in Copilot chat
   - Use a task that requires inspection and editing: "Inspect test_import_mulag.py, run it, and report results. Do not ask permission."
   - Agent should NOT ask before reading/running/reporting
   - Agent should use `/home/rrlab/miniconda3/envs/env_isaaclab/bin/python` directly

## Architecture

```
VS Code Codex Chat
    ↓ (user selects DeepSeek V4 Pro model)
    ↓ (AGENTS.md in repo loaded automatically)
    ↓ (injects modelProvider = "moonbridge" via extension)
    ↓
Moon Bridge / LiteLLM
    ↓ (translates Codex Responses API → DeepSeek format)
    ↓
DeepSeek V4 Pro API
    ↓ (returns tool calls: grep, file-read, execute)
    ↓
Moon Bridge (executes tools)
    ↓ (returns results to DeepSeek)
    ↓
Agent behavior applied:
    - AGENTS.md rules: Autonomy + safety
    - Config permissions: workspace-write sandbox
    - Memory: context from memory.md
    ↓
Codex Chat displays results
```

## If setup is lost again

1. **AGENTS.md, memory.md, todo.md are in ~/rrlab_mulag** — Always safe
2. **Config is in ~/.config/codex-deepseek/** — Restore from backup:
   ```bash
   mkdir -p ~/.config/codex-deepseek
   cp ~/rrlab_mulag/.codex-config/deepseek-v4-pro.config.toml ~/.config/codex-deepseek/
   ```

3. **Verify everything:**
   ```bash
   ls -la ~/rrlab_mulag/{AGENTS.md,memory.md,todo.md}
   ls -la ~/.config/codex-deepseek/deepseek-v4-pro.config.toml
   ```

## Next steps

1. Open VS Code
2. Select DeepSeek V4 Pro in Copilot chat
3. Use explicit task framing: "Do not ask permission. Use terminal and Python to inspect, edit, verify."
4. Update memory.md with working patterns
5. Check todo.md for pending improvements

## References

- Codex agent configuration: https://developers.openai.com/codex/guides/agents-md
- Codex permissions: https://developers.openai.com/codex/config-reference
- DeepSeek API: https://api-docs.deepseek.com
- Moon Bridge routing: Moon Bridge/LiteLLM bridges Codex APIs to other models
