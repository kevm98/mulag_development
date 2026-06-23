# Agent behavior for this repository

You are working as a coding agent, not as a passive chat assistant.

## Autonomy
- The user's task message is explicit permission to inspect files, search the codebase, run safe terminal commands, edit files related to the task, and verify the result.
- Do not ask before running read-only commands, grep/search commands, Python checks, unit tests, import checks, or small non-destructive validation commands.
- Do not stop after giving advice if the task requires code inspection or execution. Use the terminal and verify.

## Safety boundaries
- Do not run `sudo`.
- Do not run `rm -rf`, `git reset --hard`, force-push, public commit amend, or destructive cleanup without explicit confirmation.
- Do not push, commit, or install new packages unless explicitly asked.
- Do not launch long-running GUI/Isaac Sim sessions unless asked. Prefer short headless or import-level checks with timeout.

## Project environment
- Main project: ~/rrlab_mulag
- IsaacLab path: ~/IsaacLab
- Conda env: env_isaaclab
- Prefer using the absolute Python path when possible:
  `/home/rrlab/miniconda3/envs/env_isaaclab/bin/python`

## Terminal rules
- For Python checks, use:
  `/home/rrlab/miniconda3/envs/env_isaaclab/bin/python <script>`
- For shell commands that need conda activation, use:
  `source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab && <command>`
- Use timeout for risky/long commands:
  `timeout 120s <command>`

## Debugging workflow
1. Inspect relevant files first.
2. Search with grep/rg before guessing.
3. Make the smallest change that solves the problem.
4. Verify with grep, import checks, tests, or a minimal run.
5. Report exactly what changed and what was verified.

## Failure handling
- If a command fails, read the error and try one focused fix.
- If the second attempt fails, inspect --help, version, imports, or config.
- Do not silently skip verification.
- Do not claim something works unless it was actually checked.

## Project memory
- If the user says `<<`, read `memory.md` and `todo.md` before acting.
- If the user says `<< status`, summarize pending work from `todo.md`.
- If the user says `>> <text>`, ask before saving that text into `memory.md`.
- Store only project decisions, environment facts, commands that worked, and unresolved blockers.
