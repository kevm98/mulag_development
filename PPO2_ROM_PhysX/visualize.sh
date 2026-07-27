#!/usr/bin/env bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/PPO2_ROM_PhysX/visualize_4cyl.sh" "$@"
