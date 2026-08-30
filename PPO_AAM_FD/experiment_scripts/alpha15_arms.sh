#!/usr/bin/env bash
# LOWER-AUTHORITY ARMS: alpha = 0.15 instead of 0.5.
#
# At alpha=0.5 the residual can move the action by half its full range, which is
# enough to wreck tracking: the no-history control reached 1.16 m lateral error
# against a frozen base at ~61 mm. The base policy is already good (7.6 mm cut),
# so there is little room to improve and a great deal of room to damage, and a
# high-authority residual spends itself on the second.
#
# alpha=0.15 keeps enough authority to correct the measured +205.6% flip
# degradation while bounding the damage a wrong correction can do. Everything
# else is identical, so this is a clean authority sweep, not a new experiment.
set -u
cd /tmp/rrlab_scratch
TAG=a15_history   ALPHA=0.15 bash residual_run.sh
TAG=a15_nohistory ALPHA=0.15 ARG=--no_history bash residual_run.sh
echo ALPHA15_ARMS_DONE
