#!/bin/zsh
# Weekly forward-test logging (run by launchd; safe to run by hand).
#
#   Wednesday: champion game predictions + weekly title odds   (before Thursday night's game)
#   Friday:    injury-challenger game predictions              (after final injury reports)
#
# Safeguards: wrong branch -> abort; uncommitted code -> the Python refuses; games that have
# kicked off are never logged; failures raise a macOS notification and are written to the log.
# DRY_RUN=1 prints predictions without writing the ledgers.

set -u
REPO="${NFL_REPO:-$HOME/Developer/nfl-playoff-model}"
EXPECTED_BRANCH="${NFL_LEDGER_BRANCH:-live/ledger-and-simulator}"
LOGDIR="$HOME/Library/Logs/nfl-playoff-model"
LOG="$LOGDIR/weekly.log"
PY="$REPO/.venv/bin/python"

mkdir -p "$LOGDIR"
cd "$REPO" || { echo "$(date) repo missing"; exit 1; }

notify() { /usr/bin/osascript -e "display notification \"$1\" with title \"NFL model\"" >/dev/null 2>&1; }
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

branch=$(git branch --show-current)
if [ "$branch" != "$EXPECTED_BRANCH" ]; then
  log "ABORT: on branch '$branch', expected '$EXPECTED_BRANCH' (ledger commits must land there)"
  notify "Weekly logging skipped: wrong git branch ($branch)"
  exit 1
fi

DRY=""; [ "${DRY_RUN:-0}" = "1" ] && DRY="--dry-run"
dow=$(date +%u)          # 1=Mon ... 7=Sun
rc=0

run() {
  log "running: $*"
  "$@" >>"$LOG" 2>&1 || { log "FAILED (exit $?): $*"; rc=1; }
}

case "$dow" in
  3)  # Wednesday
      run "$PY" -W ignore -m src.live.run --models ratings_qb_logit_v1 --commit $DRY
      if [ -z "$DRY" ]; then run "$PY" -W ignore -m src.simulation.title_odds --log --commit
      else run "$PY" -W ignore -m src.simulation.title_odds; fi ;;
  5)  # Friday
      run "$PY" -W ignore -m src.live.run --models ratings_qb_injury_logit_v1 --commit $DRY ;;
  *)  log "nothing scheduled for weekday $dow" ;;
esac

if [ "$rc" -ne 0 ]; then notify "Weekly logging had a failure. See ~/Library/Logs/nfl-playoff-model/weekly.log"; else log "done"; fi
exit $rc
