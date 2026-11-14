#!/bin/bash
# Build s3://quakescope-picks-2026/pnw-eqt-2023/ end to end on a Linux host in us-east-2:
# download each yearly tar from dasway (resumable; the server drops long transfers),
# convert, build the station table, check every count, then upload.
#
#   bash scripts/pnw_eqt_2023_build.sh WORKDIR [EXPECT_PICKS EXPECT_FILES]
#
# Uploads only if scripts/pnw_eqt_2023_finalize.py passes. Progress goes to
# WORKDIR/log.txt and, when LOG_S3 is set, is copied there every few minutes.
set -uo pipefail
W=$1; EXPECT_PICKS=${2:-}; EXPECT_FILES=${3:-}
REPO=$(cd "$(dirname "$0")/.." && pwd)
DASWAY=https://dasway.ess.washington.edu/shared/niyiyu/PNW-all-picks
DEST=s3://quakescope-picks-2026/pnw-eqt-2023/
mkdir -p "$W/dl" "$W/out" "$W/logs"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$W/log.txt"; }
push_log() { [ -n "${LOG_S3:-}" ] && aws s3 cp --quiet "$W/log.txt" "$LOG_S3" || true; }

fetch() {  # url dest: resume until the byte count matches Content-Length
  local url=$1 dst=$2 exp got
  exp=$(curl -sI --http1.1 "$url" | awk 'tolower($1)=="content-length:"{print $2+0}')
  for _ in $(seq 1 80); do
    got=$(stat -c %s "$dst" 2>/dev/null || echo 0)
    [ "$got" = "$exp" ] && return 0
    curl -sSL --http1.1 -C - --speed-limit 50000 --speed-time 60 -o "$dst" "$url" 2>>"$W/logs/curl.err"
  done
  return 1
}
export -f fetch

year() {
  local y=$1 tar=$W/dl/ml_picks_pnw_$1.tar
  fetch "$DASWAY/ml_picks_pnw_$y.tar" "$tar" || { log "$y DOWNLOAD FAILED"; return 1; }
  python3 "$REPO/scripts/convert_pnw_eqt_picks.py" "$tar" --year "$y" --out "$W/out" \
      > "$W/logs/$y.json" 2> "$W/logs/$y.err" || { log "$y CONVERT FAILED $(tail -1 "$W/logs/$y.err")"; return 1; }
  rm -f "$tar"
  log "$y ok $(cat "$W/logs/$y.json")"
  push_log
}
export -f year log push_log
export W REPO DASWAY

log "start, repo $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
push_log
seq 2002 2023 | xargs -P 4 -I{} bash -c 'year {}'
n_ok=$(grep -c " ok {" "$W/log.txt")
[ "$n_ok" = 22 ] || { log "only $n_ok of 22 years converted, not uploading"; push_log; exit 1; }

fetch "$DASWAY/PNWStationXML.tar" "$W/dl/PNWStationXML.tar" || { log "station tar download failed"; push_log; exit 1; }
python3 "$REPO/scripts/build_pnw_stations.py" "$W/dl/PNWStationXML.tar" "$W/out/stations_all.parquet" 2>&1 | tee -a "$W/log.txt"
python3 "$REPO/scripts/fill_pnw_missing_stations.py" "$W/out" 2>&1 | tee -a "$W/log.txt"
rm -f "$W/dl/PNWStationXML.tar"

args=()
[ -n "$EXPECT_PICKS" ] && args+=(--expect-picks "$EXPECT_PICKS")
[ -n "$EXPECT_FILES" ] && args+=(--expect-files "$EXPECT_FILES")
if ! python3 "$REPO/scripts/pnw_eqt_2023_finalize.py" "$W/out" "${args[@]}" 2>&1 | tee -a "$W/log.txt"; then
  log "CHECKS FAILED, not uploading"; push_log; exit 1
fi

log "checks passed, uploading"
push_log
aws s3 sync "$W/out" "$DEST" --exclude "stations_all.parquet" --only-show-errors 2>&1 | tee -a "$W/log.txt"
aws s3 sync "$W/out" "$DEST" --exclude "stations_all.parquet" --only-show-errors 2>&1 | tee -a "$W/log.txt"
log "upload done: $(aws s3 ls "$DEST" --recursive --summarize | tail -2 | tr '\n' ' ')"
push_log
