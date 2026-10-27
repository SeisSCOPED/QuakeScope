#!/bin/zsh
# usage: run_amps.sh ITEMS_FILE TAG  -> out/amp_<TAG>_main.csv, out/amp_<TAG>_pr.csv
S=/private/tmp/claude-501/-Users-marinedenolle-GitHub-QuakeScope/8480fb4f-6290-4209-ac73-e3197db2b77d/scratchpad/pr71
cd /Users/marinedenolle/GitHub/QuakeScope
items=("${(@f)$(cat $1)}")
for arm in main pr; do
  root=/Users/marinedenolle/GitHub/QuakeScope
  [ $arm = pr ] && root=/Users/marinedenolle/GitHub/QuakeScope-pr71
  VELIIR_SOURCE=/private/tmp/claude-501/-Users-marinedenolle-GitHub-QuakeScope/8480fb4f-6290-4209-ac73-e3197db2b77d/scratchpad/pr71/scripts/veliir_arm64.c OMP_NUM_THREADS=1 .pixi/envs/dev/bin/python -I $S/scripts/run_amps.py $root $arm $S/inv \
    $S/out/amp_$2_$arm.csv $items > $S/out/log_amp_$2_$arm.txt 2>&1
  grep -v Warn $S/out/log_amp_$2_$arm.txt | tail -4
done
