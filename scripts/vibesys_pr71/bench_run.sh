#!/bin/zsh
# usage: bench_run.sh ARM REP -> out/bench_<ARM>_<REP>_g*.json; prints wall seconds
S=/private/tmp/claude-501/-Users-marinedenolle-GitHub-QuakeScope/8480fb4f-6290-4209-ac73-e3197db2b77d/scratchpad/pr71
cd /Users/marinedenolle/GitHub/QuakeScope
root=/Users/marinedenolle/GitHub/QuakeScope; [ $1 = pr ] && root=/Users/marinedenolle/GitHub/QuakeScope-pr71
t0=$(python3 -c 'import time;print(time.time())')
for g in 0 1 2 3; do
  items=("${(@f)$(cat $S/out/bench_g$g.txt)}")
  VELIIR_SOURCE=$S/scripts/veliir_arm64.c OMP_NUM_THREADS=2 .pixi/envs/dev/bin/python -I $S/scripts/bench_worker.py $root $1 $S/inv $S/out/bench_$1_$2_g$g.json $items > $S/out/bench_$1_$2_g$g.log 2>&1 &
done
wait
python3 -c "import time;print('$1 rep $2 wall', round(time.time()-$t0,2))"
