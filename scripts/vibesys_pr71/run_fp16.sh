#!/bin/zsh
cd /Users/marinedenolle/GitHub/QuakeScope
for c in 0 1 2 3; do
  items=("${(@f)$(awk -v c=$c 'NR%4==c' /private/tmp/claude-501/-Users-marinedenolle-GitHub-QuakeScope/8480fb4f-6290-4209-ac73-e3197db2b77d/scratchpad/pr71/out/items.txt)}")
  OMP_NUM_THREADS=2 .pixi/envs/dev/bin/python -I /private/tmp/claude-501/-Users-marinedenolle-GitHub-QuakeScope/8480fb4f-6290-4209-ac73-e3197db2b77d/scratchpad/pr71/scripts/run_arm.py /Users/marinedenolle/GitHub/QuakeScope-pr71 pr_fp16 original /private/tmp/claude-501/-Users-marinedenolle-GitHub-QuakeScope/8480fb4f-6290-4209-ac73-e3197db2b77d/scratchpad/pr71/out/picks_original_pr_fp16_c$c.csv $items > /private/tmp/claude-501/-Users-marinedenolle-GitHub-QuakeScope/8480fb4f-6290-4209-ac73-e3197db2b77d/scratchpad/pr71/out/log_original_pr_fp16_c$c.txt 2>&1 &
done
wait
echo FP16 DONE
