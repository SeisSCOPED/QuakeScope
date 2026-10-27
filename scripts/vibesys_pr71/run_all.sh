#!/bin/zsh
cd /Users/marinedenolle/GitHub/QuakeScope
items=("${(@f)$(cat /private/tmp/claude-501/-Users-marinedenolle-GitHub-QuakeScope/8480fb4f-6290-4209-ac73-e3197db2b77d/scratchpad/pr71/out/items.txt)}")
for w in jma_wc original; do for arm in vanilla pr_fp32 pr_bf16; do
  echo "$w $arm"
done; done | xargs -P 6 -L 1 zsh -c 'OMP_NUM_THREADS=2 .pixi/envs/dev/bin/python -I /private/tmp/claude-501/-Users-marinedenolle-GitHub-QuakeScope/8480fb4f-6290-4209-ac73-e3197db2b77d/scratchpad/pr71/scripts/run_arm.py /Users/marinedenolle/GitHub/QuakeScope-pr71 $1 $0 /private/tmp/claude-501/-Users-marinedenolle-GitHub-QuakeScope/8480fb4f-6290-4209-ac73-e3197db2b77d/scratchpad/pr71/out/picks_$0_$1.csv "${(@f)$(cat /private/tmp/claude-501/-Users-marinedenolle-GitHub-QuakeScope/8480fb4f-6290-4209-ac73-e3197db2b77d/scratchpad/pr71/out/items.txt)}" > /private/tmp/claude-501/-Users-marinedenolle-GitHub-QuakeScope/8480fb4f-6290-4209-ac73-e3197db2b77d/scratchpad/pr71/out/log_$0_$1.txt 2>&1'
echo ALL DONE
