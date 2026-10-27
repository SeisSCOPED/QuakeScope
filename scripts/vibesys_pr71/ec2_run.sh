#!/bin/bash
# PR #71 timing on one EC2 instance: vanilla vs PR fp32, 4 workers x 2 threads,
# 4 alternating reps. Called by user-data as: run.sh RESULT_PUT_URL LOG_PUT_URL
RESULT_URL=$1
LOG_URL=$2
export HOME=/root
cd /work
OUT=/work/out
mkdir -p $OUT
finish() {
  tar czf /work/results.tar.gz -C /work out
  curl -fsS -X PUT --upload-file /work/results.tar.gz "$RESULT_URL"
  curl -fsS -X PUT --upload-file /var/log/qsbench.log "$LOG_URL"
  shutdown -h now
}
trap finish EXIT

PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
$PY -m venv /opt/v
/opt/v/bin/pip install -q --upgrade pip
/opt/v/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
/opt/v/bin/pip install -q seisbench==0.12.5 s3fs boto3 earthscope-sdk==1.8.0 pyarrow

mkdir -p $HOME/.seisbench/models/v3
cp -r /work/sbcache/phasenet $HOME/.seisbench/models/v3/
for g in 0 1 2 3; do sed "s#DATA/#/work/data/#" /work/groups/g$g.txt > /work/groups/abs_g$g.txt; done

{
  echo "== lscpu"; lscpu
  echo "== instance"; curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" http://169.254.169.254/latest/meta-data/instance-type; echo
  echo "== python"; /opt/v/bin/python -V
  echo "== torch"; /opt/v/bin/python -c "
import torch; print(torch.__version__, 'mkldnn', torch.backends.mkldnn.is_available(),
 'fused', all(hasattr(torch.ops.mkldnn, o) for o in ('_convolution_pointwise', '_convolution_transpose_pointwise')))
print(torch.__config__.show())"
  echo "== gcc"; gcc --version | head -1
  echo "== pip freeze"; /opt/v/bin/pip freeze
} > $OUT/env.txt 2>&1

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2

run_arm() {  # ARM REP
  local root=/work/main; [ "$1" = pr ] && root=/work/pr
  local t0=$(date +%s.%N)
  for g in 0 1 2 3; do
    mapfile -t items < /work/groups/abs_g$g.txt
    /opt/v/bin/python -I /work/scripts/bench_worker.py $root $1 /work/inv \
      $OUT/bench_$1_$2_g$g.json "${items[@]}" > $OUT/bench_$1_$2_g$g.log 2>&1 &
  done
  wait
  echo "$1 rep $2 wall $(awk "BEGIN{print $(date +%s.%N) - $t0}")" >> $OUT/walls.txt
}

for r in 1 2 3 4; do
  if [ $((r % 2)) -eq 1 ]; then run_arm vanilla $r; run_arm pr $r; else run_arm pr $r; run_arm vanilla $r; fi
done
echo done >> $OUT/walls.txt
