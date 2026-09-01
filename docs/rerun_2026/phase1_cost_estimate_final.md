# Phase 1 Cost Estimate — RETRACTED

**Retracted 2026-09-01.** The cost table that was here is void.

It reported ~$13,920 across the five campaigns, built on a claimed SCEDC rate of
30 seconds per band-day. That rate was produced by dividing one job's 1,788 s
runtime by 60 band-days, when the shard it ran actually held **460** station-days
— the divisor was invented, not measured. The job also ran on container image
`9abd01c`, which predates the retry-bounding fixes, so its runtime does not
describe current code either.

The table additionally assumed EarthScope reads at the same speed as SCEDC.
EarthScope I/O has never been profiled; [19_earthscope_access.md](19_earthscope_access.md)
records the open suspicion that it is substantially slower because it stores one
multi-channel object per station-day. Since campaigns 3–5 are 91% of the
station-days, that assumption carries most of the total.

Full account of what went wrong: [PHASE1_FINAL_REPORT.md](PHASE1_FINAL_REPORT.md).

## The calculator itself is fine

`scripts/calculate_phase1_cost.py` is unaffected — it takes measured
seconds-per-band-day as input and its per-campaign station-day counts match
[21_queues_written.md](21_queues_written.md) exactly. It was fed bad inputs, not
written wrong. Re-run it once Phase 1 has produced real measurements on
`quakescope_v3_worker:3`:

```bash
python scripts/calculate_phase1_cost.py \
  --scedc-seconds <measured> \
  --earthscope-seconds <measured> \
  --procs <measured> \
  --output docs/rerun_2026/phase1_cost_estimate_final.md
```

Until then the campaign has **no cost estimate**. The last figure with a
defensible basis is the ~$15,800 in [21_queues_written.md](21_queues_written.md),
derived from the SCEDC-measured 34 s/band-day — and that document already flags
the two things that qualify it.
