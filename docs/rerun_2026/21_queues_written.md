# 21 — The queues are written

2026-08-30. `shards.jsonl` is immutable once written; changing a date range or a
station list now means a new campaign prefix, not an edit.

| campaign | stations | shards | station-days | weight |
|---|--:|--:|--:|---|
| scedc | 1,128 | 8,479 | 4,106,669 | `jma_wc` |
| ncedc | 2,116 | 14,941 | 5,979,675 | `jma_wc` |
| earthscope | 51,846 | 153,208 | 67,983,975 | `jma_wc` |
| obs | 3,389 | 6,566 | 996,536 | `obs` |
| western | 24,113 | 72,505 | 33,799,828 | `original` |
| **total** | | **255,699** | **112,866,683** | |

Range 2010.001–2026.001 for all five, per [11](11_launch_plan.md). Verified after
writing: the shard count and station-day sum read back from S3 match the plan
exactly for every campaign.

Station lists come from `networks/<NET>.zip` — the per-network metadata the 2025
run used — selected by the network lists in `sb_catalog/configs/networks/`.
Western is the exception and comes from `western_states.csv`, since it is
selected geographically rather than by network.

## These counts are higher than the earlier estimates

| campaign | [12](12_output_storage.md) said | planned | ratio |
|---|--:|--:|--:|
| scedc | 2,467,740 | 4,106,669 | 1.66× |
| ncedc | 4,551,557 | 5,979,675 | 1.31× |
| earthscope onshore | 44,127,796 | 67,983,975 | 1.54× |
| obs | 950,793 | 996,536 | 1.05× |
| western | 33,776,383 | 33,799,828 | 1.00× |

Western matches because it is planned from the same file the estimate used. The
others do not, and most of the gap is **location-code separation**, which
[17](17_launch_conventions.md) specifies and the earlier estimate did not apply:
`networks/CI.zip` holds 1,128 station-locations across 801 station codes, 1.41×.
`CI.ACP.` and `CI.ACP.2C` are separate units of work because they are separate
instruments. The remainder is operating-window coverage.

At the SCEDC-measured 34 s per band-day this is ~1.07M vCPU-hours, **~$15,800**.

## Two things that qualify that number

**EarthScope reads are unprofiled and look much slower** — see
[19](19_earthscope_access.md). The 34 s figure was measured on SCEDC, where a
station-day fetches one object per channel. EarthScope stores one multi-channel
object per station-day and parses it whole. Campaigns 3, 4 and 5 are 91% of the
station-days, so if that ratio is materially worse the total moves with it. This
is the measurement to take before launching those three.

**`UL` has no metadata.** It is listed in `ncedc.txt` but `networks/UL.zip` does
not exist, so it is absent from the queue. One network of seven; worth deciding
whether it belongs before treating NCEDC as complete.
