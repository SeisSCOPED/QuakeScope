"""Finish a pnw-eqt-2023 build: absolute manifest paths, run record, README.json, and the checks.

    python scripts/pnw_eqt_2023_finalize.py OUT_DIR [--expect-picks N --expect-files N]

Exits non-zero if any manifest row count differs from its Parquet footer, if a
station with picks has no row in stations.parquet, or if the totals differ
from --expect-*. Run it before uploading OUT_DIR to the prefix.
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

BASE = "s3://quakescope-picks-2026/pnw-eqt-2023/"
CITE = ("Ni, Y., Hutko, A., Skene, F., Denolle, M., Malone, S., Bodin, P., Hartog, R., and Wright, A. "
        "(2023). Curated Pacific Northwest AI-ready Seismic Dataset. Seismica, 2(1). "
        "https://doi.org/10.26443/seismica.v2i1.368")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--expect-picks", type=int)
    ap.add_argument("--expect-files", type=int)
    a = ap.parse_args()
    out = a.out

    n_picks = n_files = n_sd = mism = 0
    tids, nets = set(), set()
    for f in sorted(glob.glob(f"{out}/manifests/*.json")):
        m = json.load(open(f))
        for x in m["files"]:
            rel = x["path"].replace(BASE, "")
            if pq.ParquetFile(os.path.join(out, rel)).metadata.num_rows != x["rows"]:
                mism += 1
            x["path"] = BASE + rel
            n_files += 1
            nets.add(x["network"])
        n_picks += m["n_picks"]
        n_sd += m["station_days"]
        tids.update(r["tid"] for r in m["records"])
        json.dump(m, open(f, "w"))
    on_disk = len(glob.glob(f"{out}/picks/**/*.parquet", recursive=True))
    st = pd.read_parquet(f"{out}/stations.parquet")
    missing = tids - set(st.id)

    os.makedirs(f"{out}/runs", exist_ok=True)
    json.dump({
        "run_id": "pnw-eqt-ni2023", "model": "EQTransformer", "weight": "pnw",
        "weight_source": "PNW weights transfer-learned on the PNW-ML curated dataset "
                         "(Ni et al. 2023, Seismica, doi:10.26443/seismica.v2i1.368); dasway niyiyu/PNW-ML/pnw.pt.v1",
        "p_threshold": "0.1", "s_threshold": "0.1",
        "thresholds_note": "inferred: lowest conf in the picks is 0.100 for P and S",
        "filter": "bandpass 1-45 Hz, 2 corners, zero-phase (pnw.json.v1)", "component_order": "ZNE",
        "picked": "2022 (tar member dates Nov 2022) on UW hardware",
        "source": "https://dasway.ess.washington.edu/shared/niyiyu/PNW-all-picks/ml_picks_pnw_<YYYY>.tar",
    }, open(f"{out}/runs/pnw-eqt-ni2023.json", "w"), indent=1)
    n_fdsn = int(st.metadata_source.str.startswith("fdsn").sum())
    json.dump({
        "catalogue": "pnw-eqt-2023",
        "description": "Pacific Northwest phase picks from EQTransformer with the PNW weights, transfer-learned on the "
                       "curated PNW-ML dataset (Ni et al., 2023). 2002-2023, picked in 2022, converted in 2026-11 from "
                       "the original per-station-day SeisBench pickles. Same layout and pick schema as western/ and "
                       "quakescope-2025/.",
        "differs_from_western": ["amp and amp_vel are empty (no amplitudes measured)",
                                 "cha is the band only (EH, HH, BH)",
                                 "rid is pnw-eqt-ni2023 for every row",
                                 "P and S threshold 0.1 (inferred), not 0.2: more low-confidence picks"],
        "layout": {
            "picks": "picks/network=<NET>/year=<YYYY>/month=<MM>/pnweqt-<source year>-<part>.parquet, "
                     "partitioned by peak time (UTC)",
            "manifests": "manifests/pnweqt-<YYYY>.json: files with row counts, and one record per "
                         "station-band-day (tid, cha, yr, doy, npks)",
            "stations": f"stations.parquet: western/stations.parquet columns plus metadata_source; "
                        f"{len(st) - n_fdsn:,} rows from PNWStationXML.tar (2022-01), {n_fdsn} stations installed "
                        f"later from the IRIS/EarthScope FDSN station service",
            "runs": "runs/pnw-eqt-ni2023.json"},
        "counts": {"picks": n_picks, "station_days": n_sd, "station_locations_with_picks": len(tids),
                   "networks": len(nets), "files": n_files, "period": "2002-01-01 to 2023-12-31"},
        "station_days_note": "a station-day appears only if the original run wrote a pick file for it; "
                             "days without a file may be no data or no picks",
        "cite": [CITE],
        "converter": "scripts/convert_pnw_eqt_picks.py, scripts/build_pnw_stations.py, "
                     "scripts/fill_pnw_missing_stations.py, scripts/pnw_eqt_2023_finalize.py (SeisSCOPED/QuakeScope)",
    }, open(f"{out}/README.json", "w"), indent=1)

    print(f"picks {n_picks:,}  files {n_files} (on disk {on_disk})  station-days {n_sd:,}  "
          f"stations with picks {len(tids):,}  networks {len(nets)}  footer mismatches {mism}  "
          f"stations missing from table {len(missing)}")
    bad = mism or missing or on_disk != n_files
    bad = bad or (a.expect_picks is not None and n_picks != a.expect_picks)
    bad = bad or (a.expect_files is not None and n_files != a.expect_files)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
