"""Draft figures for the 2026 data report (update of Ni et al., 2025, Seismica).

Everything is read anonymously from the public parts of s3://quakescope-picks-2026:
the station tables, the per-station-day availability tables of `western` and
`obs`, the 2025 manifests (per-station-day pick counts of the 2025 run), and one
network-month of picks from each catalogue for the pick-level comparison.

    python docs/paper_2026/make_figures.py fetch   --cache <dir>   # ~3.5 GB, 5 to 15 min
    python docs/paper_2026/make_figures.py figures --cache <dir>   # PNGs + numbers.json

Numbers quoted in OUTLINE.md come from figures/numbers.json, never typed.
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

BUCKET = "quakescope-picks-2026"
HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
C26, C25, C_OBS, GREY = "#2a78d6", "#eb6834", "#1b9e77", "#9a978e"
PRIORITY = ["HH", "EH", "SH", "BH", "BN", "DP", "HN", "CN"]   # constants.CHANNEL_PRIORITY, land weights
STATUS_ORDER = ["loaded", "no_data", "not_found", "empty_read", "unread", "unknown"]
STATUS_COLOR = {"loaded": C26, "no_data": "#b9b6ad", "not_found": "#8c6d46", "empty_read": "#e3b448",
                "unread": "#c0392b", "unknown": "#e7e7e2"}


# ----------------------------------------------------------------------------- fetch
def fetch(cache: Path):
    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    cache.mkdir(parents=True, exist_ok=True)

    def get(key):
        dest = cache / key
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            fs.get(f"{BUCKET}/{key}", str(dest))
        return dest

    for cat in ("western", "obs", "quakescope-2025"):
        get(f"{cat}/stations.parquet")
    for cat in ("western", "obs"):
        keys = [k.split("/", 1)[1] for k in fs.glob(f"{BUCKET}/{cat}/availability/network=*/availability.parquet")]
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(get, keys))
        print(cat, len(keys), "availability tables", flush=True)

    nets = set()
    for cat in ("western", "obs"):
        nets |= set(pd.read_parquet(cache / cat / "stations.parquet").id.str.split(".").str[0])
    man = [k.split("/", 1)[1] for k in fs.ls(f"{BUCKET}/quakescope-2025/manifests/")]
    man = [k for k in man if k.rsplit("docdb2025-", 1)[1][:-5] in nets]
    print(f"2025 manifests: {len(man)} networks", flush=True)
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(get, man))
    compact_2025(cache)

    # one network-month of picks from each catalogue, for figure 4
    for cat in ("western", "quakescope-2025"):
        for k in fs.ls(f"{BUCKET}/{cat}/picks/network=CI/year=2019/month=07/"):
            get(k.split("/", 1)[1])
    for k in fs.ls(f"{BUCKET}/obs/picks/network=XO/year=2018/month=08/"):
        get(k.split("/", 1)[1])
    for k in fs.ls(f"{BUCKET}/quakescope-2025/picks/network=XO/year=2018/month=08/"):
        get(k.split("/", 1)[1])


def compact_2025(cache: Path):
    """2025 manifests -> one Parquet of (tid, cha, date, npks), stations of western/obs only."""
    import orjson
    out = cache / "q2025_records.parquet"
    if out.exists():
        return
    ids = set()
    for cat in ("western", "obs"):
        ids |= set(pd.read_parquet(cache / cat / "stations.parquet").id)
    parts = []
    for f in sorted((cache / "quakescope-2025" / "manifests").glob("*.json")):
        recs = orjson.loads(f.read_bytes()).get("records", [])
        if not recs:
            continue
        df = pd.DataFrame(recs, columns=["tid", "cha", "yr", "doy", "npks"])
        df = df[df.tid.isin(ids)]
        if len(df):
            parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df.yr * 1000 + df.doy, format="%Y%j")
    df = df[["tid", "cha", "date", "npks"]].astype({"npks": "int32"})
    df.to_parquet(out)
    print(f"2025 records kept: {len(df):,}", flush=True)


# ----------------------------------------------------------------------------- load
def load(cache: Path):
    st = {c: pd.read_parquet(cache / c / "stations.parquet") for c in ("western", "obs", "quakescope-2025")}
    av = {}
    for cat in ("western", "obs"):
        a = pd.concat([pd.read_parquet(p, columns=["tid", "date", "status", "cha", "npks"])
                       for p in sorted((cache / cat / "availability").glob("network=*/availability.parquet"))],
                      ignore_index=True)
        a["date"] = pd.to_datetime(a.date)
        for c in ("status", "cha"):
            a[c] = a[c].astype("category")
        av[cat] = a
    q = pd.read_parquet(cache / "q2025_records.parquet")
    q["cha"] = q.cha.astype("category")
    return st, av, q


def compare_station_days(av_cat, q):
    """One row per station-day picked by both runs, 2025 counts on the band 2026 used."""
    a = av_cat[av_cat.status == "loaded"][["tid", "date", "cha", "npks"]].rename(columns={"npks": "n26"})
    a["cha"] = a.cha.astype(str)
    qq = q.assign(cha=q.cha.astype(str)).rename(columns={"npks": "n25"})
    m = a.merge(qq, on=["tid", "date", "cha"], how="inner")
    return m


# ----------------------------------------------------------------------------- figures
def figures(cache: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9})
    FIG.mkdir(exist_ok=True)
    N = {"built": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "source": f"s3://{BUCKET} (public objects)"}
    st, av, q = load(cache)
    w, o, q_st = st["western"], st["obs"], st["quakescope-2025"]
    aw, ao = av["western"], av["obs"]
    lw, lo = aw[aw.status == "loaded"], ao[ao.status == "loaded"]

    # --- numbers for the text
    for cat, a in (("western", aw), ("obs", ao)):
        l = a[a.status == "loaded"]
        N[cat] = {"station_locations_in_table": int(len(st[cat])),
                  "station_days_in_epochs": int(len(a)),
                  "status": {k: int(v) for k, v in a.status.value_counts().items()},
                  "loaded_zero_picks": int((l.npks == 0).sum()),
                  "stations_loaded": int(l.tid.nunique()),
                  "picks_manifest_npks": int(l.npks.sum()),
                  "first_day": str(l.date.min().date()), "last_day": str(l.date.max().date()),
                  "bands_loaded": {k: int(v) for k, v in l.cha.value_counts().items() if v}}
    in_w = q.tid.isin(w.id)
    N["q2025_on_western_stations"] = {"station_days": int(q[in_w].drop_duplicates(["tid", "date"]).shape[0]),
                                      "station_band_days": int(in_w.sum()),
                                      "stations": int(q[in_w].tid.nunique()),
                                      "picks": int(q[in_w].npks.sum()),
                                      "zero_pick_records": int((q[in_w].npks == 0).sum())}

    # ================= Figure 1: maps
    import cartopy.crs as ccrs, cartopy.feature as cf
    sd_w = lw.groupby("tid", observed=True).size()
    sd_o = lo.groupby("tid", observed=True).size()
    ww = w.set_index("id").join(sd_w.rename("sd")).fillna({"sd": 0})
    oo = o.set_index("id").join(sd_o.rename("sd")).fillna({"sd": 0})
    fig = plt.figure(figsize=(12, 5.8))
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.LambertConformal(central_longitude=-115, central_latitude=40))
    ax1.set_extent([-130, -100, 27, 56], ccrs.PlateCarree())
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.Robinson(central_longitude=-150))
    ax2.set_global()
    for ax in (ax1, ax2):
        ax.add_feature(cf.LAND, facecolor="#f2f1ec"); ax.add_feature(cf.COASTLINE, lw=0.4, color="#777")
    ax1.add_feature(cf.STATES, lw=0.3, edgecolor="#aaa")
    none = ww[ww.sd == 0]
    ax1.scatter(none.longitude, none.latitude, s=2, marker="x", lw=0.3, color=GREY, transform=ccrs.PlateCarree(),
                label=f"in table, no station-day loaded ({len(none):,})")
    yes = ww[ww.sd > 0].sort_values("sd")
    sc = ax1.scatter(yes.longitude, yes.latitude, c=np.log10(yes.sd / 365.25), s=4, cmap="viridis",
                     vmin=-1.5, vmax=1.6, transform=ccrs.PlateCarree(), label=f"loaded ({len(yes):,})")
    ax1.legend(loc="lower left", fontsize=7, frameon=False, markerscale=3)
    ax1.set_title("a  western: station-years read and picked, 1986-2026", loc="left")
    yes = oo[oo.sd > 0].sort_values("sd")
    ax2.scatter(yes.longitude, yes.latitude, c=np.log10(yes.sd / 365.25), s=7, cmap="viridis", vmin=-1.5, vmax=1.6,
                transform=ccrs.PlateCarree())
    ax2.set_title(f"b  obs: {len(yes):,} offshore station-locations, 1993-2026", loc="left")
    cb = fig.colorbar(sc, ax=[ax1, ax2], orientation="horizontal", fraction=0.04, pad=0.04, shrink=0.5)
    cb.set_label("station-years loaded (log10)")
    fig.savefig(FIG / "fig1_station_years.png", bbox_inches="tight"); plt.close(fig)

    # ================= Figure 2: what happened to every western station-day, per year; 2025 for comparison
    aw["year"] = aw.date.dt.year
    tab = aw.groupby(["year", "status"], observed=True).size().unstack(fill_value=0).reindex(columns=STATUS_ORDER, fill_value=0)
    qw = q[in_w].copy(); qw["year"] = qw.date.dt.year
    q_sd = qw.drop_duplicates(["tid", "date"]).groupby("year").size()          # station-days, any band
    q_pk = qw.groupby("year").npks.sum()
    rank = {b: i for i, b in enumerate(PRIORITY)}
    q1 = qw.assign(rk=qw.cha.astype(str).map(rank).fillna(99)).sort_values("rk").drop_duplicates(["tid", "date"])
    q_pk1 = q1.groupby("year").npks.sum()
    w_pk = lw.groupby(lw.date.dt.year).npks.sum()
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1]))
    bottom = np.zeros(len(tab))
    for s in STATUS_ORDER:
        ax[0].bar(tab.index, tab[s] / 1e6, bottom=bottom, color=STATUS_COLOR[s], width=0.85, label=s)
        bottom += tab[s].values / 1e6
    ax[0].plot(q_sd.index, q_sd / 1e6, "o-", ms=3, color=C25, lw=1.4, label="2025 run, station-days with a record")
    ax[0].set_ylabel("station-days (M)"); ax[0].legend(fontsize=7, ncol=4, frameon=False, loc="upper left")
    ax[0].set_title("a  western station-days inside FDSN operating epochs, by outcome (availability table)", loc="left")
    ax[1].plot(w_pk.index, w_pk / 1e6, "o-", ms=3, color=C26, label="2026 original (manifest counts)")
    ax[1].plot(q_pk.index, q_pk / 1e6, "o-", ms=3, color=C25, label="2025 instance, every band, same stations")
    ax[1].plot(q_pk1.index, q_pk1 / 1e6, "o--", ms=3, color=C25, mfc="white",
               label="2025 instance, one band per station-day (2026 priority)")
    ax[1].set_ylabel("picks per year (M)"); ax[1].legend(fontsize=8, frameon=False)
    ax[1].set_title("b  picks per year on the western station table", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "fig2_station_day_accounting.png", bbox_inches="tight"); plt.close(fig)
    N["fig2"] = {"status_by_year": {int(y): {k: int(v) for k, v in r.items()} for y, r in tab.iterrows()},
                 "q2025_station_days_by_year": {int(k): int(v) for k, v in q_sd.items()},
                 "q2025_picks_by_year": {int(k): int(v) for k, v in q_pk.items()},
                 "q2025_picks_by_year_one_band": {int(k): int(v) for k, v in q_pk1.items()},
                 "w2026_picks_by_year": {int(k): int(v) for k, v in w_pk.items()}}

    # ================= Figure 3: station-day overlap and pick-count ratio, 2002-2025
    lw26 = lw[(lw.date >= "2002-01-01") & (lw.date < "2025-04-01")]
    k26 = set(zip(lw26.tid.astype(str), lw26.date))
    qd = qw[(qw.date < "2025-04-01")].drop_duplicates(["tid", "date"])
    k25_all = qd[["tid", "date", "npks"]]
    k25_pos = set(zip(qd.loc[qd.npks > 0, "tid"], qd.loc[qd.npks > 0, "date"]))
    k25 = set(zip(qd.tid, qd.date))
    both, only26, only25 = len(k26 & k25), len(k26 - k25), len(k25 - k26)
    m = compare_station_days(aw[(aw.date >= "2002-01-01") & (aw.date < "2025-04-01")], q[in_w])
    m = m[(m.n26 > 0) & (m.n25 > 0)]
    m["year"] = m.date.dt.year
    ratio_y = m.groupby("year").apply(lambda g: pd.Series({"median": np.median(g.n26 / g.n25),
                                                           "q25": np.percentile(g.n26 / g.n25, 25),
                                                           "q75": np.percentile(g.n26 / g.n25, 75)}))
    # 2026-only station-days by reason they were absent in 2025
    only = lw26[[(t, d) not in k25 for t, d in zip(lw26.tid.astype(str), lw26.date)]]
    q_tids = set(q_st.id)
    reason = np.where(~only.tid.astype(str).isin(q_tids), "station not in 2025 table", "station in 2025 table, day absent")
    reason_ct = pd.Series(reason).value_counts()
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    edges = np.logspace(0, 4.5, 60)
    h = ax[0].hist2d(m.n25, m.n26, bins=[edges, edges], cmap="Blues", norm=matplotlib.colors.LogNorm())
    ax[0].plot([1, 3e4], [1, 3e4], color="#555", lw=0.8)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("picks per station-day, 2025 instance"); ax[0].set_ylabel("2026 original")
    ax[0].set_title(f"a  {len(m)/1e6:.1f} M station-days picked by both, same band", loc="left")
    m["ncomp"] = np.where(m.cha == "EH", "EH", "other")
    by_band = m.groupby("cha").apply(lambda g: pd.Series({"median": np.median(g.n26 / g.n25), "n": len(g),
                                                           "q25": np.percentile(g.n26 / g.n25, 25),
                                                           "q75": np.percentile(g.n26 / g.n25, 75)}))
    by_band = by_band[by_band.n >= 10000].sort_values("n", ascending=False)
    xs = np.arange(len(by_band))
    ax[1].bar(xs, by_band["median"], color=[C25 if b in ("EH", "SH") else C26 for b in by_band.index])
    ax[1].errorbar(xs, by_band["median"], yerr=[by_band["median"] - by_band.q25, by_band.q75 - by_band["median"]],
                   fmt="none", ecolor="#333", lw=0.8)
    ax[1].set_xticks(xs); ax[1].set_xticklabels([f"{b}\n{n/1e6:.1f} M" for b, n in by_band.n.items()], fontsize=7.5)
    ax[1].axhline(1, color="#555", lw=0.8); ax[1].set_yscale("log")
    ax[1].set_ylabel("2026 / 2025 picks per station-day")
    ax[1].set_title("b  by band: median and IQR (station-days)", loc="left")
    lab = ["both runs", "2026 only", "2025 only"]
    ax[2].bar(lab, np.array([both, only26, only25]) / 1e6, color=[C26, "#7fb2e5", C25])
    for i, v in enumerate([both, only26, only25]):
        ax[2].text(i, v / 1e6, f"{v/1e6:.2f} M", ha="center", va="bottom", fontsize=8)
    ax[2].set_ylabel("station-days, 2002 to 2025-03 (M)")
    ax[2].set_title("c  western station-days with a record", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "fig3_2025_vs_2026_station_days.png", bbox_inches="tight"); plt.close(fig)
    N["fig3"] = {"station_days_both": both, "only_2026": only26, "only_2025": only25,
                 "only_2026_reason": {k: int(v) for k, v in reason_ct.items()},
                 "only_2025_with_picks": len(k25_pos - k26),
                 "pairs_compared": int(len(m)),
                 "median_ratio_all": float(np.median(m.n26 / m.n25)),
                 "median_ratio_by_year": {int(k): round(float(v), 3) for k, v in ratio_y["median"].items()},
                 "median_ratio_by_band": {b: {"median": round(float(r["median"]), 3), "station_days": int(r.n)}
                                          for b, r in by_band.iterrows()}}
    # station-days with 2025 picks and no 2026 loaded record: what 2026 says about them
    q25d = qd[qd.npks > 0][["tid", "date", "npks"]]
    x = q25d.merge(aw[["tid", "date", "status"]].assign(tid=lambda d: d.tid.astype(str)), on=["tid", "date"], how="left")
    x = x[x.status.astype(object) != "loaded"]
    x["net"] = x.tid.str.split(".").str[0]
    N["fig3"]["only_2025_with_picks_by_2026_status"] = {str(k): int(v) for k, v in
                                                         x.status.astype(object).fillna("outside FDSN epochs").value_counts().items()}
    N["fig3"]["only_2025_picks"] = int(x.npks.sum())
    N["fig3"]["only_2025_top_networks"] = {k: int(v) for k, v in x.net.value_counts().head(15).items()}

    # ================= Figure 4: pick-level comparison at Ridgecrest, CI 2019-07
    key = ["tid", "cha", "pha", "peak"]
    p26 = pd.read_parquet(cache / "western/picks/network=CI/year=2019/month=07").drop_duplicates(key)
    p25 = pd.read_parquet(cache / "quakescope-2025/picks/network=CI/year=2019/month=07").drop_duplicates(key)
    for df in (p26, p25):
        df["date"] = df.peak.dt.floor("D")
    b26 = p26.groupby(["tid", "date"]).cha.first().rename("cha26")
    p25 = p25.join(b26, on=["tid", "date"]); p25 = p25[p25.cha == p25.cha26]
    common = p25[["tid", "date"]].drop_duplicates()
    p26 = p26.merge(common, on=["tid", "date"]); p25 = p25.merge(common, on=["tid", "date"])

    def nearest(a, b, tol=0.5):
        a = a.sort_values("peak"); b = b.sort_values("peak")[["tid", "pha", "peak"]].rename(columns={"peak": "pb"})
        r = pd.merge_asof(a, b, left_on="peak", right_on="pb", by=["tid", "pha"], direction="nearest",
                          tolerance=pd.Timedelta(seconds=tol))
        r["dt"] = (r.peak - r.pb).dt.total_seconds()
        return r
    m26, m25 = nearest(p26, p25), nearest(p25, p26)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    bins = np.linspace(0.2, 1, 17); mid = 0.5 * (bins[1:] + bins[:-1])
    for mm, c, lab in ((m26, C26, "2026 found in 2025"), (m25, C25, "2025 found in 2026")):
        for ph, ls in (("P", "-"), ("S", "--")):
            g = mm[mm.pha == ph]
            ax[0].plot(mid, g.groupby(pd.cut(g.conf, bins), observed=False).dt.apply(lambda s: s.notna().mean()).values,
                       ls, color=c, lw=1.5, label=f"{lab}, {ph}")
    ax[0].set_ylim(0, 1); ax[0].set_xlabel("conf"); ax[0].set_ylabel("fraction matched within 0.5 s")
    ax[0].legend(fontsize=7, frameon=False); ax[0].set_title("a  agreement by confidence", loc="left")
    for df, c, lab in ((p26, C26, "2026 original"), (p25, C25, "2025 instance")):
        for ph, ls in (("P", "-"), ("S", "--")):
            ax[1].hist(df.loc[df.pha == ph, "conf"], bins=np.linspace(0.2, 1, 41), histtype="step", ls=ls,
                       color=c, lw=1.3, label=f"{lab} {ph}")
    ax[1].set_yscale("log"); ax[1].set_xlabel("conf"); ax[1].legend(fontsize=7, frameon=False)
    ax[1].set_title("b  confidence distributions", loc="left")
    stats = {}
    for ph, ls in (("P", "-"), ("S", "--")):
        g = m26[(m26.pha == ph) & m26.dt.notna() & (m26.conf >= 0.5)]
        ax[2].hist(g.dt * 1e3, bins=np.linspace(-500, 500, 101), histtype="step", ls=ls, color="#333", lw=1.3,
                   label=f"{ph}: median {g.dt.median()*1e3:+.0f} ms, |dt| {g.dt.abs().median()*1e3:.0f} ms")
        stats[ph] = {"median_ms": round(g.dt.median() * 1e3, 1), "median_abs_ms": round(g.dt.abs().median() * 1e3, 1),
                     "n_matched_conf05": int(len(g))}
    ax[2].set_yscale("log"); ax[2].set_xlabel("2026 minus 2025 (ms), conf >= 0.5")
    ax[2].legend(fontsize=7, frameon=False); ax[2].set_title("c  timing of matched picks", loc="left")
    fig.suptitle("CI network, July 2019 (Ridgecrest), station-days and bands picked by both runs", x=0.01, ha="left", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig4_ridgecrest_pick_agreement.png", bbox_inches="tight"); plt.close(fig)
    N["fig4"] = {"timing": stats}
    for thr in (0.2, 0.5):
        for ph in "PS":
            x = m26[(m26.pha == ph) & (m26.conf >= thr)]; y = m25[(m25.pha == ph) & (m25.conf >= thr)]
            N["fig4"][f"{ph}_conf{thr}"] = {"n2026": int(len(x)), "found_in_2025": round(float(x.dt.notna().mean()), 3),
                                           "n2025": int(len(y)), "found_in_2026": round(float(y.dt.notna().mean()), 3)}

    # ================= Figure 5: OBS, PickBlue vs instance on the same ocean-bottom station-days
    o_ids = set(o.id)
    mo = compare_station_days(ao, q[q.tid.isin(o_ids)])
    mo = mo[(mo.n26 > 0) | (mo.n25 > 0)]
    od = o.set_index("id")
    mo["depth"] = -od.elevation.reindex(mo.tid).values
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    g = mo[(mo.n26 > 0) & (mo.n25 > 0)]
    edges = np.logspace(0, 4, 50)
    ax[0].hist2d(g.n25, g.n26, bins=[edges, edges], cmap="Greens", norm=matplotlib.colors.LogNorm())
    ax[0].plot([1, 1e4], [1, 1e4], color="#555", lw=0.8); ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("2025 instance, picks per station-day"); ax[0].set_ylabel("2026 PickBlue")
    ax[0].set_title(f"a  {len(g):,} OBS station-days picked by both", loc="left")
    lo2 = lo.assign(depth=-od.elevation.reindex(lo.tid.astype(str)).values, cha=lo.cha.astype(str))
    for b, c in (("HH", C26), ("BH", "#7570b3"), ("EL", C_OBS), ("EH", "#e6ab02")):
        gg = lo2[lo2.cha == b]
        if len(gg):
            per = gg.groupby("tid").agg(depth=("depth", "first"), n=("npks", "median"))
            ax[1].scatter(per.depth, per.n, s=6, alpha=0.6, color=c, label=f"{b} ({len(per)})")
    ax[1].set_yscale("log"); ax[1].set_xlabel("water depth (m)"); ax[1].set_ylabel("median picks per station-day")
    ax[1].legend(fontsize=7, frameon=False, title="band"); ax[1].set_title("b  2026 obs by depth and band", loc="left")
    lo2["year"] = lo2.date.dt.year
    yr = lo2.groupby("year").agg(sd=("tid", "size"), st=("tid", "nunique"))
    ax[2].bar(yr.index, yr.sd / 1e3, color=C_OBS); ax[2].set_ylabel("station-days loaded (k)")
    ax[2].set_title("c  2026 obs station-days per year", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "fig5_obs.png", bbox_inches="tight"); plt.close(fig)
    # 2026-only by set difference on (tid, date), any band: `mo` is an inner merge on the same band,
    # so station-days with no 2025 record at all are not in it.
    qo = q[q.tid.isin(o_ids)]
    k25_any = set(zip(qo.tid.astype(str), qo.date))
    k25_pos = set(zip(qo.loc[qo.npks > 0, "tid"].astype(str), qo.loc[qo.npks > 0, "date"]))
    lo_pos = lo[lo.npks > 0]
    k26_pos = set(zip(lo_pos.tid.astype(str), lo_pos.date))
    N["fig5"] = {"obs_station_days_both": int(len(g)), "median_ratio_pickblue_over_instance": float(np.median(g.n26 / g.n25)),
                 "obs_station_days_with_2026_picks": len(k26_pos),
                 "obs_station_days_2026_picks_no_2025_record": len(k26_pos - k25_any),
                 "obs_station_days_2026_picks_no_2025_picks": len(k26_pos - k25_pos)}

    # ================= Figure 6: daily picks at example stations, both runs (cf. Ni et al. 2025 Fig. 4)
    examples = ["CI.CGO.", "UW.BVW.", "UW.RCM.", "NC.KCT.", "NN.RRK."]
    examples = [e for e in examples if e in set(lw.tid.astype(str))][:6]
    fig, axs = plt.subplots(len(examples), 1, figsize=(11, 1.5 * len(examples)), sharex=True)
    for ax, tid in zip(np.atleast_1d(axs), examples):
        a = lw[lw.tid == tid].groupby("date").npks.sum()
        b = q[q.tid == tid].groupby("date").npks.sum()
        ax.plot(b.index, b.rolling(28, min_periods=7).mean(), color=C25, lw=0.9, label="2025 instance, all bands")
        ax.plot(a.index, a.rolling(28, min_periods=7).mean(), color=C26, lw=0.9, label="2026 original")
        bands = "/".join(sorted(set(lw[lw.tid == tid].cha.astype(str))))
        ax.set_yscale("log"); ax.set_ylabel("picks/day")
        ax.set_title(f"{tid}  (2026 band {bands}; 28-day mean)", loc="left", fontsize=8.5, pad=2)
    np.atleast_1d(axs)[0].legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig6_example_stations.png", bbox_inches="tight"); plt.close(fig)

    (FIG / "numbers.json").write_text(json.dumps(N, indent=1, default=str))
    print(json.dumps({k: v for k, v in N.items() if k not in ("fig2",)}, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fetch", "figures", "all"])
    ap.add_argument("--cache", type=Path, default=Path("paper_cache"))
    a = ap.parse_args()
    if a.stage in ("fetch", "all"):
        fetch(a.cache)
    if a.stage in ("figures", "all"):
        figures(a.cache)
