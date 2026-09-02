"""Scatter the 2025 catalogue's associated events. A 2025 artefact.

**Not part of the 2026 workflow, and not runnable in the worker image.** It
reads the `events` collection, which only the DocumentDB backend writes and only
the association step fills - and the 2026 run is picker-only, keeps its state in
S3, and ships neither `pymongo` nor an associator. Nothing imports this module.

Kept, and renamed rather than deleted, because it is the only record of how the
2025 catalogue was looked at. To run it you need the analysis-side dependencies
(`pymongo`, `matplotlib`) and a live DocumentDB endpoint; the tutorials' pixi
environment has them.

The 2026 equivalent is `scripts/campaign_dashboard.py`, which reads Parquet from
S3 and draws its own SVG.
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from .mongo_db import SeisBenchDatabase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("db_uri", type=str)
    parser.add_argument("--database", type=str, default="tutorial")
    args = parser.parse_args()

    plot_events(args.db_uri, args.database)


def plot_events(db_uri: str, database: str, savefig: bool = False) -> None:
    db = SeisBenchDatabase(db_uri, database)

    cursor = db.database["events"].find()
    events = pd.DataFrame(list(cursor))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cb = ax.scatter(events["x"], events["y"], c=events["depth"], vmin=0, s=4)
    ax.set_aspect("equal")
    ax.set_xlabel("East [km]")
    ax.set_ylabel("North [km]")
    cbar = fig.colorbar(cb, label="Depth [km]")
    cbar.ax.invert_yaxis()
    if savefig:
        fig.savefig("events.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
