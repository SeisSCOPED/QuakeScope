# 03 — DocumentDB: reuse the cluster, new database, provenance

## 1. Reuse or new? (the decision, and why)

**Reuse the existing DocumentDB cluster. Create a new *database* inside it
for this run** (a DocumentDB/MongoDB cluster holds many independent
databases; a new one is created implicitly the first time you write to it).

Why you must NOT write the new run into the old database (e.g. `earthscope`):

- **The skip logic would silently skip your whole campaign.** Before loading
  each station-day, the picker checks the `picks_record` collection for an
  entry with that station/channel/year/day — *regardless of which model or
  weight produced it* (`utils.py: get_picks_record`). Every station-day
  already picked in the old campaign would be skipped.
- Identical picks would collide with the unique index on
  `(tid, cha, pha, peak)` and be dropped, mixing the two catalogs.

Why not a brand-new cluster: extra ~$200/month, extra networking setup, and
no benefit — databases inside one cluster are fully independent, and the
`sb_runs` provenance (below) plus the database name give you a clean
separation.

**Suggested name: `quakescope2026`.** Use it everywhere `--database` appears.

## 2. Check the cluster still exists

Console → search **DocumentDB** (region us-east-2) → **Clusters**.

- **Cluster there and "Available"** → copy the connection endpoint
  (click the cluster → Connectivity & security) and go to step 3.
- **Cluster stopped** → select it → Actions → Start. (Clusters auto-restart
  after 7 days stopped, beware.)
- **No cluster** → check **Snapshots** in the left menu. If a final snapshot
  exists from the last campaign, **Restore** it (choose the same VPC and
  security group as the EC2 controller, instance class `db.r6g.large` is
  plenty). If there is no snapshot either, create a new cluster
  (Clusters → Create → engine defaults, 1 instance, `db.r6g.large`,
  set a master username/password and save them in your password manager) —
  and note the old picks are then gone unless someone exported them.

## 3. The controller EC2 instance

DocumentDB accepts connections **only from inside its VPC** — your laptop
cannot reach it directly. That's why the workflow uses a small EC2 instance
("controller") in the same VPC + security group, where you run notebooks 2–4
and submit jobs.

Console → **EC2** → **Instances** (us-east-2):

- If the old controller still exists (stopped), select → Instance state →
  Start, then connect: either click **Connect → EC2 Instance Connect**
  (browser terminal, zero setup) or `ssh -i <key.pem> ec2-user@<public-ip>`.
- If not, launch a new one: **Launch instance** → Amazon Linux 2023,
  `t3.medium`, create/download a key pair, **same VPC and security group as
  the DocumentDB cluster** (or default VPC + add the default security group
  to the DB cluster), 30 GB disk. Then on the instance:

```bash
sudo yum install -y git
git clone https://github.com/SeisSCOPED/QuakeScope.git
curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && bash miniconda.sh -b -p ~/miniconda
~/miniconda/bin/pip install pymongo pandas numpy boto3 pytz tqdm jupyter
wget https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem
aws configure   # same access key as your laptop
```

- The **security group** of the DB must allow inbound TCP **27017** from the
  controller's security group, and — important for Batch — **from the
  Fargate jobs' security group too** (they were the same group last time;
  keep it that way).

## 4. Fill in `parameters.py`

Edit `sb_catalog/src/parameters.py` (on the controller, and mirror it on
your laptop copy). The URI format:

```python
DOCDB_ENDPOINT_URI = (
    "mongodb://<username>:<password>@"
    "<cluster-endpoint>:27017/"
    "?tls=true&tlsCAFile=global-bundle.pem"
    "&retryWrites=false"
)
```

- `<cluster-endpoint>` from the DocumentDB console (looks like
  `xxxx.cluster-yyyy.us-east-2.docdb.amazonaws.com`).
- `global-bundle.pem` must exist in the directory you run from (the wget
  above; the Docker image already contains it at `/code/`).
- Password with special characters must be URL-encoded.

## 5. Populate station metadata (notebook 2)

On the controller, run
[notebooks/2_prepare_station_metadata.ipynb](../../notebooks/2_prepare_station_metadata.ipynb)
(via `jupyter nbconvert --execute`, JupyterLab, or paste into ipython) —
**changing the database name** in the connect cell:

```python
db = SeisBenchDatabase(DOCDB_ENDPOINT_URI, "quakescope2026")
```

This loads every `networks/*.zip` station file into the new database's
`stations` collection (~a few minutes) and creates the indexes. Verify:

```python
db.get_stations(None, "BK,NC,CI,UW")   # should return a DataFrame of stations
```

## 6. How provenance works (what `sb_runs` gives you)

Every picking job writes one document to the `sb_runs` collection the first
time it inserts a pick:

```
{ _id: <run_id>, model: "PhaseNet", weight: "quakescope2026",
  p_threshold: 0.2, s_threshold: 0.2, components_loaded: "ZNE12",
  seisbench_version: "...", weight_version: "...", timestamp: ... }
```

and every document in `picks`, `classifies`, and `picks_record` carries that
`rid`. So within `quakescope2026`, you can always answer "which weights made
this pick" with a join on `rid`. This is your provenance chain:
**database name = campaign, `sb_runs` = exact configuration per job.**

Note: the QuakeXNet *classifier* weight name is not recorded in `sb_runs`
(only the picker's). The database name + the git tag of the image is what
ties classifier output to its weights — one more reason to use a fresh
database name and to pin the image SHA in the job definition.

Next: [04_batch_setup.md](04_batch_setup.md)
