# The campaign worker. Kept deliberately minimal: this image is pulled for every
# Batch task, and a campaign is ~250,000 shards across 100k+ job launches, so
# anything carried here is carried that many times.
#
# Everything needed only to READ the catalogue afterwards - plots, notebooks,
# the DocumentDB client, cross-weight benchmarking - lives in Dockerfile.analysis
# instead. Adding a package here should mean a new import on the picking path.

# -slim, not the full python image: the Debian-based `python:3.12` carries a
# compiler toolchain and dev headers this never uses. Every dependency below
# resolves to a manylinux wheel for cp312 (verified: 68 packages, zero sdists),
# so nothing is built from source and the toolchain is dead weight.
FROM python:3.12-slim

WORKDIR /code

# --no-cache-dir throughout: pip's download cache is written into the layer and
# then never read again, because each RUN is a separate layer.
#
# torch only. seisbench does not import torchvision or torchaudio, and they are
# not small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# pyarrow is required by parquet_writer, which is the v3 output path - without
# it a picking job ImportErrors before it reads a byte.
#
# Deliberately NOT installed. Verified absent by resolving this exact line for
# cp312 and by importing worker and picker in a venv built without them: none
# appears in sys.modules, and the tests pass.
#   pyocto       only `picker.run_association`, which needs DocumentDB
#   pymongo      only the DocumentDB backend, isolated in `mongo_db`
#   ipython      interactive only
#   joblib       unused since the nested Parallel was removed
#   torchvision  seisbench needs torch alone, and neither is small
#   torchaudio
#
# What CANNOT be removed, though it looks removable: matplotlib, pytz, lxml,
# sqlalchemy, pillow and tqdm are all pulled in transitively - obspy requires
# matplotlib, seisbench requires obspy. Dropping them would mean dropping
# obspy, which is the waveform layer. This line resolves to 68 packages;
# trimming further means changing what the picker imports, not what is listed
# here.
RUN pip install --no-cache-dir \
    seisbench s3fs boto3==1.35.81 earthscope-sdk==1.0.0b0 pyarrow

# NOT fetched here: truststore.pki.rds.amazonaws.com/global/global-bundle.pem.
# It is the RDS trust store for DocumentDB, it is referenced nowhere in the
# source, and a v3 campaign has no database to trust. It lives in the analysis
# image, which does talk to DocumentDB.

# copy the content of the local src directory to the working directory
COPY src/ src/

# Pretrained weights, committed in the repo and copied straight in - so the
# build fetches no weights over the network at all and the image cannot vary
# between builds.
#
#   jma_wc          campaigns 1-3 (SCEDC, NCEDC, EarthScope)
#   obs             campaign 4 (OBS)
#   original        campaign 5 (western)
#   quakescope2026  the v7 fine-tune; not used this run, and not published
#                   upstream, so the repo is its only source
#
# All four are ~11 MB together, which is noise against torch, and dropping the
# unpublished one would leave it with no source inside any worker - so this
# stays whole rather than being trimmed per campaign.
#
# This replaces a 156 MB tarball from a personal site dated 2023-06-14. Its 40
# PhaseNet files were byte-identical to the current official ones, so nothing
# regressed in dropping it - but it predated six weights published since, and
# jma_wc was one of them. jma_wc drives 78M of the campaign's 113M station-days,
# so every worker downloaded it from hifis-storage.desy.de during startup
# instead. Observed in the 2026-09-01 SCEDC smoke test.
#
# Anything else SeisBench offers still works; it just downloads on first use,
# which is fine for ad-hoc runs and never happens on a campaign path. Note that
# picker.py's own --weight default is `instance`, which is NOT here - the `work`
# subcommand used by campaigns defaults to jma_wc and is unaffected.
#
# To add a weight, see models/v3/phasenet/README.md.
COPY models/v3/ /root/.seisbench/models/v3/


# command to run on container start
ENTRYPOINT ["python", "-m", "src.picker"]
