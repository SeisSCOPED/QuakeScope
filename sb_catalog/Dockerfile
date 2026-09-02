# set base image (host OS)
FROM python:3.12

# set the working directory in the container
WORKDIR /code

# install dependencies
# pytorch-cpu. torchvision and torchaudio are not imported anywhere in the
# pipeline - seisbench needs only torch - and they are not small.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# SeisBench.
# pyarrow is required by parquet_writer, which is the v3 output path - without
# it a picking job ImportErrors before it reads a byte.
#
# Deliberately NOT installed. Verified by importing worker and picker in a venv
# built without them: none appears in sys.modules, and the tests pass.
#   pyocto       only `picker.run_association`, which needs DocumentDB
#   pymongo      only the DocumentDB backend, now isolated in `mongo_db`
#   ipython      interactive only
#   joblib       unused since the nested Parallel was removed
#   torchvision  seisbench needs torch alone, and these are not small
#   torchaudio
# tqdm is also dropped from this line but still arrives: it is a declared
# seisbench dependency, so pip installs it either way.
#
# The two DocumentDB imports are local to their call sites, so the picker runs
# without either driver. Re-adding one should mean a new import, not a habit.
RUN pip install seisbench s3fs boto3==1.35.81 earthscope-sdk==1.0.0b0 pyarrow

# get AWS DocumentDB public key
RUN wget https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem

# copy the content of the local src directory to the working directory
COPY src/ src/

# Pretrained weights. Only the ones a campaign actually selects, committed in
# the repo, copied straight in - so the build fetches no weights over the
# network at all and the image cannot vary between builds.
#
#   jma_wc          campaigns 1-3 (SCEDC, NCEDC, EarthScope)
#   obs             campaign 4 (OBS)
#   original        campaign 5 (western)
#   quakescope2026  the v7 fine-tune; not used this run, and not published
#                   upstream, so the repo is its only source
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
