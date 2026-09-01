# set base image (host OS)
FROM python:3.12

# set the working directory in the container
WORKDIR /code

# install dependencies
# pytorch-cpu
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# SeisBench
# pyarrow is required by parquet_writer, which is the v3 output path - without
# it a picking job ImportErrors before it reads a byte.
RUN pip install seisbench pyocto ipython s3fs pymongo boto3==1.35.81 tqdm joblib earthscope-sdk==1.0.0b0 pyarrow

# get AWS DocumentDB public key
RUN wget https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem

# copy the content of the local src directory to the working directory
COPY src/ src/

# Pretrained PhaseNet weights, from SeisBench's own model repository -
# seisbench.remote_model_root, https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/models/v3/
#
# This replaces a 156 MB tarball from a personal site dated 2023-06-14. Its 40
# PhaseNet files were checked and are byte-identical to the official ones, so
# nothing regressed in switching - but it predates six weights published since,
# and **jma_wc is one of them**. jma_wc drives campaigns 1-3, 78M of the
# campaign's 113M station-days, so every worker was instead downloading it
# during startup: 1,500 cold-start fetches of an unpinned external host in the
# critical path. Observed in the 2026-09-01 SCEDC smoke test.
#
# The list is explicit rather than list_pretrained(), so a build is reproducible
# and a weight appearing or vanishing upstream cannot silently change the image.
# The build fails loudly if any is unreachable, which is the intent - a missing
# weight should not become a runtime download on 1,500 workers.
RUN python -c "\
import seisbench.models as sbm; \
ws = ['aq2009','diting','ethz','geofon','instance','iquique','jma','jma_wc', \
      'lendb','neic','obs','original','phasenet_sn','pisdl','scedc','stead','volpick']; \
[sbm.PhaseNet.from_pretrained(w) for w in ws]; \
print('baked %d PhaseNet weights' % len(ws))"

# Send and merge local weights into the image.
#
# Must stay AFTER the fetch above so these win on conflict. Every weight a 2026
# campaign selects - jma_wc, obs, original, quakescope2026 - is committed in the
# repo, so the campaign path stays hermetic even if upstream changes underneath
# the fetch. quakescope2026 is only here; it is not published upstream.
# See models/v3/phasenet/README.md.
COPY models/v3/ /root/.seisbench/models/v3/


# command to run on container start
ENTRYPOINT ["python", "-m", "src.picker"]
