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

# TODO !!!TEMPORARY!!! load model weights from Jannes personal data repo.
# We may only include those needed
RUN wget https://munchmeyer.de/data/sbmodels_v3_230614.tgz \
    && mkdir models \
    && mkdir -p /root/.seisbench/models/v3/  \
    && tar xvf sbmodels_v3_230614.tgz -C models/ \
    && cp -r models/phasenet /root/.seisbench/models/v3/phasenet \
    && rm -rf models/ \
    && rm -rf sbmodels_v3_230614.tgz

# Send and merge local weights into the image
COPY models/v3/ /root/.seisbench/models/v3/


# command to run on container start
ENTRYPOINT ["python", "-m", "src.picker"]
