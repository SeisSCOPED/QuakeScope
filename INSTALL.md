# QuakeScope: Cloud Deployment Installation Guide

This guide covers setup for **production cloud deployment** on AWS. For local development and tutorials, see [INSTALL_TUTORIALS.md](INSTALL_TUTORIALS.md).

## Quick Start

```bash
git clone https://github.com/SeisSCOPED/QuakeScope.git
cd QuakeScope

# See full runbook for step-by-step AWS setup:
cat docs/rerun_2026/README.md
```

## System Requirements

### AWS Account & Permissions
- AWS account with appropriate IAM permissions (see [docs/rerun_2026/01_aws_basics.md](docs/rerun_2026/01_aws_basics.md))
- CloudFormation, EC2, Batch, Fargate, DocumentDB, S3, ECR access
- ~$2,000-5,000/month for continuous picking (Fargate Spot + storage)

### Local Development Environment
- Docker 20.10+
- Python 3.9+ (for CI/CD scripts)
- AWS CLI v2 configured with credentials
- Git with SSH access to GitHub

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/SeisSCOPED/QuakeScope.git
cd QuakeScope
```

### 2. Set Up AWS Environment

See [docs/rerun_2026/01_aws_basics.md](docs/rerun_2026/01_aws_basics.md) for:
- AWS region configuration (recommended: `us-east-2`)
- IAM role setup
- VPC and security group configuration
- S3 bucket access

Quick setup:
```bash
aws configure

# Verify access
aws s3 ls scedc-pds/continuous_waveforms/ --no-sign-request
```

### 3. Prepare Custom Weights

See [docs/rerun_2026/02_weights_and_container.md](docs/rerun_2026/02_weights_and_container.md) for:
- Converting phasenet-retrain checkpoint to SeisBench format
- Placing weights in `sb_catalog/models/`
- Building Docker image with custom weights

```bash
cd sb_catalog/models/v3/phasenet

# Convert v7 checkpoint (example)
python convert_checkpoint.py \
    --checkpoint /path/to/best.pt \
    --name quakescope2026 \
    --verify

# Verify weights are in place
ls -la *.v1
```

### 4. Build & Push Docker Image

```bash
# Build container
docker build -t quakescope:latest .

# Tag for ECR
aws ecr get-login-password --region us-east-2 | docker login \
    --username AWS --password-stdin <account-id>.dkr.ecr.us-east-2.amazonaws.com

docker tag quakescope:latest \
    <account-id>.dkr.ecr.us-east-2.amazonaws.com/quakescope:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-east-2.amazonaws.com/quakescope:latest
```

### 5. Set Up DocumentDB

See [docs/rerun_2026/03_documentdb.md](docs/rerun_2026/03_documentdb.md) for:
- Creating DocumentDB cluster
- Configuring credentials and backups
- Setting up database and collections for picks/events

```bash
# Example: Create cluster (via CloudFormation)
aws cloudformation create-stack \
    --stack-name quakescope-documentdb \
    --template-body file://templates/documentdb.yaml \
    --region us-east-2
```

### 6. Configure Batch/Fargate

See [docs/rerun_2026/04_batch_setup.md](docs/rerun_2026/04_batch_setup.md) for:
- Creating Batch compute environments
- Setting up job queues
- Configuring Fargate Spot for cost optimization

```bash
# Create compute environment
aws batch create-compute-environment \
    --compute-environment-name quakescope-fargate-spot \
    --type MANAGED \
    --state ENABLED \
    --compute-resources type=FARGATE_SPOT,maxvCpus=256,...
```

### 7. Submit Picking Jobs

See [docs/rerun_2026/05_submitting_jobs.md](docs/rerun_2026/05_submitting_jobs.md) and [notebooks/3_submit_job.ipynb](notebooks/3_submit_job.ipynb) for:
- Job configuration (date ranges, stations, thresholds)
- Submitting to Batch
- Monitoring job progress
- Retrieving results from DocumentDB

```bash
# Example: Submit job for specific date range
python -c "
import quakescope_submit
quakescope_submit.submit_picking_job(
    start_date='2019-07-04',
    end_date='2019-07-06',
    weight='quakescope2026',
    database='quakescope_2026'
)
"
```

## Docker Container Structure

The Dockerfile includes:
- **Base**: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`
- **Dependencies**: ObsPy, SeisBench, PyOcto, pymongo
- **Custom weights**: Baked into `/root/.seisbench/models/v3/phasenet/`
- **Entrypoint**: Picking pipeline callable from AWS Batch

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install seismic packages
RUN pip install obspy seisbench pyocto pymongo s3fs

# Copy custom weights
COPY sb_catalog/models/v3/phasenet/*.v1 \
    /root/.seisbench/models/v3/phasenet/

# Set entry point
ENTRYPOINT ["python", "-m", "sb_catalog.src.picking"]
```

## Configuration Files

### Environment Variables (in Batch job submission)

```bash
export MODEL_NAME="quakescope2026"        # Custom v7 weights
export P_THRESHOLD="0.2"                 # PhaseNet P detection threshold
export S_THRESHOLD="0.2"                 # PhaseNet S detection threshold
export DATABASE_URL="mongodb+srv://..."  # DocumentDB connection
export DATABASE_NAME="quakescope_2026"   # Database name
export S3_BUCKET="scedc-pds"            # Data source
```

### Job Configuration (JSON)

```json
{
  "jobName": "quakescope-ridgecrest-2019",
  "jobQueue": "quakescope-fargate-spot",
  "jobDefinition": "quakescope:1",
  "containerOverrides": {
    "environment": [
      {"name": "START_DATE", "value": "2019-07-04"},
      {"name": "END_DATE", "value": "2019-07-06"},
      {"name": "NETWORKS", "value": "CI,BK"},
      {"name": "DATABASE_NAME", "value": "quakescope_2026"}
    ],
    "vcpus": 4,
    "memory": 8000,
    "resourceRequirements": [
      {"type": "VCPU", "value": "4"},
      {"type": "MEMORY", "value": "8000"}
    ]
  }
}
```

## Troubleshooting

### Docker Build Issues
```bash
# Rebuild without cache
docker build --no-cache -t quakescope:latest .

# Verify weights are copied
docker run --rm quakescope:latest \
    python -c "import seisbench.models as sbm; print(sbm.PhaseNet.from_pretrained('quakescope2026'))"
```

### AWS Batch Job Failures
```bash
# Check job logs in CloudWatch
aws logs tail /aws/batch/job --follow

# Inspect failed job status
aws batch describe-jobs --jobs <job-id> --region us-east-2
```

### DocumentDB Connectivity
```bash
# Test connection from EC2/Fargate
python -c "
import pymongo
client = pymongo.MongoClient('<connection-string>')
print(client.list_database_names())
"
```

### S3 Data Access
```bash
# Verify anonymous S3 access
aws s3 ls scedc-pds/continuous_waveforms/ --no-sign-request

# Check bucket permissions
aws s3api get-bucket-policy --bucket scedc-pds
```

## Next Steps

1. **Complete setup**: Follow [docs/rerun_2026/README.md](docs/rerun_2026/README.md) step-by-step
2. **Validate models**: Run smoke tests ([INSTALL_TUTORIALS.md](INSTALL_TUTORIALS.md))
3. **Monitor deployment**: Set up CloudWatch dashboards ([docs/rerun_2026/06_monitoring.md](docs/rerun_2026/06_monitoring.md))
4. **Query results**: Use [notebooks/4_check_database.ipynb](notebooks/4_check_database.ipynb)

## Support

- **Documentation**: [docs/](docs/) folder
- **Issues**: https://github.com/SeisSCOPED/QuakeScope/issues
- **Discussions**: https://github.com/SeisSCOPED/QuakeScope/discussions
- **Email**: mdenolle@uw.edu

## References

- AWS Batch documentation: https://docs.aws.amazon.com/batch/
- Amazon DocumentDB: https://docs.aws.amazon.com/documentdb/
- ECR repositories: https://docs.aws.amazon.com/ecr/
- SeisBench: https://seisbench.readthedocs.io/
