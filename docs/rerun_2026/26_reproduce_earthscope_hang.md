# 26 — Reproducing the EarthScope read hang by hand

A step-by-step to demonstrate the failure in item **0g** with nothing but the
AWS CLI on an EC2 instance. No pipeline, no container, no Python.

**Why EC2 and not a laptop.** EarthScope permit `ListObjectsV2` from anywhere
but `GetObject` **only from us-east-2**. From a laptop a read returns 403, which
looks like a permissions problem and proves nothing about the hang. The failure
only appears from inside the region.

**Do not use the campaign's refresh token for this.** `earthscope_sdk`'s refresh
grant saves a rotated token to local SDK state rather than back to Secrets
Manager, so using `quakescope/earthscope-refresh-token` on a scratch box can
invalidate the credential every campaign job depends on. Log in interactively
instead — step 3.

---

## 1. Launch a scratch instance in us-east-2

Region matters; nothing else does. `t3.micro` is enough — this is a request-shape
test, not a throughput test.

```bash
aws ec2 run-instances --region us-east-2 \
  --image-id resolve:ssm:/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
  --instance-type t3.micro \
  --metadata-options "HttpTokens=required" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=es-hang-probe}]' \
  --count 1
```

Note the `InstanceId`, then connect. Session Manager avoids opening SSH:

```bash
aws ssm start-session --region us-east-2 --target i-XXXXXXXXXXXX
```

If SSM is not set up on the instance profile, launch it into a subnet with a
public IP and use `--key-name` plus SSH instead. Either is fine.

## 2. Confirm you are actually in us-east-2

The whole point of the exercise. On the instance:

```bash
TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/placement/region; echo
```

Must print `us-east-2`.

## 3. Get EarthScope credentials interactively

```bash
sudo dnf install -y python3-pip
pip3 install --user "earthscope-sdk==1.0.0b0"
export PATH="$HOME/.local/bin:$PATH"

es login          # opens a device-code flow; follow the URL it prints
es user get-aws-credentials --role s3-miniseed-v2
```

That prints temporary keys. Export them:

```bash
eval "$(es user get-aws-credentials --role s3-miniseed-v2 --format env)"
# or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN by hand
export AWS_DEFAULT_REGION=us-east-2
```

## 4. Show that LIST works

This is the part that already works in production — the hit-rate survey listed
310,347 restricted station-days with zero errors.

```bash
AP=earthscope-mseed-v2-4fdodyzpsz8u8uyi3pa9qsw9oid1suse2a-s3alias

time aws s3api list-objects-v2 \
  --bucket "$AP" \
  --prefix miniseed/AV/2018/135/ \
  --max-items 10 \
  --query 'Contents[].{Key:Key,Size:Size}' --output table
```

Expect a table of objects in under a second. **Record the `Key` and `Size` of
one of them** — the size matters for step 6.

## 5. Show that HEAD works

```bash
KEY=miniseed/AV/2018/135/ACH.AV.2018.135      # substitute a real key from step 4

time aws s3api head-object --bucket "$AP" --key "$KEY"
```

Expect `ContentLength` and `LastModified` back promptly. If this hangs, the
problem is earlier than GET and that is itself the finding.

## 6. Show that GET hangs

This is the failure. Use an explicit short timeout so the shell returns rather
than sitting there:

```bash
time aws s3api get-object \
  --bucket "$AP" --key "$KEY" \
  --cli-read-timeout 60 --cli-connect-timeout 15 \
  /tmp/out.mseed
```

**Expected in the pipeline:** no bytes, no error, until something gives up. The
worker's own bound is `STATION_DAY_TIMEOUT=900`, i.e. fifteen minutes — longer
than the mean time to interruption in this Spot pool, so in production Spot kills
the worker before the timeout fires and the claim strands.

Try a 1 KB ranged read too. If the range succeeds and the full GET does not, the
problem is transfer rather than authorisation:

```bash
time aws s3api get-object \
  --bucket "$AP" --key "$KEY" --range bytes=0-1023 \
  --cli-read-timeout 60 /tmp/head.bin && ls -l /tmp/head.bin
```

## 7. Control: the same code path against Open Data

Open Data needs no credentials and reads at 90.3 MB/s in production, so if this
also hangs the problem is the instance or the network, not EarthScope.

```bash
time aws s3 cp --no-sign-request \
  s3://earthscope-geophysical-data/miniseed/AK/2020/309/PS09.AK.2020.309 \
  /tmp/open.mseed
```

Expect ~140 MB in a couple of seconds.

## 8. If you want the layer-by-layer breakdown

The pipeline already carries a diagnostic that walks DNS, TCP, a signed HEAD, a
1 KB ranged GET and a full GET, comparing the restricted access point against an
Open Data control over the identical code path:

```bash
aws batch submit-job --region us-east-2 \
  --job-name diag-es --job-queue niyiyu_earthscope_missing_station \
  --job-definition quakescope_2026_earthscope:8 \
  --container-overrides '{"command":["diag-earthscope","AV","CC","UW"]}'
```

It runs in the container with the secret already injected, so it needs no
credential handling of its own.

## 9. Tear down

```bash
aws ec2 terminate-instances --region us-east-2 --instance-ids i-XXXXXXXXXXXX
```

---

## What each outcome would mean

| observation | reading |
|---|---|
| LIST ok, HEAD ok, ranged GET ok, full GET hangs | transfer-level: throttling, a stalled connection, or object size |
| LIST ok, HEAD ok, ranged GET hangs | request-level: the access point is answering metadata but not data |
| LIST ok, HEAD 403 | entitlement, not a hang — a different problem from 0g |
| everything hangs including Open Data | the instance or its network, not EarthScope |
| all of it works from EC2 | the failure is specific to Fargate's networking, not the region — look at the public-subnet/IGW path and the absence of an S3 gateway endpoint |

That last row is worth taking seriously. The compute environment runs tasks in
**public subnets behind an Internet Gateway with no NAT and no S3 gateway
endpoint** ([OPTIMISE.md](OPTIMISE.md) item 4a). A plain EC2 instance in a
default VPC does not reproduce that path, so a clean EC2 result would point at
the Fargate networking rather than exonerate EarthScope.
