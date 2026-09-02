# 26 — Reproducing the EarthScope read denial by hand

> ## ✅ RESOLVED 2026-09-02 — and it was our bug
>
> The denial reproduced below is real, but the conclusion this document drew
> from it was wrong. It is **not** an entitlement gap, and **no request should
> be sent to EarthScope.**
>
> An unscoped credential for `s3-miniseed-v2` grants `s3:ListBucket` but not
> `s3:GetObject`. Adding one query parameter to the token exchange fixes it:
>
> ```python
> client.user.get_aws_credentials(role="s3-miniseed-v2", network="FDSN:AV")
> ```
>
> Verified interactively from EC2 in us-east-2: the identical object that
> returns `AccessDenied` unscoped returns bytes when scoped. Temporary networks
> (codes starting with a digit or X/Y/Z) additionally need `year=`.
>
> Fixed in `s3_helper.py` via `es_scope()`, requiring `earthscope-sdk>=1.8.0`
> — the first release that passes query parameters through. See
> [19](19_earthscope_access.md).
>
> **Keep reading only for the method.** Steps 4–6 are still the right way to
> take an S3 access problem apart, and §4a — that `aws s3api` silently truncates
> keys at `#`, turning a readable object into a 404 — cost a day on its own and
> will do so again.
>
> **The lesson worth carrying:** listing succeeded throughout, which read as
> proof the role was fine and pointed every diagnosis at entitlement. LIST and
> GET are separate grants. A successful listing says nothing about read access.

A step-by-step to demonstrate the failure in item **0g** with nothing but the
AWS CLI, in-region. No pipeline, no container, no Python.

**What you are demonstrating:** `ListObjectsV2` on the restricted access point
succeeds and `GetObject` returns `AccessDenied` — instantly, not slowly.

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

## 0. Try CloudShell first — it may save the whole exercise

AWS CloudShell gives an in-region shell with the CLI already installed and
nothing to launch or tear down. Open the console, **set the region to
us-east-2**, and skip to step 3.

Whether it satisfies EarthScope's region check depends on how that check is
written — a condition on the AWS region will pass, one scoped to EC2 source-IP
ranges may not. Thirty seconds to find out, and if it fails the EC2 route below
still works.

## 1. Launch a scratch instance in us-east-2

Region matters; nothing else does. `t3.micro` is enough — this is a request-shape
test, not a throughput test.

**An instance profile is required, not optional.** Without one the SSM agent has
no credentials to register with and `start-session` fails with
`TargetNotConnected`, which looks like a networking problem and is not. Create it
once:

```bash
aws iam create-role --role-name EC2SSMProbe \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name EC2SSMProbe \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam create-instance-profile --instance-profile-name EC2SSMProbe
aws iam add-role-to-instance-profile \
  --instance-profile-name EC2SSMProbe --role-name EC2SSMProbe
```

`EC2SSMProbe` already exists in this account as of 2026-09-02; check before
creating it again.

Then launch **with the profile attached**:

```bash
aws ec2 run-instances --region us-east-2 \
  --image-id resolve:ssm:/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
  --instance-type t3.micro \
  --iam-instance-profile Name=EC2SSMProbe \
  --metadata-options "HttpTokens=required" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=es-probe}]' \
  --count 1 --query 'Instances[0].InstanceId' --output text
```

Wait for the agent to register — a minute or two after the instance is running:

```bash
aws ssm describe-instance-information --region us-east-2 \
  --filters "Key=InstanceIds,Values=i-XXXXXXXXXXXX" \
  --query 'InstanceInformationList[0].PingStatus' --output text     # want: Online
aws ssm start-session --region us-east-2 --target i-XXXXXXXXXXXX
```

**If `TargetNotConnected` persists**, check these in order — the first two are
the usual causes and neither is obvious from the error:

1. `describe-instances … IamInstanceProfile` returns `null` → the profile did not
   attach. `associate-iam-instance-profile` fixes it on a *running* instance, but
   the agent may need `reboot-instances` to re-read credentials.
2. `describe-instances … State.Name` is `terminated` or `shutting-down` → you are
   waiting on an instance that no longer exists.
3. The subnet has no route to the internet and no SSM VPC endpoints.

Placing it in the campaign VPC (`vpc-0543376e`, public subnets behind
`igw-b6f0e6de`) makes the test a closer match to what the Fargate workers see.
A default VPC also works and is simpler.

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
# `es` lives in earthscope-CLI. The SDK ships no console script at all.
pip3 install --user earthscope-cli
export PATH="$HOME/.local/bin:$PATH"

es login          # opens a device-code flow; follow the URL it prints
es user get-aws-credentials s3-miniseed-v2
```

That prints temporary keys. Export them:

```bash
eval "$(es user get-aws-credentials s3-miniseed-v2 --format env)"
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
  --prefix miniseed/AV/2019/187/ \
  --max-items 10 \
  --query 'Contents[].{Key:Key,Size:Size}' --output table
```

Expect a table of objects in under a second. **Record the `Key` and `Size` of
one of them** — the size matters for step 6.

## 4a. The AWS CLI cannot address these keys

**Every restricted object carries a `#N` version suffix** — there are no keys
without one:

```
miniseed/AV/2019/187/ACH.AV.2019.187#1
miniseed/AV/2019/187/ADAG.AV.2019.187#3
miniseed/AV/2019/187/ADKI.AV.2019.187#2
```

`aws s3api` truncates the key at `#`, asks for `ACH.AV.2019.187`, and gets
**404 / NoSuchKey**. That is the CLI mangling the request, not EarthScope's
answer — and it points to the opposite conclusion: "the data is missing" rather
than "the data is unreadable".

**404 and 403 mean different things here.** With `s3:ListBucket`, S3 returns 404
for a key that does not exist and 403 for one you may not read. Only the 403
supports an entitlement claim. A ticket written from the CLI output would have
been wrong, and EarthScope would rightly have rejected it.

As it turned out, a ticket written from the *correct* 403 would also have been
wrong — the request was unscoped. See the banner at the top.

Use **boto3**, which encodes the key correctly:

```bash
python3 - <<'PY'
import boto3, botocore
AP  = "earthscope-mseed-v2-4fdodyzpsz8u8uyi3pa9qsw9oid1suse2a-s3alias"
KEY = "miniseed/AV/2019/187/ACH.AV.2019.187#1"
s3  = boto3.client("s3", region_name="us-east-2")
print("caller:", boto3.client("sts").get_caller_identity()["Arn"])
for label, fn in (
    ("HEAD",     lambda: s3.head_object(Bucket=AP, Key=KEY)),
    ("GET 1KB",  lambda: s3.get_object(Bucket=AP, Key=KEY, Range="bytes=0-1023")),
    ("GET full", lambda: s3.get_object(Bucket=AP, Key=KEY)),
):
    try:
        r = fn(); print(label, "OK", r.get("ContentLength"), "bytes")
    except botocore.exceptions.ClientError as e:
        print(label, e.response["Error"]["Code"])
PY
```

**Confirmed 2026-09-02**, EC2 in us-east-2, fresh interactive login:

```
caller:   arn:aws:sts::457219964709:assumed-role/earthscope-idm-mseed-v2/euid=...
HEAD      403: Forbidden
GET 1KB   AccessDenied
GET full  AccessDenied
```

The container reported the same denial using the campaign's own refresh token.
**Two independent credentials, same result** — this is not a token problem, and
LIST succeeds in both.

## 5. Show that HEAD works

```bash
KEY=miniseed/AV/2019/187/ACH.AV.2018.135      # substitute a real key from step 4

time aws s3api head-object --bucket "$AP" --key "$KEY"
```

Expect `ContentLength` and `LastModified` back promptly. If this hangs, the
problem is earlier than GET and that is itself the finding.

## 6. Show that GET is denied

This is the failure. Use an explicit short timeout so the shell returns rather
than sitting there:

```bash
time aws s3api get-object \
  --bucket "$AP" --key "$KEY" \
  --cli-read-timeout 60 --cli-connect-timeout 15 \
  /tmp/out.mseed
```

**Expected:** `An error occurred (AccessDenied) when calling the GetObject
operation`, returned immediately. Measured from Fargate in us-east-2 with the
production credential, the same three steps take 1.4 s / 0.0 s / 0.0 s — LIST
returns 163 objects, HEAD returns 403, GET is denied.

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
| LIST ok, HEAD 403, GET AccessDenied | **the observed case** — an UNSCOPED credential. Add `network=FDSN:<NET>` (and `year=` for temporary networks) before suspecting entitlement |
| LIST ok, HEAD 404, GET NoSuchKey | the key was mangled — almost certainly `#` via `aws s3api`. Retry with boto3 |
| LIST ok, HEAD ok, GET AccessDenied | narrower than the observed case: read denied but metadata allowed |
| everything hangs including Open Data | the instance or its network, not EarthScope |
| all of it works from EC2 | the failure is specific to Fargate's networking, not the region — look at the public-subnet/IGW path and the absence of an S3 gateway endpoint |

That last row is worth taking seriously. The compute environment runs tasks in
**public subnets behind an Internet Gateway with no NAT and no S3 gateway
endpoint** ([OPTIMISE.md](OPTIMISE.md) item 4a). A plain EC2 instance in a
default VPC does not reproduce that path, so a clean EC2 result would point at
the Fargate networking rather than exonerate EarthScope.
