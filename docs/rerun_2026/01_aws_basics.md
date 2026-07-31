# 01 — AWS from the ground up: console, credentials, CLI

This guide assumes you have not touched AWS in a while. It gets you from
"I have a UW email" to "my laptop can talk to AWS".

## 1. What the AWS console is

The **AWS Management Console** is the web interface at
<https://console.aws.amazon.com>. Everything we do (Batch, DocumentDB, EC2,
S3, billing) is a "service" inside it. Three permanent orientation points:

- **Search bar (top center)** — type a service name ("Batch", "IAM",
  "DocumentDB", "EC2", "CloudWatch") and hit enter. This is how you navigate;
  ignore the home-page tiles.
- **Region selector (top right, e.g. "Ohio / us-east-2")** — AWS resources
  live in one region. **All QuakeScope infrastructure is in us-east-2
  (Ohio).** If a page looks empty ("no clusters", "no job queues"), the #1
  cause is being in the wrong region.
- **Account menu (top right, your name)** — shows your 12-digit **account
  ID**. Write it down; IAM role ARNs contain it.

## 2. Signing in

Depending on how the account was set up you have one of:

- **IAM user**: sign-in page asks for account ID (or alias), user name,
  password. Use the sign-in URL of the form
  `https://<ACCOUNT_ID>.signin.aws.amazon.com/console`.
- **Root user**: the email address that owns the account. Works, but avoid
  daily use; if it's all you have, sign in with it and create yourself an IAM
  user with `AdministratorAccess` (Console → IAM → Users → Create user).

If you can't sign in at all: "Forgot password" on the sign-in page (root user
uses the account email), or ask whoever administers the account (department /
CloudBank / the students' setup) to reset your IAM user.

## 3. Create fresh CLI credentials (access key)

The console is for looking; scripts talk to AWS with an **access key** (a
key-ID + secret pair). Old keys from the last campaign may be deactivated —
just make a new one:

1. Console → **IAM** → **Users** → click your user name.
2. **Security credentials** tab → **Access keys** → **Create access key**.
3. Use case: **Command Line Interface (CLI)** → confirm → **Create**.
4. Copy both the **Access key ID** and the **Secret access key** now (the
   secret is shown only once). Store them in a password manager.
5. While you're there, deactivate/delete any old keys you no longer use.

## 4. Install and configure the AWS CLI on your laptop

```bash
brew install awscli
```

```bash
aws configure
```

It asks four questions:

| Prompt | Answer |
|---|---|
| AWS Access Key ID | the key you just created |
| AWS Secret Access Key | its secret |
| Default region name | `us-east-2` |
| Default output format | `json` |

Verify it works — this should print your account ID and user ARN:

```bash
aws sts get-caller-identity
```

The same `aws configure` step must also be repeated later **on the EC2
controller instance** (Phase C), which is where jobs are actually submitted
from.

## 5. Set a billing guardrail (do this once, seriously)

1. Console → search **Billing and Cost Management** → **Budgets** →
   **Create budget**.
2. Template: *Monthly cost budget*. Amount: e.g. **$500**. Email: your
   address.
3. You'll get an email at 85% and 100% of the budget — your early warning if
   a campaign is bigger than expected.

Also bookmark **Cost Explorer** (same Billing console) — it shows daily spend
by service; "ECS/Fargate" is the picking jobs, "DocumentDB" is the database,
"EC2" is the controller instance.

## 6. Rough cost expectations

Fargate Spot in us-east-2 is roughly **$0.10–0.12 per hour** for one picking
job (8 vCPU + 16 GB at ~70% spot discount). So:

- 100 jobs × 10 hours ≈ **$100–120**.
- DocumentDB: the always-on cluster is the standing cost — a `db.r6g.large`
  instance is ≈ **$0.28/hr ≈ $200/month**; stop or snapshot+delete it between
  campaigns.
- The EC2 controller (e.g. t3.medium) ≈ $0.04/hr; **stop it when idle**.
- S3 reads from NCEDC/SCEDC/EarthScope public buckets are free to you.

Next: [02_weights_and_container.md](02_weights_and_container.md)
