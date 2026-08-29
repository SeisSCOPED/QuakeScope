# 15 — Knowing what is running, and stopping it fast

Cost Explorer lags about a day. That is too slow to catch the three ways a Spot
campaign quietly overspends:

1. an instance left running after a test,
2. **EC2 instances at all** - campaigns run on Fargate, so any EC2 is unexpected,
3. a managed job silently satisfied **on-demand** at roughly 3× the Spot price —
   which happened during the v3 smoke test and was visible only as
   `InstanceLifecycle` in the EC2 console.

Two independent layers cover this: an hourly report you read, and an AWS-native
budget alarm that fires even if the first one breaks.

## Layer 1 — hourly report (GitHub Actions)

`.github/workflows/aws-watch.yml` runs `scripts/aws_watch.py` every hour across
us-west-2, us-east-1 and us-east-2 and reports:

- every running instance, its type, **spot vs on-demand**, and age
- hourly burn at the *current* spot price for that type and AZ
- the projection to $/day and $/month if nothing changes
- open spot requests, which can relaunch instances you just terminated
- ready-to-paste emergency-stop commands, in the right order

It is read-only and changes nothing in the account.

**Notification model.** One issue, labelled `aws-watch`, is the live dashboard.
Its body is rewritten hourly, which is silent. A *comment* — which emails
everyone subscribed to the repo — is posted only when the state **changes**:
idle → running, a new warning appears, or everything stops. An hourly job that
notified every hour would train you to ignore it.

That gives you email out of the box, since GitHub emails issue activity. For
push notifications, the GitHub mobile app covers the same events; for WhatsApp
or Slack, add a step to the workflow posting the same `report.md` to a webhook.

### Setup

Preferred, no stored credentials — GitHub OIDC:

1. Create an IAM role trusting `token.actions.githubusercontent.com`, scoped to
   this repo, with the read-only policy below.
2. Set repository **variable** `AWS_WATCH_ROLE_ARN` to its ARN.

Fallback, if OIDC is not set up: create a read-only IAM user and set repository
**secrets** `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. The workflow picks
whichever is configured.

Minimum policy — describe-only, no ability to launch or terminate anything:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "ec2:DescribeInstances",
      "ec2:DescribeSpotInstanceRequests",
      "ec2:DescribeSpotPriceHistory",
      "pricing:GetProducts"
    ],
    "Resource": "*"
  }]
}
```

Deliberately read-only: a monitor that can stop things is a monitor that can
stop the wrong thing. Stopping stays a human action.

### Turning it on

Three things must all be true, and none of them is automatic:

1. **`aws-watch.yml` must be on the default branch.** GitHub runs `schedule:`
   triggers *only* from the default branch. On a feature branch the file is
   inert — it will not fire once, ever. (Push-triggered workflows are different
   and do run from any branch, which is why `pages.yml` worked from
   `rerun-2026-prep`.)
2. **Credentials must be configured** — repository variable
   `AWS_WATCH_ROLE_ARN`, or the two access-key secrets. Without either, every
   run fails on the first AWS call.
3. **You must be watching the repo**, so issue comments become email. Check at
   *Watch → All Activity*, or:

```bash
gh api repos/SeisSCOPED/QuakeScope/subscription -q .subscribed
```

Until one of these is set the workflow still runs on schedule, but **skips
cleanly** rather than failing: it writes what to configure into the run summary
and exits successfully. That is deliberate — an hourly job that fails hourly
emails you hourly, which trains you to filter exactly the notifications this
exists to deliver, and a real AWS problem then looks identical to an unconfigured
one.

Verify it works without waiting an hour — run it by hand:

```bash
gh workflow run aws-watch.yml
gh run list --workflow aws-watch.yml --limit 3
```

### Turning it off

Pick the level that matches what you actually want to stop.

**Stop the checks entirely** (no runs, no issue updates, no email):

```bash
gh workflow disable aws-watch.yml     # re-enable with: gh workflow enable
```

Or in the browser: **Actions → AWS activity watch → ⋯ → Disable workflow**.

**Keep the dashboard, stop the email.** The issue body still updates hourly so
you can look when you want, but comments stop reaching your inbox — unsubscribe
from that one issue (*Unsubscribe* in the right-hand sidebar), or:

```bash
gh api -X PUT repos/SeisSCOPED/QuakeScope/issues/<n>/subscription -f ignored=true
```

**Stop all repo email but keep everything running:** set the repo to
*Participating and @mentions* instead of *All Activity*. You will then only be
emailed if the bot @-mentions you, which it does not — so this effectively
silences it while leaving the dashboard live.

**Pause without disabling:** delete the `schedule:` block and keep
`workflow_dispatch:`. The workflow then only runs when you ask it to.

Two things worth knowing:

- **GitHub auto-disables scheduled workflows after 60 days without repo
  activity**, and emails the repo admins when it does. If the alerts go quiet
  during a long gap, check whether this happened rather than assuming all is
  well.
- Disabling the workflow disables the *watching*, not the *spending*. Nothing
  about turning alerts off stops an instance.

### Running it yourself

```bash
pixi run -e cloud watch            # what is running right now
pixi run -e cloud watch-md         # markdown, same as the issue
python scripts/aws_watch.py --quiet # silent when idle, for a local cron
```

## Layer 2 — AWS Budgets (the backstop)

The workflow depends on GitHub Actions, credentials, and the repo. Budgets does
not. Set one up so a runaway campaign is caught even if layer 1 is broken:

**Billing → Budgets → Create budget → Cost budget**, monthly, with alerts at
50/80/100% of a figure you would be unhappy to exceed, emailing you. Add a
second **daily** budget at a low threshold — a daily alarm catches a runaway in
hours rather than at month end, which is the whole point.

Budgets alerts on spend, not on instances, so it lags a few hours — hence both
layers.

## Emergency stop

Terminate, do not stop: a stopped instance still bills its EBS volume and
still counts as a leak.

```bash
# what is running, with the exact stop commands for whatever it finds
pixi run -e cloud watch
```

From the console:

1. Set the region — nothing stops in the wrong region, and a campaign may span
   several.
2. **EC2 → Instances**, clear the default *Running* filter so terminated
   instances are visible too, select, *Instance state → Terminate*.
3. **EC2 → Spot Requests** → cancel any `open`/`active`. Cancelling a request
   does **not** terminate its instance, and a persistent request will relaunch
   one, so do both.
4. Verify **Instances** filtered to *Running* is empty.

> Whatever orchestrator you use, check afterwards rather than trusting the
> teardown command. `aws_watch.py` queries EC2 directly, which is the only
> account of what is actually running.

## Known standing resources

The first run of this watcher found, in us-east-2:

- `niyiyu-quakescope-web-service` (`t2.large`, on-demand) running **11,210 hours**
  — about 467 days, ~$67/month, on the order of $1,000 to date. Not part of any
  campaign. Left alone because it may be a live service, but it is exactly the
  kind of thing this exists to surface.
- Batch compute environments `niyiyu_earthscope` and `pickblue_OBS`, still
  `ENABLED`. Idle and therefore free, but they will accept jobs.
