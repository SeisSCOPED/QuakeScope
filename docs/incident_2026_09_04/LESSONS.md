# Lessons — what went wrong, and what to do differently

Written for whoever picks this up next: a future agent working in this
repository, and the [UW Scalable Data Systems
group](https://sds.cs.washington.edu/people/), with whom we expect to co-design
work that hits the same shapes. It is a record of eight defects and what each
one teaches, not a narrative of the week.

Context in one paragraph: QuakeScope runs a few hundred AWS Batch workers that
read seismic waveforms from an archive whose credentials are issued per
network and per year. On 2026-09-04 the operator of that archive told us our
fleet was denial-of-servicing their credentials endpoint. It was: 351,735
token requests in four hours, every one rejected. Five defects in one file
caused it. Fixing those introduced two more, and one of my own fixes nearly
introduced a third.

---

## The one that matters most

**A 4xx is a verdict on the request. Retrying it is the whole bug.**

Every other defect is a variation on failing to distinguish *this request was
wrong* from *the service is busy*. 400, 403, 404 mean the answer will not
change; 429 and 5xx mean it might. A client that treats them alike converts one
refusal into unbounded load, and the multiplier is whatever loop sits above it —
here, one credential lookup per station-day per object, 11,315 per shard.

Concretely: cache refusals as hard as successes, keyed by the exact scope that
was refused, and never send that scope again for the life of the process.

---

## Defects, and the general shape of each

### 1. A refusal was never remembered
Successes were cached by scope; refusals were not. Every one of 11,315 calls
per shard re-ran the exchange.

**Shape:** asymmetric caching. If you cache the happy path, ask what happens to
the unhappy one. Negative results are usually cheaper to cache and more
valuable, because failures repeat harder than successes.

### 2. Escalation removed scope instead of adding it
On any denial the client flipped a network to "the other" scoping. For a
temporary FDSN code that meant dropping the year — producing a request that can
only ever be answered 400, which is exactly what the archive operator noticed.

**Shape:** a retry that changes the request must only ever make it *more*
specific. Broadening on failure is how you turn one bad request into a class of
bad requests. Also: it was reachable from a 403, the one status where a
differently-shaped request provably cannot help.

### 3. A new SDK client per call
The SDK caches issued credentials on the client instance. Building a fresh one
per call threw that away and forced a round trip plus an OAuth refresh.

**Shape:** know what your SDK caches and where. "Construct it fresh each time"
is the safe-looking default that quietly disables the library's own protections.

### 4. Auth failures fell into the generic retry loop
`InvalidRefreshTokenError` has no `.response` attribute, so it slipped past
every `exc.response.status_code` branch and reached a retry loop that re-ran the
refresh grant five times per scope, with a fixed 5-second sleep. Once the
account was blocked, every network took that path on 208 workers at once.

**Shape:** status-code dispatch that assumes an HTTP-shaped exception. Some
errors are about *your credentials*, not about *this request*, and they must
stop everything rather than be retried per-scope. And a fixed sleep synchronises
the fleet — use exponential backoff with full jitter.

### 5. A shadowed exception handler spun a tight loop
`PermissionError` and `FileNotFoundError` both subclass `OSError`, and an
`except OSError` sat before both. Neither could ever run. `s3fs` maps S3
`AccessDenied` to `PermissionError` with no `errno`, so a denied object fell out
of the `OSError` branch without returning, the enclosing `while True` went round
again, and the object was re-`HEAD`ed at network speed until a 900-second
timeout. One shard sat on a single denied object for 447 minutes.

**Shape:** Python's exception hierarchy silently makes later handlers dead code.
Order narrow before broad, and pin it with a test that parses the handler chain
— but see "test the branch, not the shape" below.

### 6. The learned scope mode did not outlive the helper
*Found by EarthScope's own developers, reviewing our fix.* The verdict cache was
moved to process scope so it would survive across shards. The learned
"this network needs a year" flag was not. The cache key is built *from* that
flag, so shard 1 filed a refusal under the year-less key, escalated, and
succeeded — and shard 2 rebuilt the same year-less key, found shard 1's 400, and
gave up on the network for the life of the process.

**Shape:** **state that is read together must share a lifetime.** This is the
most transferable lesson here. If A is used to compute the key for B, then A and
B must be scoped identically. Two caches with different lifetimes and a shared
key space will disagree, and the disagreement is invisible in single-iteration
tests.

Note the severity inversion: the fix for a bandwidth problem created a
data-loss problem. Worth checking, whenever you bound something, whether you
have converted "wasteful" into "wrong".

### 7. The SDK pinned a context-dependent runner
*Found by a dry run, not by any test.* The SDK picks a synchronous runner the
first time anything is syncified and caches it for the life of the context: no
running event loop gets a simple in-thread runner, a running loop gets a
background-thread one. Our first credential exchange came from synchronous
setup code; every later one came from inside an `async def` read loop. So the
cached runner was wrong for every call after the first, and raised a bare
`RuntimeError` before any request was sent — which, being bare, carried no
status, was not terminal, and burned five retries.

**Shape:** same as #6 one level up — caching something context-dependent for a
lifetime longer than the context. Also: a **bare exception type is a
classification failure**. Anything that reaches a retry loop without a status
will be retried, so the loop's default must be "give up loudly", not "try
again".

This cost every shard that crossed a year boundary, and would have cost every
shard outliving the one-hour credential TTL — which is most of a real campaign,
since shards run ~23 hours.

### 8. The fix that would have broken everything
Pinning the runner required importing two SDK-*private* modules. The first
version imported them unguarded. The offline test harness, whose fake SDK has no
such modules, reported 0 requests on every check — because the `ImportError` was
being swallowed by the same generic retry loop from #4.

**Shape:** depending on a library's private API is sometimes correct, but it
must degrade. The guarded version falls back to the ordinary constructor and
logs why: losing multi-year shards is survivable, an `ImportError` on every
exchange in the process is not.

---

## Testing

**Test the branch, not the shape.** D5 had a test asserting handler *order* via
AST. It passed throughout, and told us nothing about whether the branch worked,
because the branch had never run — not in production across five days of logs,
not in either dry run. Only a test that drives the read loop with a filesystem
raising `PermissionError` shows the difference: the incident build was still
looping after 201 HEADs; the fixed build returns after 3.

**An offline harness cannot see context-dependent bugs.** The self-test calls
the read helper synchronously, so the SDK always picks the right runner there.
Defect #7 was invisible to it by construction, and would have stayed invisible
however many checks we added. Some classes of bug require a real run.

**A/B against the broken build.** The most valuable single artifact here is a
harness that runs both the current and the incident build against identical
scripted responses and counts requests that would leave the process. It turns
"we fixed it" into a number, and it keeps working as a regression detector: CI
asserts that the old build still fails all six checks, so if the harness ever
stops seeing the bug, the harness is broken.

**Verify with a parser, not a grep.** Our first published figures came from
substring counts over log lines, which count the request, the warning about it,
and each retry. Year-less requests were reported as 20,024; parsing the actual
request lines gives 3,925, from two codes rather than eight. The *ratio*
survived because numerator and denominator inflated together, which is what
makes this error hard to notice.

---

## Operations

**A promise is not a control.** After the incident we set the fleet targets to
zero and wrote in a public report that they would stay there. Nothing enforced
it: the fleet controller is a scheduled workflow that any of eleven
collaborators could dispatch, and every job definition still pointed at the
build that caused the incident. The control is a quarantine list, checked
against the live job definition before submitting anything, that fails the run
rather than trusting the config file to describe itself.

**Documents assert state; accounts have state.** This project's recurring
failure is a document claiming a deployment that never happened. The audit
report described the incident build in the past tense while all three campaign
job definitions still ran it. Check the account, not the doc — and when you
write the doc, write it so that the act of publishing cannot falsify it.

**Per-scope rate limits do not bound a fleet.** The local limiter allows three
exchanges per scope per five minutes. With ~1,600 planned network-years across
832 processes, that permits an aggregate far above the rate that got us
reported. What actually bounds the fleet is the verdict cache — a correctness
property, not a rate limit. If you need a rate limit, it has to be global.

**Count the processes, not the workers.** Process-wide caches multiply by
`--procs` and by worker count: 208 workers × 4 processes is 832 independent
caches, each of which must learn every verdict once. Spot churn resets them.
"One request per refused scope" is true per process and misleading per fleet.

**You will not detect this yourself unless you instrument for it.** We learned
about the incident from the people we were overloading. Nothing in the stack
alarmed. A metric filter on the outbound-request log line, with an alarm wired
to zero the fleet, is trivial next to the cost of the alternative.

**Least privilege is about your own blast radius, not the incident.** The
campaign role held `AmazonS3FullAccess` throughout. It had nothing to do with
the overload — archive reads use the archive operator's credentials — but it
meant every worker could delete any of the nine buckets in the account,
including other people's, while the code contains two real deletion paths and
the catalogue bucket has no versioning. Scoped now; the method, and the reason
simulation beats reading the policy, is in
[`../aws_least_privilege.md`](../aws_least_privilege.md).

**Retention must outlive the investigation.** The log group keeps five days.
Every number in the incident report rested on data due to be deleted before the
follow-up work was finished. Export first, analyse second.

---

## For a co-designed system

If you are designing the archive side rather than the client side, the defects
above suggest what a client will get wrong, and therefore what the service can
usefully do:

- **Make refusals cheap and unambiguous.** Distinct status codes for "no access"
  and "no such thing" let a client cache correctly. We conflated them for two
  weeks because an unscoped credential lists successfully and only fails at the
  read.
- **Say when a retry could succeed.** `Retry-After` on 429 and 5xx, and nothing
  resembling it on 4xx, tells a well-written client exactly what to do.
- **Publish the scoping rule as data, not prose.** We inferred "codes starting
  with a digit or X/Y/Z need a year" from documentation. A wrong guess in either
  direction costs a request per network, and there is no way for a client to
  discover it except by being refused.
- **Rate-limit the client rather than trusting it.** A 429 with a budget is a
  control the client cannot forget to implement. Our worst hour would have been
  one rejected minute.
- **Expect authentication failures to be indistinguishable from authorisation
  failures** in a client's exception handling, and consider making them
  impossible to confuse in the response.
