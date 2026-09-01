# CloudWatch Budget Alert Setup (5 min)

**Goal:** Email alert when daily spend reaches $250  
**Why:** Catch cost overruns early, not on the final bill

---

## Step-by-Step Instructions

### 1. Open AWS Billing Console

Navigate to: **https://console.aws.amazon.com/billing/**

Sign in if prompted. Make sure you're in the **root account** (not a sub-account).

---

### 2. Go to Budgets

In the left sidebar, click **Budgets** (under "Cost Management")

If you don't see it, use the search bar at the top:
- Type: `Budgets`
- Click the result

---

### 3. Click "Create budget"

Button in the top-right of the Budgets page.

---

### 4. Choose Budget Type

**Budget type:** Select **Cost budget** (should be default)

Click **Next**

---

### 5. Set Budget Details

Fill in:

| Field | Value |
|-------|-------|
| **Budget name** | `QuakeScope-2026-daily` |
| **Budgeting method** | Simple budgeting |
| **Period** | Daily |
| **Start date** | 2026-09-01 (or today) |
| **Budgeted amount** | 250.00 |

Click **Next**

---

### 6. Add Alert Threshold

Button: **Add alert threshold**

Fill in:

| Field | Value |
|-------|-------|
| **Alert type** | Actual |
| **Alert threshold** | 100 (percent) |
| **Email recipients** | mdenolle@uw.edu |

Click **Save** (small button on the right side of the alert row)

---

### 7. Review & Confirm

Review all settings:
- Budget name: `QuakeScope-2026-daily`
- Period: Daily
- Amount: $250.00
- Alert: At 100%, email to mdenolle@uw.edu

Click **Create budget**

---

### 8. Verify

You should see:
- Success message at the top
- Budget appears in the list below

Check your email (mdenolle@uw.edu) for confirmation from AWS.

---

## What You'll Get

Once the budget is created:

✅ **Daily email** if you spend >$250 in a calendar day  
✅ **Dashboard view** showing year-to-date spending (in Budgets console)  
✅ **Early warning** before costs spiral

---

## Optional: Add a Second Alert (Earlier Warning)

If you want to catch overruns even sooner, add another alert at 80%:

1. Go back to Budgets → click your budget
2. Click **Edit** (top-right)
3. Under "Budget alerts", click **Add alert threshold** again
4. Set to **80** percent (same email)
5. Click **Update budget**

This way you get warned at $200/day and again at $250/day.

---

## Done!

You now have:
- ✅ Daily budget: $250
- ✅ Email alert at 100%
- ✅ Dashboard for monitoring

**Next:** Commit these docs and proceed to result collection when Phase 1 jobs complete (~30 min).
