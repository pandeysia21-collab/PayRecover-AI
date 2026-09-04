# PayRecover AI 💳

**AI-Powered Payment Failure & Revenue Recovery Agent**

PayRecover AI predicts the probability that a failed payment can be recovered, chooses the next-best recovery action, applies explicit safety guardrails, executes a sandbox/offline action, observes the available outcome, and records an audit trail.

## Problem
Payment failures create revenue at risk. Blindly retrying every failed payment can waste attempts and create a poor customer experience. The system should identify promising recovery opportunities while knowing when to stop.

## Solution
1. **ML scoring** — Gradient Boosting predicts recovery probability.
2. **Agent policy** — probability + retry history determine the next action.
3. **Policy gate** — hard stopping rules prevent unsafe repeated retries.
4. **Sandbox executor** — demonstrates bounded action execution without touching real money.
5. **Outcome observation** — offline/synthetic recovery outcome is displayed when available.
6. **Audit trail** — records decision, guardrail, execution status, and outcome.

## Agent workflow
`Detect → Score → Decide → Gate → Execute → Observe → Audit`

| Condition | Agent decision | Execution |
|---|---|---|
| Retry count ≥ 3 | Stop repeated retries | BLOCKED |
| Probability < 35% | Avoid repeated retry | BLOCKED |
| Probability ≥ 70% | Retry payment immediately | SANDBOX RETRY |
| 35%–<70% | Suggest alternative payment method | SANDBOX ALTERNATIVE |

## Safety / guardrails
- Maximum retry limit: **3**
- Automatic retry is blocked below **35%** probability
- Medium-probability cases are routed to an alternative payment method
- Every decision is auditable
- Execution is explicitly sandbox/offline
- No live customer payment is claimed by this submission

## ML methodology
The model uses transaction information, payment method, failure reason, customer history, retry activity, time since failure, average transaction amount, subscription status, and engineered behavioural features including success rate, failure rate, amount deviation, total payment history, and retry pressure.

Current dashboard evaluation: **68.67% accuracy** and **73.48% ROC-AUC**.

## Offline batch evaluation
The project includes a synthetic/offline batch evaluation with revenue at risk, eligible recovery opportunities, expected recovery, and observed recovery.

**Important:** observed recovery is an offline/synthetic evaluation, not live money recovered from real customers. It should not be interpreted as causal lift without a controlled baseline.

## Run locally
```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Keep `app.py`, `payrecover_model.pkl`, `batch_recovery_results.csv`, and `prediction_history.csv` in the same directory.

## Demo flow
1. Open **Dashboard**.
2. Open **🤖 Recovery Agent**.
3. Select a failed transaction.
4. Show probability, retry count, policy gate, and bounded action.
5. Click **Run Bounded Recovery Agent**.
6. Show the execution trace and audit record.
7. Demonstrate a stopped case where the retry limit blocks execution.

## Production boundary
This submission demonstrates the complete recovery-control loop using synthetic/offline data and a sandbox executor. A production payment adapter would additionally require payment-platform test mode, authorization, idempotency, webhook verification, rate limits, monitoring, and a hard stop/rollback path.
