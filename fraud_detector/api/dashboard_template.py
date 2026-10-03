"""
fraud_detector/api/dashboard_template.py
────────────────────────────────────────
Single-file self-contained operations & review dashboard.

Offline guarantee:
  • ZERO external network requests (no CDNs, no Google fonts, no remote assets).
  • All CSS, SVG icons, and vanilla JavaScript are bundled inline.
  • XSS sanitized: all dynamic inputs rendered via document.createTextNode.
"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fraud Detection Operations Console</title>
<style>
  :root {
    --bg-dark: #0f172a;
    --card-bg: #1e293b;
    --border-color: #334155;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --primary: #3b82f6;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --critical: #8b5cf6;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background-color: var(--bg-dark);
    color: var(--text-main);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    padding: 24px;
    line-height: 1.5;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 16px;
    margin-bottom: 24px;
  }
  h1 { font-size: 20px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
  .status-badge {
    background: rgba(16, 185, 129, 0.15);
    color: var(--success);
    border: 1px solid var(--success);
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 500;
  }
  .grid-metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }
  .kpi-card {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 16px;
  }
  .kpi-title { font-size: 13px; color: var(--text-muted); margin-bottom: 4px; }
  .kpi-value { font-size: 24px; font-weight: 700; color: var(--text-main); }
  .layout-grid {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 24px;
    margin-bottom: 24px;
  }
  @media (max-width: 900px) { .layout-grid { grid-template-columns: 1fr; } }
  .panel {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 24px;
  }
  .panel-title { font-size: 16px; font-weight: 600; margin-bottom: 16px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 10px; text-align: left; border-bottom: 1px solid var(--border-color); }
  th { color: var(--text-muted); font-weight: 500; }
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
  }
  .badge-approve { background: rgba(16, 185, 129, 0.2); color: var(--success); }
  .badge-step-up { background: rgba(245, 158, 11, 0.2); color: var(--warning); }
  .badge-review { background: rgba(59, 130, 246, 0.2); color: var(--primary); }
  .badge-block { background: rgba(239, 68, 68, 0.2); color: var(--danger); }
  .btn {
    background: var(--primary);
    color: white;
    border: none;
    padding: 8px 14px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
  }
  .btn:hover { opacity: 0.9; }
  .btn-sm { padding: 4px 8px; font-size: 11px; margin-right: 4px; }
  .btn-fraud { background: var(--danger); }
  .btn-legit { background: var(--success); }
  .form-group { margin-bottom: 14px; }
  label { display: block; font-size: 12px; color: var(--text-muted); margin-bottom: 4px; }
  input, select {
    width: 100%;
    padding: 8px 12px;
    background: #0f172a;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: var(--text-main);
    font-size: 13px;
  }
  .cm-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    text-align: center;
  }
  .cm-box {
    background: #0f172a;
    padding: 12px;
    border-radius: 6px;
    border: 1px solid var(--border-color);
  }
  .cm-label { font-size: 11px; color: var(--text-muted); }
  .cm-count { font-size: 20px; font-weight: 700; margin-top: 4px; }
  #score-result { margin-top: 14px; padding: 10px; border-radius: 6px; font-size: 12px; display: none; }
</style>
</head>
<body>

<header>
  <h1>Fraud Detection Operations & Decision Console</h1>
  <div class="status-badge" id="backend-status">Connected (Local UTC)</div>
</header>

<div class="grid-metrics">
  <div class="kpi-card">
    <div class="kpi-title">Total Scored</div>
    <div class="kpi-value" id="kpi-total">0</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Review Queue</div>
    <div class="kpi-value" id="kpi-queue">0</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Precision</div>
    <div class="kpi-value" id="kpi-precision">0.0%</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Recall</div>
    <div class="kpi-value" id="kpi-recall">0.0%</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">False Positive Rate</div>
    <div class="kpi-value" id="kpi-fpr">0.0%</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">p95 Latency</div>
    <div class="kpi-value" id="kpi-latency">0.0 ms</div>
  </div>
</div>

<div class="layout-grid">
  <div>
    <!-- Review Queue -->
    <div class="panel">
      <div class="panel-title">Analyst Review Queue (Awaiting Action)</div>
      <table id="review-table">
        <thead>
          <tr>
            <th>Tx ID</th>
            <th>User</th>
            <th>Amount</th>
            <th>Score</th>
            <th>Flags</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody id="review-body">
          <tr><td colspan="6" style="text-align: center; color: var(--text-muted);">Review queue is empty.</td></tr>
        </tbody>
      </table>
    </div>

    <!-- Live Transactions -->
    <div class="panel">
      <div class="panel-title">Live Scored Transactions</div>
      <table id="tx-table">
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Tx ID</th>
            <th>User</th>
            <th>Amount</th>
            <th>Score</th>
            <th>Decision</th>
            <th>Reasons</th>
          </tr>
        </thead>
        <tbody id="tx-body">
          <tr><td colspan="7" style="text-align: center; color: var(--text-muted);">No transactions scored yet.</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <div>
    <!-- Confusion Matrix -->
    <div class="panel">
      <div class="panel-title">Confusion Matrix</div>
      <div class="cm-grid">
        <div class="cm-box">
          <div class="cm-label">True Positives (TP)</div>
          <div class="cm-count" id="cm-tp" style="color: var(--success);">0</div>
        </div>
        <div class="cm-box">
          <div class="cm-label">False Positives (FP)</div>
          <div class="cm-count" id="cm-fp" style="color: var(--danger);">0</div>
        </div>
        <div class="cm-box">
          <div class="cm-label">False Negatives (FN)</div>
          <div class="cm-count" id="cm-fn" style="color: var(--danger);">0</div>
        </div>
        <div class="cm-box">
          <div class="cm-label">True Negatives (TN)</div>
          <div class="cm-count" id="cm-tn" style="color: var(--success);">0</div>
        </div>
      </div>
    </div>

    <!-- Manual Scoring Form -->
    <div class="panel">
      <div class="panel-title">Submit Transaction For Scoring</div>
      <form id="score-form">
        <div class="form-group">
          <label for="f-tx-id">Transaction ID</label>
          <input type="text" id="f-tx-id" placeholder="txn-manual-001">
        </div>
        <div class="form-group">
          <label for="f-user-id">User ID</label>
          <input type="text" id="f-user-id" placeholder="usr-manual-001">
        </div>
        <div class="form-group">
          <label for="f-merchant-id">Merchant ID</label>
          <input type="text" id="f-merchant-id" placeholder="mer-manual-001">
        </div>
        <div class="form-group">
          <label for="f-amount">Amount (in paise, integer)</label>
          <input type="number" id="f-amount" placeholder="75000">
        </div>
        <div class="form-group">
          <label for="f-channel">Channel</label>
          <select id="f-channel">
            <option value="ONLINE">ONLINE</option>
            <option value="UPI">UPI</option>
            <option value="POS">POS</option>
            <option value="APP">APP</option>
          </select>
        </div>
        <button type="submit" class="btn" id="btn-score-submit" style="width: 100%;">Evaluate Transaction</button>
      </form>
      <div id="score-result"></div>
    </div>
  </div>
</div>

<script>
  function formatPaise(p) {
    if (isNaN(p)) return '₹0.00';
    return '₹' + (p / 100).toFixed(2);
  }

  function getDecisionBadge(decision) {
    var cls = 'badge-approve';
    if (decision === 'STEP_UP') cls = 'badge-step-up';
    else if (decision === 'REVIEW') cls = 'badge-review';
    else if (decision === 'BLOCK') cls = 'badge-block';
    var span = document.createElement('span');
    span.className = 'badge ' + cls;
    span.textContent = decision || 'APPROVE';
    return span;
  }

  async function loadMetrics() {
    try {
      var res = await fetch('/v1/metrics');
      if (!res.ok) return;
      var data = await res.json();
      document.getElementById('kpi-total').textContent = data.total_scored;
      document.getElementById('kpi-queue').textContent = data.review_queue_length;
      document.getElementById('kpi-precision').textContent = (data.precision * 100).toFixed(1) + '%';
      document.getElementById('kpi-recall').textContent = (data.recall * 100).toFixed(1) + '%';
      document.getElementById('kpi-fpr').textContent = (data.false_positive_rate * 100).toFixed(1) + '%';
      document.getElementById('kpi-latency').textContent = data.p95_latency_ms.toFixed(1) + ' ms';

      var cm = data.confusion_matrix || {};
      document.getElementById('cm-tp').textContent = cm.tp || 0;
      document.getElementById('cm-fp').textContent = cm.fp || 0;
      document.getElementById('cm-fn').textContent = cm.fn || 0;
      document.getElementById('cm-tn').textContent = cm.tn || 0;

      // Update Review Queue
      var rBody = document.getElementById('review-body');
      rBody.innerHTML = '';
      if (!data.review_queue || data.review_queue.length === 0) {
        var tr = document.createElement('tr');
        var td = document.createElement('td');
        td.colSpan = 6;
        td.style.textAlign = 'center';
        td.style.color = '#94a3b8';
        td.textContent = 'Review queue is empty.';
        tr.appendChild(td);
        rBody.appendChild(tr);
      } else {
        data.review_queue.forEach(function(item) {
          var tr = document.createElement('tr');

          var tdId = document.createElement('td');
          tdId.textContent = item.transaction_id;
          var tdUser = document.createElement('td');
          tdUser.textContent = item.user_id;
          var tdAmt = document.createElement('td');
          tdAmt.textContent = formatPaise(item.amount);
          var tdScore = document.createElement('td');
          tdScore.textContent = item.score.toFixed(3);
          var tdFlags = document.createElement('td');
          tdFlags.textContent = (item.flags || []).join(', ') || 'N/A';

          var tdAct = document.createElement('td');
          var bFraud = document.createElement('button');
          bFraud.className = 'btn btn-sm btn-fraud';
          bFraud.textContent = 'Mark Fraud';
          bFraud.onclick = function() { submitReview(item.transaction_id, 'FRAUD'); };

          var bLegit = document.createElement('button');
          bLegit.className = 'btn btn-sm btn-legit';
          bLegit.textContent = 'Mark Legit';
          bLegit.onclick = function() { submitReview(item.transaction_id, 'LEGIT'); };

          tdAct.appendChild(bFraud);
          tdAct.appendChild(bLegit);

          tr.appendChild(tdId);
          tr.appendChild(tdUser);
          tr.appendChild(tdAmt);
          tr.appendChild(tdScore);
          tr.appendChild(tdFlags);
          tr.appendChild(tdAct);
          rBody.appendChild(tr);
        });
      }

      // Update Scored Transactions
      var txBody = document.getElementById('tx-body');
      txBody.innerHTML = '';
      if (!data.recent_transactions || data.recent_transactions.length === 0) {
        var tr = document.createElement('tr');
        var td = document.createElement('td');
        td.colSpan = 7;
        td.style.textAlign = 'center';
        td.style.color = '#94a3b8';
        td.textContent = 'No transactions scored yet.';
        tr.appendChild(td);
        txBody.appendChild(tr);
      } else {
        data.recent_transactions.forEach(function(tx) {
          var tr = document.createElement('tr');

          var tdTime = document.createElement('td');
          tdTime.textContent = (tx.scored_at || '').substring(11, 19);
          var tdId = document.createElement('td');
          tdId.textContent = tx.transaction_id;
          var tdUser = document.createElement('td');
          tdUser.textContent = tx.user_id;
          var tdAmt = document.createElement('td');
          tdAmt.textContent = formatPaise(tx.amount);
          var tdScore = document.createElement('td');
          tdScore.textContent = (tx.score !== undefined) ? tx.score.toFixed(3) : '0.000';

          var tdDec = document.createElement('td');
          tdDec.appendChild(getDecisionBadge(tx.decision));

          var tdFlags = document.createElement('td');
          tdFlags.textContent = (tx.flags || []).slice(0, 2).join('; ') || 'None';

          tr.appendChild(tdTime);
          tr.appendChild(tdId);
          tr.appendChild(tdUser);
          tr.appendChild(tdAmt);
          tr.appendChild(tdScore);
          tr.appendChild(tdDec);
          tr.appendChild(tdFlags);
          txBody.appendChild(tr);
        });
      }
    } catch (e) {
      console.error(e);
    }
  }

  async function submitReview(txId, label) {
    try {
      var res = await fetch('/v1/review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transaction_id: txId, label: label })
      });
      if (res.ok) {
        loadMetrics();
      }
    } catch (e) {
      console.error(e);
    }
  }

  document.getElementById('score-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    var resBox = document.getElementById('score-result');
    resBox.style.display = 'block';

    var txId = document.getElementById('f-tx-id').value.trim();
    var userId = document.getElementById('f-user-id').value.trim();
    var merchantId = document.getElementById('f-merchant-id').value.trim();
    var amountStr = document.getElementById('f-amount').value.trim();
    var channel = document.getElementById('f-channel').value;

    if (!txId || !userId || !merchantId || !amountStr) {
      resBox.style.background = 'rgba(239, 68, 68, 0.2)';
      resBox.style.color = '#ef4444';
      resBox.textContent = 'Validation error: Please fill in all required fields.';
      return;
    }

    var amount = parseInt(amountStr, 10);
    if (isNaN(amount) || amount <= 0) {
      resBox.style.background = 'rgba(239, 68, 68, 0.2)';
      resBox.style.color = '#ef4444';
      resBox.textContent = 'Validation error: Amount must be a positive integer (paise).';
      return;
    }

    var payload = {
      transaction_id: txId,
      user_id: userId,
      merchant_id: merchantId,
      amount: amount,
      currency: 'INR',
      timestamp: new Date().toISOString(),
      channel: channel,
      transaction_type: 'PURCHASE'
    };

    try {
      var res = await fetch('/v1/score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      var data = await res.json();
      if (res.ok) {
        resBox.style.background = 'rgba(16, 185, 129, 0.2)';
        resBox.style.color = '#10b981';
        resBox.textContent = 'Success! Score: ' + data.score.toFixed(3) + ' | Decision: ' + (data.decision || data.risk_level);
        loadMetrics();
      } else {
        resBox.style.background = 'rgba(239, 68, 68, 0.2)';
        resBox.style.color = '#ef4444';
        resBox.textContent = 'Error: ' + JSON.stringify(data.detail || data.error);
      }
    } catch (err) {
      resBox.style.background = 'rgba(239, 68, 68, 0.2)';
      resBox.style.color = '#ef4444';
      resBox.textContent = 'Network or server error: ' + err.message;
    }
  });

  // Initial load
  loadMetrics();
  setInterval(loadMetrics, 3000);
</script>

</body>
</html>
"""
