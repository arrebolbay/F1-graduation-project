/* ============================================================
 * F1 摩纳哥停站策略系统 · 前端逻辑
 * 模块一: 赛前模拟(DP) | 模块二: 赛中 Stackelberg 决策 | 历史记录
 * 图表全部为手绘 SVG,无外部依赖(离线可用)
 * ============================================================ */
"use strict";

let META = null;
let PR_MODE = "normal";
let PR_HAS_RUN = false;
let ST_SCENARIO = "undercut";
let LAST_SIM = null;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function fmtSeconds(s) {
  if (s == null || isNaN(s)) return "—";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s - h * 3600 - m * 60;
  return `${h}:${String(m).padStart(2, "0")}:${sec.toFixed(1).padStart(4, "0")}`;
}

function compColor(c) {
  return { C5: "#e10600", C4: "#ffd24d", C3: "#b8bdc7" }[c] || "#8d95ab";
}

function rateClass(r) {
  return r >= 60 ? "hi" : r >= 30 ? "mid" : "lo";
}

function rateColor(r) {
  return r >= 60 ? "#23d18b" : r >= 30 ? "#ffd24d" : "#ff3b2e";
}

/* ---------------- SVG: 策略时间线 ---------------- */
function svgTimeline(stints, pitLaps) {
  const total = stints.reduce((a, s) => a + s.laps, 0) || 1;
  const W = 920, H = 118, padL = 12, padR = 12;
  const innerW = W - padL - padR;
  let x = padL, blocks = "", ticks = "", pits = "";
  stints.forEach((s) => {
    const w = innerW * s.laps / total;
    blocks += `<rect x="${x.toFixed(1)}" y="34" width="${Math.max(w - 2, 2).toFixed(1)}" height="42" rx="6" fill="${compColor(s.compound)}" opacity="0.92"/>`;
    if (w > 40) {
      blocks += `<text x="${(x + w / 2).toFixed(1)}" y="60" text-anchor="middle" font-size="13" font-weight="800" fill="#10131c">${s.compound}</text>`;
    }
    if (w > 110) {
      blocks += `<text x="${(x + w / 2).toFixed(1)}" y="93" text-anchor="middle" font-size="10.5" fill="#8d95ab">${s.laps} 圈 · L${s.start}-${s.end}</text>`;
    }
    x += w;
  });
  const step = total > 50 ? 10 : 5;
  for (let lap = 0; lap <= total; lap += step) {
    const tx = padL + innerW * lap / total;
    ticks += `<line x1="${tx.toFixed(1)}" y1="78" x2="${tx.toFixed(1)}" y2="86" stroke="#39415a"/>` +
             `<text x="${tx.toFixed(1)}" y="102" text-anchor="middle" font-size="10" fill="#6b7390">${lap}</text>`;
  }
  (pitLaps || []).forEach((pl) => {
    const tx = padL + innerW * pl / total;
    pits += `<line x1="${tx.toFixed(1)}" y1="22" x2="${tx.toFixed(1)}" y2="80" stroke="#ff3b2e" stroke-width="2" stroke-dasharray="4 3"/>` +
            `<text x="${tx.toFixed(1)}" y="16" text-anchor="middle" font-size="10" font-weight="700" fill="#ff3b2e">PIT · L${pl}</text>`;
  });
  return `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">${blocks}${ticks}${pits}</svg>`;
}

/* ---------------- SVG: 横向条形图(对比/轮胎选择) ---------------- */
function svgHBar(items, opts = {}) {
  const W = 920, rowH = 46, padL = opts.padL || 150, padR = 110, padT = 10;
  const H = padT + items.length * rowH + 14;
  const max = Math.max(...items.map((d) => d.value), 1e-6);
  let g = "";
  items.forEach((d, i) => {
    const y = padT + i * rowH;
    const bw = (W - padL - padR) * d.value / max;
    g += `<text x="${padL - 12}" y="${y + 22}" text-anchor="end" font-size="12.5" fill="#c7cde0">${esc(d.label)}</text>`;
    g += `<rect x="${padL}" y="${y + 8}" width="${Math.max(bw, 2).toFixed(1)}" height="22" rx="6" fill="${d.color || "#3f8cff"}" opacity="0.9"/>`;
    g += `<text x="${(padL + bw + 10).toFixed(1)}" y="${y + 24}" font-size="12" font-weight="800" fill="#e9edf5">${esc(d.valueText)}</text>`;
    if (d.sub) {
      g += `<text x="${padL}" y="${y + 42}" font-size="10" fill="#6b7390">${esc(d.sub)}</text>`;
    }
  });
  return `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">${g}</svg>`;
}

/* ---------------- SVG: 折线图(累计用时) ---------------- */
function svgLineChart(series, opts = {}) {
  const W = 920, H = 300, padL = 66, padR = 16, padT = 28, padB = 36;
  const n = Math.max(...series.map((s) => s.data.length), 2);
  let ymin = Infinity, ymax = -Infinity;
  series.forEach((s) => s.data.forEach((v) => {
    if (v < ymin) ymin = v;
    if (v > ymax) ymax = v;
  }));
  if (!isFinite(ymin)) { ymin = 0; ymax = 1; }
  const ypad = (ymax - ymin) * 0.08 || 1;
  ymin -= ypad; ymax += ypad;
  const X = (i) => padL + (W - padL - padR) * (i / (n - 1));
  const Y = (v) => padT + (H - padT - padB) * (1 - (v - ymin) / (ymax - ymin));
  let g = "";
  for (let k = 0; k <= 4; k++) {
    const v = ymin + (ymax - ymin) * k / 4, y = Y(v);
    g += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}" stroke="#232c41" stroke-dasharray="3 4"/>` +
         `<text x="${padL - 8}" y="${(y + 3.5).toFixed(1)}" text-anchor="end" font-size="10" fill="#6b7390">${opts.yFmt ? opts.yFmt(v) : v.toFixed(0)}</text>`;
  }
  const step = Math.max(1, Math.round((n - 1) / 8));
  for (let i = 0; i < n; i += step) {
    g += `<text x="${X(i).toFixed(1)}" y="${H - 14}" text-anchor="middle" font-size="10" fill="#6b7390">${i + 1}</text>`;
  }
  g += `<text x="${W - padR}" y="${H - 14}" text-anchor="end" font-size="10" fill="#6b7390">圈数</text>`;
  series.forEach((s, si) => {
    const pts = s.data.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
    g += `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="2.2" stroke-linejoin="round"/>`;
    g += `<g transform="translate(${padL + si * 170},16)"><rect width="14" height="4" rx="2" fill="${s.color}"/><text x="20" y="6" font-size="11" fill="#aab2c6">${esc(s.name)}</text></g>`;
  });
  return `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">${g}</svg>`;
}

/* ---------------- SVG: 超越圈数直方图 ---------------- */
function svgHist(counts, opts = {}) {
  const W = 920, H = 190, padL = 40, padR = 14, padT = 16, padB = 34;
  const n = counts.length || 1;
  const max = Math.max(...counts, 1);
  const bw = (W - padL - padR) / n;
  let g = `<line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" stroke="#2c3550"/>`;
  counts.forEach((c, i) => {
    const h = (H - padT - padB) * c / max;
    const x = padL + i * bw;
    const color = i === 0 ? "#ffd24d" : "#3f8cff";
    if (c > 0) {
      g += `<rect x="${(x + 1).toFixed(1)}" y="${(H - padB - h).toFixed(1)}" width="${Math.max(bw - 2, 1).toFixed(1)}" height="${h.toFixed(1)}" rx="2" fill="${color}" opacity="0.9"><title>第 ${i} 圈: ${c} 次</title></rect>`;
    }
    const step = n > 40 ? 10 : n > 16 ? 5 : 1;
    if (i % step === 0) {
      g += `<text x="${(x + bw / 2).toFixed(1)}" y="${H - padB + 16}" text-anchor="middle" font-size="9.5" fill="#6b7390">${i}</text>`;
    }
  });
  g += `<text x="${W - padR}" y="${H - 6}" text-anchor="end" font-size="10" fill="#6b7390">完成超越所需圈数(0 = 无需追赶)</text>`;
  if (opts.title) {
    g += `<text x="${padL}" y="12" font-size="11" fill="#aab2c6">${esc(opts.title)}</text>`;
  }
  return `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">${g}</svg>`;
}

/* ---------------- 时钟 / 选项卡 ---------------- */
function tickClock() {
  const el = $("#clock");
  if (el) el.textContent = new Date().toLocaleTimeString("zh-CN", { hour12: false });
}

function initTabs() {
  $$(".nav-pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".nav-pill").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const tab = btn.dataset.tab;
      $$(".pane").forEach((p) => p.classList.remove("active"));
      $(`#pane-${tab}`).classList.add("active");
      if (tab === "history") loadHistory();
      if (tab === "data") loadTables();
    });
  });
}

/* ---------------- 下拉框填充 ---------------- */
function fillDriverSelect(sel, codes, selected) {
  sel.innerHTML = codes.map((d) =>
    `<option value="${d.code}" ${d.code === selected ? "selected" : ""}>` +
    `${d.code} · ${esc(d.name)}(${esc(d.style)})</option>`).join("");
}


/* ============================================================
 * 模块一: 赛前模拟
 * ============================================================ */
async function runPrerace(logHistory = true) {
  const driver = $("#pr-driver").value;
  const btn = $("#pr-run");
  btn.disabled = true;
  $("#pr-results").innerHTML = `<div class="card"><div class="placeholder">正在轮胎库存约束下求解 DP 最优策略…</div></div>`;
  try {
    const res = await fetch("/api/prerace", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        driver: driver,
        mode: PR_MODE,
        tires: getTires("pr"),
        log: !!logHistory,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "请求失败");
    PR_HAS_RUN = true;
    renderPrerace(data);
  } catch (e) {
    $("#pr-results").innerHTML = `<div class="card"><div class="placeholder">出错: ${esc(e.message)}</div></div>`;
  } finally {
    btn.disabled = false;
  }
}

function renderPrerace(d) {
  const cmp = d.compare || {};
  const delta = cmp.delta_s;
  const deltaTxt = delta == null ? "—" :
    (delta > 0 ? `两停多花 +${delta.toFixed(1)}s` : delta < 0 ? `两停更快 ${delta.toFixed(1)}s` : "两停无损失");
  const pitStr = (d.pit_laps || []).length ? d.pit_laps.join(", ") : "—";

  const hero = `
    <div class="hero-top">
      <span class="hero-code">${esc(d.driver)}</span>
      <span class="hero-name">${esc(d.name)}</span>
      <span class="tag-soft">驾驶风格: ${esc(d.style)}</span>
      <span class="tag-soft">基准圈速: ${d.base_time.toFixed(2)}s</span>
      <span class="tag-soft">${esc(d.mode_name)}</span>
    </div>
    <div class="strategy-text">${esc(d.strategy_text)}</div>
    ${d.rule_note ? `<div class="rule-inline">&#9432; ${esc(d.rule_note)}</div>` : ""}
    <div class="metrics">
      <div class="metric"><div class="val">${d.stops}</div><div class="lab">停站次数</div></div>
      <div class="metric"><div class="val yellow">L${esc(pitStr)}</div><div class="lab">进站圈</div></div>
      <div class="metric"><div class="val green">${esc(d.total_display)}</div><div class="lab">预计总用时</div></div>
      <div class="metric"><div class="val ${delta > 0 ? "red" : "green"}">${esc(deltaTxt)}</div><div class="lab">正常 vs 两停</div></div>
      <div class="metric"><div class="val ${d.tire_penalty_s > 0 ? "red" : "green"}">${d.tire_penalty_s != null ? (d.tire_penalty_s > 0 ? "+" : "") + d.tire_penalty_s.toFixed(1) + "s" : "—"}</div><div class="lab">练习/排位磨损代价</div></div>
    </div>`;

  const stintRows = d.stints.map((s) => `
    <tr>
      <td>L${s.start} – L${s.end}</td>
      <td><span class="comp-chip" style="background:${compColor(s.compound)}">${s.compound}</span></td>
      <td>${s.laps}</td>
      <td>${esc(s.set_label || "—")}</td>
      <td>${s.avg_lap.toFixed(3)}s</td>
      <td>${s.start_lap_time.toFixed(2)}s → ${s.end_lap_time.toFixed(2)}s</td>
      <td>${s.stint_time.toFixed(1)}s</td>
    </tr>`).join("");

  const compareItems = [
    { label: "最优策略(正常)", value: cmp.normal_s || d.total_s, valueText: fmtSeconds(cmp.normal_s || d.total_s), color: "#23d18b", sub: esc(cmp.normal_text || d.strategy_text) },
    { label: "强制两停", value: cmp.two_s || (d.alt && d.alt.total_s) || d.total_s, valueText: fmtSeconds(cmp.two_s || (d.alt && d.alt.total_s) || d.total_s), color: "#3f8cff", sub: esc(cmp.two_text || (d.alt && d.alt.strategy_text) || "") },
  ];

  const lineSeries = [];
  if (d.series) {
    lineSeries.push({ name: d.mode_name, color: "#e10600", data: d.series.cumulative });
  }
  if (d.alt && d.alt.series) {
    lineSeries.push({ name: d.alt.mode_name, color: "#3f8cff", data: d.alt.series.cumulative });
  }

  $("#pr-results").innerHTML = `
    <div class="card">${hero}</div>
    <div class="card">
      <h3>策略时间线 <small style="color:var(--muted);font-weight:400">色块 = 轮胎配方,红虚线 = 进站</small></h3>
      <div class="chart-box">${svgTimeline(d.stints, d.pit_laps)}</div>
    </div>
    <div class="card">
      <h3>分段明细</h3>
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr><th>圈段</th><th>配方</th><th>圈数</th><th>使用套装</th><th>平均圈速</th><th>首/末圈圈速</th><th>段用时</th></tr></thead>
          <tbody>${stintRows}</tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <h3>正常策略 vs 强制两停</h3>
      <div class="chart-box">${svgHBar(compareItems)}</div>
      <p>${esc(deltaTxt)}(基于 DP 求解的期望总用时)</p>
      ${cmp.note ? `<div class="rule-inline">&#9432; ${esc(cmp.note)}</div>` : ""}
    </div>
    <div class="card">
      <h3>累计用时曲线</h3>
      <div class="chart-box">${svgLineChart(lineSeries, { yFmt: (v) => fmtSeconds(v) })}</div>
    </div>
    <div class="card">
      <h3>全场对比数据</h3>
      <p style="color:var(--muted)">点击直达站内数据表(自然过渡,不再跳转静态图片):</p>
      <div class="side-links">
        <button type="button" class="mini-link" data-goto="strategy">车手策略对比表</button>
        <button type="button" class="mini-link" data-goto="two_stop">强制两停对比表</button>
        <button type="button" class="mini-link" data-goto="compare">正常 vs 两停对比表</button>
      </div>
    </div>`;
}


/* ============================================================
 * 模块二: 赛中 Stackelberg 决策
 * ============================================================ */
function num(id, dft) {
  const v = parseFloat($(id).value);
  return isNaN(v) ? dft : v;
}

function readSimParams() {
  const noiseRaw = $("#st-noise").value.trim();
  return {
    pit_lane_transit: num("#st-pit-transit", 16.5),
    pit_lane_transit_scvsc: num("#st-pit-transit-scvsc", 6.0),
    pit_lane_transit_red: num("#st-pit-transit-red", 0),
    pit_normal_prob: num("#st-pit-prob", 0.85),
    pit_normal_mean: num("#st-pit-mean", 2.5),
    pit_abnormal_mean: num("#st-abn-mean", 5.0),
    pit_abnormal_std: num("#st-abn-std", 2.0),
    red_restart_gain: num("#st-red-gain", 0.35),
    red_restart_laps: num("#st-red-laps", 2),
    pit_reaction_laps: num("#st-react-laps", 2),
    form_noise_std: num("#st-form-noise", 0.2),
    lap_noise_std: noiseRaw === "" ? null : num("#st-noise", null),
    compound_offset: {
      C5: num("#off-C5", -0.5),
      C4: num("#off-C4", 0),
      C3: num("#off-C3", 1.0),
    },
  };
}

async function runStackelberg() {
  const myFit = findSet($("#st-my-fit") ? $("#st-my-fit").value : "");
  const rvFit = findSet($("#st-rival-fit") ? $("#st-rival-fit").value : "");
  const body = {
    scenario: ST_SCENARIO,
    my_driver: $("#st-my-driver").value,
    my_compound: getComp("#st-my-compound"),
    my_age: parseInt($("#st-my-age").value, 10),
    rival_driver: $("#st-rival-driver").value,
    rival_compound: getComp("#st-rival-compound"),
    rival_age: parseInt($("#st-rival-age").value, 10),
    current_lap: parseInt($("#st-lap").value, 10),
    gap_s: num("#st-gap", 0),
    n_sim: parseInt($("#st-n-sim").value, 10),
    params: readSimParams(),
    my_fit_compound: myFit ? myFit.compound : undefined,
    my_fit_wear: myFit ? myFit.wear : 0,
    rival_fit_compound: rvFit ? rvFit.compound : undefined,
    rival_fit_wear: rvFit ? rvFit.wear : 0,
    auto_fit: $("#st-auto-fit") ? $("#st-auto-fit").checked : true,
    tires: getTires("st"),
  };
  const btn = $("#st-run");
  btn.disabled = true;
  $("#st-results").innerHTML = `<div class="card"><div class="placeholder">蒙特卡洛模拟中(默认 N=10000)…</div></div>`;
  try {
    const res = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "请求失败");
    renderStackelberg(data);
  } catch (e) {
    $("#st-results").innerHTML = `<div class="card"><div class="placeholder">出错: ${esc(e.message)}</div></div>`;
  } finally {
    btn.disabled = false;
  }
}

function aiBadgeHtml(aist) {
  if (aist && aist.checked) {
    return aist.ok
      ? `<span class="risk-chip risk-low" id="ai-avail-badge">联网AI分析可用${aist.model ? " · " + esc(aist.model) : ""}</span>`
      : `<span class="risk-chip risk-high" id="ai-avail-badge">联网AI分析不可用 · 本地规则分析</span>`;
  }
  return `<span class="risk-chip risk-mid" id="ai-avail-badge">AI 可用性检测中…</span>`;
}

function renderStackelberg(d) {
  const my = d.inputs.my, rv = d.inputs.rival;
  const bestKey = d.recommendation.branch;
  const dec = d.decision || null;

  const summary = `
    <div class="hero-top">
      <span class="hero-code">${esc(d.scenario.toUpperCase())}</span>
      <span class="hero-name" style="font-size:18px">${esc(d.scenario_name)}</span>
      <span class="tag-soft">${esc(my.name)}(${esc(my.compound)}·胎龄${my.age}) vs ${esc(rv.name)}(${esc(rv.compound)}·胎龄${rv.age})</span>
      <span class="tag-soft">${d.fit_mode === "auto" ? "系统自动选胎" : "手动选胎"}: 我方 ${esc(my.fit_compound)}(磨损${my.fit_wear}圈) / 对手 ${esc(rv.fit_compound)}(磨损${rv.fit_wear}圈)</span>
      <span class="tag-soft">第 ${d.current_lap} 圈 / 剩余 ${d.remaining} 圈</span>
      <span class="tag-soft">差距 ${d.gap_s > 0 ? "+" : ""}${d.gap_s}s</span>
      <span class="tag-soft">N = ${d.n_sim.toLocaleString()}</span>
    </div>
    ${d.pace_note ? `<div class="branch-note" style="margin-top:8px">&#9432; 圈速差拆解 —— ${esc(d.pace_note)}</div>` : ""}
    <div class="rec-banner" style="margin-top:12px">
      <span class="rec-icon">&#9654;</span>
      <span>${esc(d.recommendation.text)}</span>
    </div>`;

  const branchesHtml = d.branches.map((b) => {
    const isBest = b.key === bestKey;
    const rate = b.success_rate;
    const se = (100 * Math.sqrt(Math.max(rate, 0.01) / 100 *
      (1 - Math.min(rate, 99.99) / 100) / Math.max(d.n_sim, 1))).toFixed(1);
    const otBox = (b.overtake_mean != null && (b.overtake_median || 0) > 0)
      ? `<div class="overtake-box">
           <div><div class="lab">若成功,完成超越约需</div></div>
           <div><span class="big">${b.overtake_mean.toFixed(1)}</span><span class="unit">圈</span></div>
         </div>
         <div class="branch-note">中位 <b>${b.overtake_median.toFixed(0)}</b> 圈 · P90 <b>${b.overtake_p90.toFixed(0)}</b> 圈 ·
           ${b.instant_pct.toFixed(0)}% 的成功样本在 1 圈内获得位置</div>`
      : `<div class="overtake-box">
           <div><div class="lab">超越方式</div></div>
           <div><span class="big">${b.overtake_mean != null ? b.overtake_mean.toFixed(1) : "—"}</span><span class="unit">圈(0=立即)</span></div>
         </div>
         <div class="branch-note">${esc(b.pos_note)}</div>`;
    const riskClass = b.risk_level === "高" ? "risk-high" : b.risk_level === "中" ? "risk-mid" : "risk-low";
    const riskBox = `
      <div class="risk-row">
        <span class="risk-chip ${riskClass}">被反超风险 ${esc(b.risk_level || "—")}</span>
        <span class="risk-item">得而复失 <b>${(b.relost_rate ?? 0).toFixed(1)}%</b></span>
        <span class="risk-item">策略增益 <b>${(b.strategy_gain_pp ?? 0) > 0 ? "+" : ""}${(b.strategy_gain_pp ?? 0).toFixed(1)}pp</b></span>
        <span class="risk-item">不动基线 <b>${(b.baseline_rate ?? 0).toFixed(1)}%</b></span>
        <span class="risk-item">扰动区间 <b>${b.robustness ? `${b.robustness.min.toFixed(1)}~${b.robustness.max.toFixed(1)}%` : "—"}</b></span>
        <span class="risk-item">翻转频繁度 <b>${(b.swap_rate ?? 0).toFixed(1)}%</b></span>
      </div>`;
    const gainBox = b.restart_gain ? `
      <div class="branch-note">静态发车增益: 我方 <b>-${b.restart_gain.my.toFixed(2)}s/圈</b> ·
        对手 <b>-${b.restart_gain.rival.toFixed(2)}s/圈</b> ·
        净得利 <b>${b.restart_gain.net > 0 ? "+" : ""}${b.restart_gain.net.toFixed(2)}s</b>
        × ${b.restart_gain.laps} 圈</div>` : "";
    return `
      <div class="branch-card ${isBest ? "best" : ""}">
        <div class="branch-head">
          <span class="branch-label">${esc(b.label)}</span>
          ${isBest ? `<span class="branch-tag">${dec && dec.kind === "exogenous" ? "期望更高情景" : "更优动作"}</span>` : ""}
        </div>
        <div class="branch-desc">${esc(b.desc)}</div>
        <div class="rate-row">
          <span class="rate-val ${rateClass(rate)}">${rate.toFixed(1)}</span>
          <span class="rate-unit">% 成功率 ±${se}pp · ${b.n_success.toLocaleString()}/${d.n_sim.toLocaleString()} 次</span>
        </div>
        <div class="rate-bar"><i style="width:${Math.max(rate, 0.5)}%"></i></div>
        ${riskBox}
        ${otBox}
        ${gainBox}
        <div class="branch-note">期望完赛时间差 <b>${b.mean_finish_delta > 0 ? "+" : ""}${b.mean_finish_delta.toFixed(2)}s</b>(正 = 我方先完赛)</div>
        <div class="chart-box">${svgHist(b.overtake_hist)}</div>
      </div>`;
  }).join("");

  const wearMap = d.tire_choice_wear || {};
  const tireItems = Object.entries(d.tire_choice || {}).map(([c, v]) => {
    const w = wearMap[c] || 0;
    return {
      label: `${c} ${w > 0 ? `(已用${w}圈)` : "新胎"} 跑 ${d.remaining} 圈`,
      value: v,
      valueText: `${(v / 60).toFixed(2)} min`,
      color: compColor(c),
    };
  });

  const CHIP_LABELS = {
    pit_lane_transit: "通道(正常)",
    pit_lane_transit_scvsc: "通道(SC/VSC)",
    pit_lane_transit_red: "红旗换胎损失",
    pit_normal_prob: "正常换胎概率",
    pit_normal_mean: "正常换胎均值",
    pit_abnormal_mean: "异常换胎均值",
    lap_noise_std: "圈速σ(手动)",
    my_sigma: "我方圈速σ",
    rival_sigma: "对手圈速σ",
    noise_mode: "σ来源",
    red_restart_gain: "红旗发车增益",
    red_restart_laps: "发车增益圈数",
    pit_reaction_laps: "跟进反应延迟",
    form_noise_std: "当日状态σ",
  };
  const chips = Object.entries(d.params_used || {}).filter(([, v]) => v != null).map(([k, v]) => {
    const text = (k === "compound_offset")
      ? `配方偏移 C5=${v.C5} C4=${v.C4} C3=${v.C3}`
      : `${CHIP_LABELS[k] || k} = ${typeof v === "number" ? v : JSON.stringify(v)}`;
    return `<span>${esc(text)}</span>`;
  }).join("");

  const ruleBox = d.rule_note ? `
    <div class="card rule-note">
      <h3>旗种规则要点</h3>
      <p>${esc(d.rule_note)}</p>
    </div>` : "";

  const inf = d.rival_pit_inference || null;
  const verdictClass = (dec && /不值得|不动/.test(dec.verdict)) ? "risk-high"
    : (dec && /临界/.test(dec.verdict)) ? "risk-mid" : "risk-low";
  const decBox = dec ? `
    <div class="card">
      <h3>决策评估 <small style="color:var(--muted);font-weight:400">我方决策 = 是否执行该策略/选哪个动作;对手跟不跟是外生不确定事件,只能按概率加权</small></h3>
      <div class="branch-desc">${esc(dec.question)}</div>
      <div class="infer-row">
        <span class="risk-chip ${verdictClass}">${esc(dec.verdict)}</span>
        <span class="risk-item">不动基线 <b>${(dec.hold?.success_rate ?? 0).toFixed(1)}%</b></span>
        ${dec.follow_prob != null ? `<span class="risk-item">对手跟进概率(外生) <b>${(dec.follow_prob * 100).toFixed(0)}%</b></span>` : ""}
        ${dec.breakeven_follow != null ? `<span class="risk-item">临界跟进概率 <b>${(dec.breakeven_follow * 100).toFixed(0)}%</b></span>` : ""}
      </div>
      ${(dec.options || []).map((o) => `
        <div class="rob-row">
          <span class="rob-name">${esc(o.label)}</span>
          <div class="rob-bar"><i style="width:${Math.max(o.success_rate, 0.5)}%"></i></div>
          <span class="rob-val">${o.success_rate.toFixed(1)}% (${(o.strategy_gain_pp ?? 0) > 0 ? "+" : ""}${(o.strategy_gain_pp ?? 0).toFixed(1)}pp)</span>
        </div>`).join("")}
      <div class="rob-row">
        <span class="rob-name">${esc(dec.hold?.label || "不动")}</span>
        <div class="rob-bar"><i style="width:${Math.max(dec.hold?.success_rate ?? 0, 0.5)}%"></i></div>
        <span class="rob-val">${(dec.hold?.success_rate ?? 0).toFixed(1)}%</span>
      </div>
      <div class="branch-note" style="margin-top:8px"><b>决策依据:</b> ${esc(dec.verdict_text)}</div>
      ${dec.follow_prob != null ? `<div class="branch-note"><b>对手行为评估(外生不确定事件,非我方决策):</b> 跟进概率约 ${(dec.follow_prob * 100).toFixed(0)}% —— ${esc(dec.follow_basis || "")}</div>` : ""}
      ${inf ? `<div class="branch-note"><b>对手轮胎证据(用于判断其跟进概率):</b> ${esc(inf.status)}(置信度${esc(inf.confidence)}) —— ${esc(inf.reasoning || "")}</div>` : ""}
    </div>` : "";

  const rob = d.robustness;
  const robBox = rob ? `
    <div class="card">
      <h3>鲁棒性检验 <small style="color:var(--muted);font-weight:400">最优分支「${esc(rob.branch)}」在关键参数扰动下的成功率</small></h3>
      <div class="rob-range">名义 <b>${rob.nominal.toFixed(1)}%</b> ·
        扰动区间 <b>${rob.min.toFixed(1)}% ~ ${rob.max.toFixed(1)}%</b> ·
        跨度 <b>${rob.range.toFixed(1)}pp</b>
        <span class="risk-item">(${rob.range <= 15 ? "结论较稳健" : rob.range <= 30 ? "结论中等稳健" : "结论对参数敏感,需谨慎采信"})</span>
      </div>
      <div class="rob-list">
        ${rob.cases.map((c) => `
          <div class="rob-row">
            <span class="rob-name">${esc(c.name)}</span>
            <div class="rob-bar"><i style="width:${Math.max(c.success_rate, 0.5)}%"></i></div>
            <span class="rob-val">${c.success_rate.toFixed(1)}%</span>
          </div>`).join("")}
      </div>
      <p class="fld-hint" style="margin-top:8px">扰动项: 衰减斜率 ±20% / 基准圈速 ±0.3s/圈 / 差距 ±2s / 波动 σ×1.5;
        区间越窄,单点成功率越可信 —— 这是模型鲁棒性的直接度量。</p>
    </div>` : "";

  $("#st-results").innerHTML = `
    <div class="card">${summary}</div>
    ${decBox}
    <div class="branch-note" style="margin:10px 2px 6px">
      <b>${dec && dec.kind === "exogenous" ? "情景模拟" : "动作对比"}:</b>
      ${dec && dec.kind === "exogenous"
        ? "对手跟不跟是外生不确定事件 —— 以下按条件句呈现各情景,并非我方可选动作;我方决策已在上方「决策评估」中给出。"
        : "以下均为我方可执行的动作,直接对比选择。"}
    </div>
    <div class="branch-grid">${branchesHtml}</div>
    ${robBox}
    ${ruleBox}
    <div class="card">
      <h3>轮胎配方选择 <small style="color:var(--muted);font-weight:400">我方 ${esc(my.name)} 从库存最优套装起步跑完剩余 ${d.remaining} 圈的期望用时</small></h3>
      <div class="chart-box">${svgHBar(tireItems, { padL: 190 })}</div>
    </div>
    <div class="card">
      <h3>AI 策略分析 ${aiBadgeHtml(d.ai_status || (META && META.ai_status))}
        <small style="color:var(--muted);font-weight:400">围绕「是否值得执行」的策略解读(对手反应以条件句呈现)</small></h3>
      <div class="btn-row" style="margin-bottom:8px">
        <button type="button" class="btn-ghost" id="ai-btn">生成 AI 策略分析</button>
        <button type="button" class="btn-ghost" id="ai-test-btn">测试 API 连接</button>
      </div>
      <div id="ai-panel" class="ai-panel">
        <p class="fld-hint">点击按钮生成分析。${(d.ai_status && d.ai_status.ok) || (META && META.ai_status && META.ai_status.ok)
          ? "联网AI分析可用,将由大模型生成。"
          : "联网AI不可用,将使用本地规则分析(修复后点「测试 API 连接」刷新状态)。"}</p>
      </div>
    </div>
    <div class="card">
      <h3>本次模拟参数</h3>
      <div class="param-chips">${chips}</div>
      <p style="margin-top:10px">模型说明: 圈速 = (基准圈速 + 配方偏移) ÷ 归一化性能保持率(暖胎+衰减形状,
        配方速度差由偏移承担);每圈圈速 ~ N(确定性圈速, σ²) —— σ 为 FastF1 摩纳哥正赛实测标定(近似正态),
        成功率/被反超风险均为该概率口径下的频率估计;
        进站损失 = 通道行驶 + 换胎混合分布(85% N(2.5,0.3)s + 15% N(5.0,2.0)s),落在实际进站圈上;
        位置模型: 名次交换分两条通道 —— 进站窗口交换自动完成,赛道超越需逐圈低概率尝试
        (摩纳哥超车极难,基础 ${(d.params_used?.monaco_pass_base ?? 0.035)} /圈,每 +1s/圈 速度优势 +${d.params_used?.monaco_pass_slope ?? 0.12});
        决策语义: 我方决策 = 是否执行/选动作;对手跟不跟 = 外生不确定事件,按跟进概率加权;
        超越圈数 = 成功样本中"完成永久反超"距当前的圈数(0 = 无需追赶)。</p>
    </div>`;

  LAST_SIM = d;
  const aiBtn = $("#ai-btn");
  if (aiBtn) aiBtn.addEventListener("click", runAiAnalysis);
  const aiTestBtn = $("#ai-test-btn");
  if (aiTestBtn) aiTestBtn.addEventListener("click", runAiTest);
}

async function runAiTest() {
  const panel = $("#ai-panel"), btn = $("#ai-test-btn");
  if (!panel) return;
  if (btn) btn.disabled = true;
  panel.innerHTML = `<p class="fld-hint">API 连通性测试中…</p>`;
  try {
    const res = await fetch("/api/ai/test");
    const d = await res.json();
    const badge = $("#ai-avail-badge");
    if (badge) badge.outerHTML = aiBadgeHtml({ checked: true, ok: d.ok, model: d.model });
    panel.innerHTML = d.ok
      ? `<div class="ai-src ai-llm">连接成功</div>
         <pre class="ai-text">模型 ${esc(d.model)} · 延迟 ${d.latency_ms}ms · 模型回复: ${esc(d.reply || "OK")}</pre>
         <div class="fld-hint">接口 ${esc(d.base_url || "")} —— 可点击「生成 AI 策略分析」</div>`
      : `<div class="ai-src ai-local">连接失败</div>
         <pre class="ai-text">${esc(d.error || "未知错误")}</pre>
         <div class="fld-hint">接口 ${esc(d.base_url || "")} · 模型 ${esc(d.model || "")} · 延迟 ${d.latency_ms}ms。
           排查: ①web/ai_config.local.json 的 api_key 是否正确;②网络能否访问 api.deepseek.com;③密钥额度是否用尽</div>`;
  } catch (e) {
    panel.innerHTML = `<div class="ai-src ai-local">测试失败</div>
      <pre class="ai-text">${esc(e.message)}</pre>
      <div class="fld-hint">后端服务不可达 —— 请先启动 web\\server.py</div>`;
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function runAiAnalysis() {
  const panel = $("#ai-panel"), btn = $("#ai-btn");
  if (!panel || !LAST_SIM) return;
  btn.disabled = true;
  panel.innerHTML = `<p class="fld-hint">分析生成中…</p>`;
  try {
    const res = await fetch("/api/ai", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ result: LAST_SIM }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "请求失败");
    panel.innerHTML = `
      <div class="ai-src ${data.source === "llm" ? "ai-llm" : "ai-local"}">
        ${data.source === "llm" ? `大模型分析 · ${esc(data.model || "")}` : "本地规则分析"}
      </div>
      <pre class="ai-text">${esc(data.text)}</pre>
      <div class="fld-hint">${esc(data.note || "")}</div>`;
  } catch (e) {
    panel.innerHTML = `<p class="fld-hint">分析失败: ${esc(e.message)}</p>`;
  } finally {
    btn.disabled = false;
  }
}

/* ============================================================
 * 模块三: 历史记录
 * ============================================================ */
async function loadHistory() {
  try {
    const res = await fetch("/api/history");
    const data = await res.json();
    renderHistory(data.records || []);
  } catch (e) {
    $("#hi-body").innerHTML = `<tr><td colspan="5" class="placeholder">加载失败: ${esc(e.message)}</td></tr>`;
  }
}

function renderHistory(records) {
  if (!records.length) {
    $("#hi-body").innerHTML = `<tr><td colspan="5" class="placeholder">暂无历史记录 — 运行一次模拟后将自动记录</td></tr>`;
    return;
  }
  $("#hi-body").innerHTML = records.map((r, i) => `
    <tr>
      <td style="white-space:nowrap">${esc(r.ts || "")}</td>
      <td>${r.type === "simulate" ? "赛中决策" : "赛前模拟"}</td>
      <td><b>${esc(r.title || "")}</b><br><small style="color:var(--muted)">${esc(r.detail || "")}</small></td>
      <td>${esc(r.result || "")}</td>
      <td><button class="btn-ghost" data-refill="${i}">回填</button></td>
    </tr>`).join("");
  $$("#hi-body [data-refill]").forEach((btn) => {
    btn.addEventListener("click", () => refillForm(records[parseInt(btn.dataset.refill, 10)]));
  });
}

function switchTab(tab) {
  $$(".nav-pill").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  $$(".pane").forEach((p) => p.classList.toggle("active", p.id === `pane-${tab}`));
  if (tab === "history") loadHistory();
  if (tab === "data") loadTables();
}

function setScenario(sc) {
  ST_SCENARIO = sc;
  $$("#sc-cards .sc-card").forEach((b) => b.classList.toggle("active", b.dataset.sc === sc));
}

function refillForm(r) {
  const p = r.payload || {};
  if (p.type === "simulate") {
    let sc = p.scenario || "undercut";
    if (sc === "red_flag") sc = "red";
    if (sc === "sc" && p.sc_type === "RED") sc = "red";   // 旧记录旗种映射
    setScenario(sc);
    if (p.my_driver) $("#st-my-driver").value = p.my_driver;
    if (p.my_compound) setComp("#st-my-compound", p.my_compound);
    if (p.my_age != null) $("#st-my-age").value = p.my_age;
    if (p.rival_driver) $("#st-rival-driver").value = p.rival_driver;
    if (p.rival_compound) setComp("#st-rival-compound", p.rival_compound);
    if (p.rival_age != null) $("#st-rival-age").value = p.rival_age;
    if (p.current_lap != null) $("#st-lap").value = p.current_lap;
    if (p.gap_s != null) $("#st-gap").value = p.gap_s;
    if (p.n_sim) $("#st-n-sim").value = String(p.n_sim);
    if (Array.isArray(p.tires) && p.tires.length) {
      TIRE_SETS.st = p.tires.map((t) => ({ ...t }));
      renderTireList("st");
      syncFitSelects();
    }
    const autoEl = $("#st-auto-fit");
    if (autoEl && p.auto_fit != null) {
      autoEl.checked = !!p.auto_fit;
    }
    syncFitMode();
    if (p.my_fit_compound != null || p.my_fit_wear != null) {
      const hit = TIRE_SETS.st.find((t) =>
        t.compound === p.my_fit_compound && t.wear === (p.my_fit_wear ?? t.wear)
        && t.available);
      if (hit) $("#st-my-fit").value = hit.id;
    }
    if (p.rival_fit_compound != null || p.rival_fit_wear != null) {
      const hit = TIRE_SETS.st.find((t) =>
        t.compound === p.rival_fit_compound && t.wear === (p.rival_fit_wear ?? t.wear)
        && t.available);
      if (hit) $("#st-rival-fit").value = hit.id;
    }
    updateFitNote();
    switchTab("stackelberg");
    runStackelberg();
  } else {
    if (p.driver) $("#pr-driver").value = p.driver;
    PR_MODE = p.mode || "normal";
    $$("#pr-mode .seg-btn").forEach((b) => b.classList.toggle("active", b.dataset.mode === PR_MODE));
    if (Array.isArray(p.tires) && p.tires.length) {
      TIRE_SETS.pr = p.tires.map((t) => ({ ...t }));
      renderTireList("pr");
    }
    switchTab("prerace");
    runPrerace();
  }
}

/* ============================================================
 * 初始化
 * ============================================================ */
function resetAdvParams() {
  const d = (META && META.defaults) || {};
  $("#st-pit-transit").value = d.pit_lane_transit ?? 16.5;
  $("#st-pit-transit-scvsc").value = d.pit_lane_transit_scvsc ?? 6.0;
  $("#st-pit-transit-red").value = d.pit_lane_transit_red ?? 0;
  $("#st-pit-prob").value = d.pit_normal_prob ?? 0.85;
  $("#st-pit-mean").value = d.pit_normal_mean ?? 2.5;
  $("#st-abn-mean").value = d.pit_abnormal_mean ?? 5.0;
  $("#st-abn-std").value = d.pit_abnormal_std ?? 2.0;
  $("#st-red-gain").value = d.red_restart_gain ?? 0.35;
  $("#st-red-laps").value = d.red_restart_laps ?? 2;
  $("#st-react-laps").value = d.pit_reaction_laps ?? 2;
  $("#st-form-noise").value = d.form_noise_std ?? 0.2;
  $("#st-noise").value = "";   // 留空 = 车手级实测 σ 自动标定
  const off = d.compound_offset || {};
  $("#off-C5").value = off.C5 ?? -0.5;
  $("#off-C4").value = off.C4 ?? 0;
  $("#off-C3").value = off.C3 ?? 1.0;
}

async function init() {
  initTabs();
  setInterval(tickClock, 1000);
  tickClock();

  try {
    const res = await fetch("/api/meta");
    META = await res.json();
  } catch (e) {
    $("#pr-results").innerHTML = `<div class="card"><div class="placeholder">无法连接后端服务: ${esc(e.message)}</div></div>`;
    return;
  }

  const drivers = META.drivers || [];
  fillDriverSelect($("#pr-driver"), drivers.filter((d) => d.has_strategy), "LEC");
  fillDriverSelect($("#st-my-driver"), drivers, "LEC");
  fillDriverSelect($("#st-rival-driver"), drivers, "VER");
  initCompPills();
  setComp("#st-my-compound", "C4");
  setComp("#st-rival-compound", "C4");
  initTires();
  resetAdvParams();

  // 快捷入口(数据表)事件委托 — 覆盖静态与动态生成的按钮
  document.addEventListener("click", onGotoClick);
  const dtRefresh = $("#dt-refresh");
  if (dtRefresh) dtRefresh.addEventListener("click", () => {
    TABLES = null;
    loadTables();
  });

  $("#pr-run").addEventListener("click", runPrerace);
  $$("#pr-mode .seg-btn").forEach((b) => b.addEventListener("click", () => {
    PR_MODE = b.dataset.mode;
    $$("#pr-mode .seg-btn").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    // 仅当已出过结果时才随模式切换重算,避免"未点击运行就输出结果"
    if (PR_HAS_RUN) runPrerace();
  }));

  $$("#sc-cards .sc-card").forEach((b) =>
    b.addEventListener("click", () => setScenario(b.dataset.sc)));
  setScenario("undercut");

  $("#st-run").addEventListener("click", runStackelberg);
  $("#adv-reset").addEventListener("click", resetAdvParams);
  $("#hi-refresh").addEventListener("click", loadHistory);
  $("#hi-clear").addEventListener("click", async () => {
    if (!confirm("确认清空全部历史记录?")) return;
    await fetch("/api/history/clear", { method: "POST" });
    loadHistory();
  });

  // 初始不自动运行赛前模拟 —— 由用户点击「运行赛前模拟」触发
}

document.addEventListener("DOMContentLoaded", init);

/* ---------------- 胎型三色胶囊(白=C3 / 黄=C4 / 红=C5) ---------------- */
function getComp(sel) {
  const el = $(sel);
  return el && el.dataset.value ? el.dataset.value : "C4";
}

function setComp(sel, code) {
  const el = $(sel);
  if (!el) return;
  el.dataset.value = code;
  $$(sel + " .comp-pill").forEach((b) =>
    b.classList.toggle("active", b.dataset.comp === code));
}

function initCompPills() {
  $$(".comp-pills .comp-pill").forEach((b) => {
    b.addEventListener("click", () => setComp("#" + b.parentElement.id, b.dataset.comp));
  });
}

/* ---------------- 轮胎高级选项(库存状态,练习/排位后) ---------------- */
const TIRE_SETS = { pr: [], st: [] };

function cloneTireTemplate() {
  return ((META && META.tire_template) || []).map((t) => ({ ...t }));
}

function renderTireList(which) {
  const wrap = $(`#${which}-tire-list`);
  if (!wrap) return;
  const sets = TIRE_SETS[which];
  const maxSets = (META && META.race_sets_max) || 7;
  const nAvail = sets.filter((t) => t.available).length;
  wrap.innerHTML =
    `<div class="tire-head">
       <span class="tire-count ${nAvail > maxSets ? "over" : ""}">正赛可用 ${nAvail} / ${maxSets} 套</span>
       <span class="tire-count-note">灰色 = 练习赛后已交还(FIA 30.5i):P1/P2/P3 后各 2 套</span>
     </div>
     <div class="tire-legend"><span>套装</span><span>配方 / 来源</span><span>已用圈数</span><span>可用</span></div>` +
    sets.map((t, i) => `
      <div class="tire-row ${t.available ? "" : "off"} ${t.locked ? "locked" : ""}" data-idx="${i}">
        <span class="tire-idx">${esc(t.id)}</span>
        <span class="comp-chip" style="background:${compColor(t.compound)}">${t.compound}</span>
        <span class="tire-src">${esc(t.source)}${t.wear > 0 ? ` · 已用${t.wear}圈` : " · 新胎"}${t.locked ? " · 已交还" : ""}</span>
        <input class="tire-wear" type="number" min="0" max="77" value="${t.wear}" title="已用圈数(= 装上后初始胎龄)" ${t.locked ? "disabled" : ""}>
        <label class="tire-avail" title="${t.locked ? "练习赛后已交还(FIA 30.5i),不可用于正赛" : "是否可用于正赛"}">
          <input type="checkbox" ${t.available ? "checked" : ""} ${t.locked ? "disabled" : ""}>
        </label>
      </div>`).join("");

  wrap.querySelectorAll(".tire-row").forEach((row) => {
    const idx = parseInt(row.dataset.idx, 10);
    const t = TIRE_SETS[which][idx];
    const wearEl = row.querySelector(".tire-wear");
    if (wearEl && !t.locked) {
      wearEl.addEventListener("change", (e) => {
        const v = Math.max(0, Math.min(77, parseInt(e.target.value, 10) || 0));
        TIRE_SETS[which][idx].wear = v;
        e.target.value = v;
        const src = row.querySelector(".tire-src");
        const tt = TIRE_SETS[which][idx];
        src.textContent = `${tt.source}${tt.wear > 0 ? ` · 已用${tt.wear}圈` : " · 新胎"}`;
        syncFitSelects();
      });
    }
    const availEl = row.querySelector(".tire-avail input");
    if (availEl && !t.locked) {
      availEl.addEventListener("change", (e) => {
        const cur = TIRE_SETS[which].filter((x, j) => j !== idx && x.available).length;
        if (e.target.checked && cur >= maxSets) {
          e.target.checked = false;
          alert(`正赛可用轮胎至多 ${maxSets} 套(练习赛后已交还 6 套,FIA 30.5i),请先取消勾选其它套装`);
          return;
        }
        TIRE_SETS[which][idx].available = e.target.checked;
        row.classList.toggle("off", !e.target.checked);
        renderTireList(which);
        syncFitSelects();
      });
    }
  });
}

function getTires(which) {
  return TIRE_SETS[which].map((t) => ({ ...t }));
}

function resetTires(which, fresh = false) {
  TIRE_SETS[which] = cloneTireTemplate();
  if (fresh) TIRE_SETS[which].forEach((t) => {
    if (!t.locked) t.wear = 0;      // 练习套保持锁定状态
  });
  renderTireList(which);
  syncFitSelects();
}

function tireLabel(t) {
  return `${t.id} · ${t.compound} · ${t.source} · ${t.wear > 0 ? `已用${t.wear}圈` : "新胎"}`;
}

function fillFitSelect(sel, sets, preferred) {
  const avail = sets.filter((t) => t.available);
  sel.innerHTML = (avail.length ? avail : sets).map((t) =>
    `<option value="${esc(t.id)}">${esc(tireLabel(t))}</option>`).join("");
  if (preferred && (avail.length ? avail : sets).some((t) => t.id === preferred)) {
    sel.value = preferred;
  }
}

function syncFitSelects() {
  const sets = TIRE_SETS.st;
  const mySel = $("#st-my-fit"), rvSel = $("#st-rival-fit");
  if (!mySel || !rvSel) return;
  const keepMy = mySel.value, keepRv = rvSel.value;
  fillFitSelect(mySel, sets, keepMy || "T5");
  fillFitSelect(rvSel, sets, keepRv || "T9");
  updateFitNote();
}

function findSet(id) {
  return TIRE_SETS.st.find((t) => t.id === id) || null;
}

function syncFitMode() {
  const autoEl = $("#st-auto-fit");
  const auto = autoEl ? autoEl.checked : true;
  const mySel = $("#st-my-fit"), rvSel = $("#st-rival-fit");
  if (mySel) mySel.disabled = auto;
  if (rvSel) rvSel.disabled = auto;
  const manualWrap = $("#st-fit-manual");
  if (manualWrap) manualWrap.classList.toggle("dimmed", auto);
  const note = $("#st-fit-wear-note");
  if (note && auto) {
    note.textContent = "自动模式: 系统将从库存中挑选最优套装(含磨损起算胎龄)";
  }
}

function updateFitNote() {
  const note = $("#st-fit-wear-note");
  if (!note) return;
  const my = findSet($("#st-my-fit") ? $("#st-my-fit").value : "");
  const rv = findSet($("#st-rival-fit") ? $("#st-rival-fit").value : "");
  note.textContent = `我方换上: ${my ? tireLabel(my) : "—"} | 对手换上: ${rv ? tireLabel(rv) : "—"}`;
}

function initTires() {
  TIRE_SETS.pr = cloneTireTemplate();
  TIRE_SETS.st = cloneTireTemplate();
  renderTireList("pr");
  renderTireList("st");
  syncFitSelects();

  $("#pr-tire-reset").addEventListener("click", () => resetTires("pr", false));
  $("#pr-tire-fresh").addEventListener("click", () => resetTires("pr", true));
  $("#st-tire-reset").addEventListener("click", () => resetTires("st", false));
  $("#st-tire-fresh").addEventListener("click", () => resetTires("st", true));
  ["#st-my-fit", "#st-rival-fit"].forEach((sel) => {
    const el = $(sel);
    if (el) el.addEventListener("change", updateFitNote);
  });
  const autoEl = $("#st-auto-fit");
  if (autoEl) autoEl.addEventListener("change", syncFitMode);
  syncFitMode();
}

/* ---------------- 数据总览(站内表格,替代静态图片跳转) ---------------- */
let TABLES = null;

async function loadTables() {
  if (TABLES) return TABLES;
  const wrap = $("#data-tables");
  if (wrap) wrap.innerHTML = `<div class="card"><div class="placeholder">正在加载数据表…</div></div>`;
  try {
    const res = await fetch("/api/tables");
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "请求失败");
    TABLES = data.tables || [];
    renderTables();
  } catch (e) {
    if (wrap) wrap.innerHTML = `<div class="card"><div class="placeholder">加载失败: ${esc(e.message)}</div></div>`;
  }
  return TABLES || [];
}

function renderTables() {
  const wrap = $("#data-tables");
  if (!wrap) return;
  wrap.innerHTML = TABLES.map((t) => `
    <div class="card table-block" id="tb-${esc(t.key)}">
      <h3>${esc(t.title)} <small style="color:var(--muted);font-weight:400">${t.rows.length} 行</small></h3>
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr>${t.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
          <tbody>${t.rows.map((r) => `<tr>${r.map((v) => `<td>${esc(v)}</td>`).join("")}</tr>`).join("")}</tbody>
        </table>
      </div>
    </div>`).join("");
}

function gotoTable(key) {
  switchTab("data");
  loadTables().then(() => {
    const el = document.querySelector("#tb-" + key);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "start" });
    el.classList.remove("flash");
    void el.offsetWidth;            // 重启动画
    el.classList.add("flash");
  });
}

function onGotoClick(e) {
  const el = e.target;
  if (!el || !el.closest) return;
  const btn = el.closest("[data-goto]");
  if (btn && btn.dataset.goto) gotoTable(btn.dataset.goto);
}
