const REPO_URL = "https://github.com/JarvisLee511/mlb-playoff-predictor-2026";
const AL = "#2a9d8f", NL = "#e76f51";
const MODEL_LABELS = {
  elo: "Elo baseline",
  lr: "Logistic regression",
  xgb: "XGBoost",
  ens: "Ensemble (calibrated)",
  skl: "Poisson-Skellam",
};
const PLOT_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#8b98a5", size: 12 },
  margin: { l: 170, r: 20, t: 10, b: 40 },
};

document.getElementById("repo-link").href = REPO_URL;

// ---- tabs ----
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
const json = (f) => fetch("data/" + f).then((r) => r.json());

// ---- today + tomorrow ----
const gameCard = (g) => {
  const p = g.p_home_ens != null ? g.p_home_ens : g.p_home_lr;
  const probs = [
    ["Ens", g.p_home_ens], ["Elo", g.p_home_elo], ["LogReg", g.p_home_lr],
    ["XGB", g.p_home_xgb], ["Skellam", g.p_home_skl],
  ].filter(([, v]) => v != null).map(([n, v]) => `${n} ${pct(v, 0)}`).join(" · ");
  return `<div class="game-card">
    <div class="time">${g.game_time_et} ET</div>
    <div class="matchup">${g.away_name} @ ${g.home_name}</div>
    <div class="pitchers">${g.away_pitcher} vs ${g.home_pitcher}</div>
    <div class="prob-bar">
      <div class="home" style="width:${(p * 100).toFixed(1)}%">${pct(p, 0)}</div>
      <div class="away" style="width:${((1 - p) * 100).toFixed(1)}%">${pct(1 - p, 0)}</div>
    </div>
    <div class="model-probs">home win — ${probs}</div>
  </div>`;
};

json("today.json").then((d) => {
  document.getElementById("today-title").textContent = `Predictions for ${d.date}`;
  const wrap = document.getElementById("today-games");
  wrap.innerHTML = d.games.length
    ? d.games.map(gameCard).join("")
    : '<p class="empty">No regular-season games scheduled today.</p>';

  if (d.tomorrow && d.tomorrow.length) {
    document.getElementById("tomorrow-title").innerHTML =
      `Tomorrow (${d.tomorrow_date}) <span class="hint">— preview; re-predicted and locked tomorrow morning</span>`;
    document.getElementById("tomorrow-games").innerHTML = d.tomorrow.map(gameCard).join("");
  }
});

// ---- odds ----
json("odds.json").then((rows) => {
  const sorted = [...rows].sort((a, b) => a.make_playoffs - b.make_playoffs);
  Plotly.newPlot(
    "odds-chart",
    [{
      type: "bar", orientation: "h",
      x: sorted.map((r) => r.make_playoffs),
      y: sorted.map((r) => r.team_name),
      marker: { color: sorted.map((r) => (r.league === "American League" ? AL : NL)) },
      hovertemplate: "%{y}: %{x:.1%}<extra></extra>",
    }],
    { ...PLOT_LAYOUT, height: 720, xaxis: { tickformat: ".0%", title: "P(make playoffs)  ·  teal = AL, orange = NL" } },
    { displayModeBar: false, responsive: true }
  );

  const cols = [
    ["team_name", "Team"], ["division", "Division"], ["current_wins", "W"],
    ["proj_wins", "Proj W"], ["make_playoffs", "Playoffs"], ["win_division", "Division"],
    ["first_round_bye", "Bye"], ["win_pennant", "Pennant"], ["win_world_series", "World Series"],
  ];
  const pctCols = new Set(["make_playoffs", "win_division", "first_round_bye", "win_pennant", "win_world_series"]);
  const body = [...rows]
    .sort((a, b) => b.win_world_series - a.win_world_series)
    .map((r) =>
      "<tr>" +
      cols.map(([k]) =>
        pctCols.has(k)
          ? `<td class="num pct-cell"><div class="pct-fill" style="width:${(r[k] * 100).toFixed(1)}%"></div><span>${pct(r[k])}</span></td>`
          : `<td${typeof r[k] === "number" ? ' class="num"' : ""}>${r[k]}</td>`
      ).join("") +
      "</tr>"
    ).join("");
  document.getElementById("odds-table").innerHTML =
    "<thead><tr>" + cols.map(([, h]) => `<th>${h}</th>`).join("") + "</tr></thead><tbody>" + body + "</tbody>";
});

// ---- tracker ----
json("accuracy.json").then((d) => {
  const cards = document.getElementById("tracker-cards");
  const banner = document.getElementById("tracker-banner");
  if (!d.n_scored) {
    cards.innerHTML = '<p class="empty">No scored predictions yet — check back after the first daily run.</p>';
    document.getElementById("tracker-chart").style.display = "none";
    document.getElementById("calibration-chart").style.display = "none";
    return;
  }

  // Sample-size honesty: below the reliability threshold, realized accuracy is
  // mostly noise, so say so instead of letting a small number read as signal.
  if (!d.reliable) {
    const best = d.summary.lr || d.summary.ens || Object.values(d.summary)[0];
    const halfWidth = best ? Math.round(((best.acc_ci[1] - best.acc_ci[0]) / 2) * 100) : null;
    banner.style.display = "";
    banner.innerHTML =
      `⚠️ Only <strong>${d.n_scored}</strong> games scored so far — accuracy needs ~${d.min_reliable_n} ` +
      `to be meaningful. The 95% margin on accuracy is still about ` +
      `<strong>±${halfWidth} points</strong>, so treat the numbers below as provisional and watch ` +
      `<strong>log loss vs the Elo baseline</strong> instead.`;
  } else {
    banner.style.display = "none";
  }

  cards.innerHTML = Object.entries(MODEL_LABELS)
    .filter(([k]) => d.summary[k])
    .map(([k, label]) => {
      const s = d.summary[k];
      const ci = s.acc_ci ? `${pct(s.acc_ci[0], 0)}–${pct(s.acc_ci[1], 0)}` : "–";
      const exp = s.expected_accuracy != null ? pct(s.expected_accuracy) : "–";
      return `<div class="stat-card">
        <div class="name">${label}</div>
        <div class="big">${s.log_loss}</div>
        <div class="row">log loss · ${s.n} games</div>
        <div class="row">accuracy ${pct(s.accuracy)} <span class="muted">(95% CI ${ci})</span></div>
        <div class="row">model expects ${exp} · Brier ${s.brier}</div>
      </div>`;
    })
    .join("");

  Plotly.newPlot(
    "tracker-chart",
    Object.keys(MODEL_LABELS).filter((k) => d.daily[k]).map((k) => ({
      type: "scatter", mode: "lines+markers",
      x: d.daily.dates, y: d.daily[k], name: MODEL_LABELS[k],
    })),
    { ...PLOT_LAYOUT, height: 360, margin: { l: 60, r: 20, t: 10, b: 40 },
      yaxis: { title: "Cumulative log loss (lower is better)" },
      legend: { orientation: "h", y: 1.12 } },
    { displayModeBar: false, responsive: true }
  );

  // ---- calibration / reliability curve ----
  const calChart = document.getElementById("calibration-chart");
  const calKeys = Object.keys(MODEL_LABELS).filter((k) => (d.calibration?.[k] || []).length);
  if (calKeys.length) {
    calChart.style.display = "";
    const traces = calKeys.map((k) => {
      const bins = d.calibration[k];
      return {
        type: "scatter", mode: "lines+markers",
        x: bins.map((b) => b.p_mean), y: bins.map((b) => b.win_rate),
        name: MODEL_LABELS[k],
        text: bins.map((b) => `n=${b.n}`), hovertemplate: "pred %{x:.0%} → won %{y:.0%} (%{text})<extra></extra>",
      };
    });
    traces.push({
      type: "scatter", mode: "lines", x: [0.2, 0.8], y: [0.2, 0.8],
      name: "perfect", line: { dash: "dot", color: "#888" }, hoverinfo: "skip",
    });
    Plotly.newPlot(
      "calibration-chart", traces,
      { ...PLOT_LAYOUT, height: 360, margin: { l: 60, r: 20, t: 10, b: 50 },
        xaxis: { title: "Predicted home-win probability", range: [0.15, 0.85], tickformat: ".0%" },
        yaxis: { title: "Actual home-win rate", range: [0, 1], tickformat: ".0%" },
        legend: { orientation: "h", y: 1.12 } },
      { displayModeBar: false, responsive: true }
    );
  } else {
    calChart.style.display = "none";
  }

  const modelKeys = Object.keys(MODEL_LABELS).filter((k) => d.summary[k]);
  const rows = [...d.recent].reverse().map((g) => {
    const cells = modelKeys.map((k) => {
      const p = g["p_home_" + k];
      if (p == null) return '<td class="num">–</td>';
      const hit = (p > 0.5 ? 1 : 0) === g.home_win;
      return `<td class="num ${hit ? "hit" : "miss"}">${pct(p, 0)} ${hit ? "✓" : "✗"}</td>`;
    }).join("");
    return `<tr><td>${g.date}</td><td>${g.away_name} ${g.away_score} @ ${g.home_name} ${g.home_score}</td>${cells}</tr>`;
  }).join("");
  document.getElementById("recent-table").innerHTML =
    "<thead><tr><th>Date</th><th>Result</th>" +
    modelKeys.map((k) => `<th class='num'>${MODEL_LABELS[k]} (home%)</th>`).join("") +
    "</tr></thead><tbody>" + rows + "</tbody>";
});

// ---- roster moves ----
json("transactions.json").then((rows) => {
  const table = document.getElementById("moves-table");
  const filters = document.getElementById("moves-filters");
  const categories = ["All", ...new Set(rows.map((r) => r.category))];
  let active = "All";

  const badgeClass = (c) =>
    c === "IL (injury)" ? "il" : c === "Call-up" ? "callup" : "other";

  const render = () => {
    const view = rows.filter((r) => active === "All" || r.category === active);
    table.innerHTML =
      "<thead><tr><th>Date</th><th>Team</th><th>Player</th><th>Type</th><th>Detail</th></tr></thead><tbody>" +
      (view.length
        ? view.map((r) =>
            `<tr><td>${r.date}</td><td>${r.team}</td><td>${r.player}</td>
             <td><span class="badge ${badgeClass(r.category)}">${r.category}</span></td>
             <td style="white-space:normal">${r.description}</td></tr>`
          ).join("")
        : '<tr><td colspan="5" class="empty">No moves in this category.</td></tr>') +
      "</tbody>";
  };

  filters.innerHTML = categories
    .map((c) => `<button data-cat="${c}" class="${c === "All" ? "active" : ""}">${c}</button>`)
    .join("");
  filters.querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => {
      active = b.dataset.cat;
      filters.querySelectorAll("button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      render();
    })
  );
  render();
});

// ---- meta ----
json("meta.json").then((m) => {
  document.getElementById("meta-line").textContent =
    `Last updated ${m.generated_at_utc} · ${m.n_predictions_logged} predictions logged, ` +
    `${m.n_scored} scored · best backtest model: ${m.best_model}`;
});
