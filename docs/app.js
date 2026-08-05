/* Bento panes are rendered here; the palette below mirrors the tokens in
   style.css so the Plotly figures sit inside the terminal rather than on top of
   it. Amber carries the primary series, chrome-white the secondary — the recipe's
   rule, and it keeps the league split non-semantic (green/red would read as
   up/down, which is wrong for a category). */
const REPO_URL = "https://github.com/JarvisLee511/mlb-playoff-predictor-2026";

/* Read straight out of the stylesheet's custom properties so the figures cannot
   drift from the tokens — one source of truth, not two copies of the palette. */
const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const C = {
  surface2: token("--color-surface-2"),
  hairline: token("--color-hairline"),
  ink: token("--color-ink"),
  muted: token("--color-muted"),
  accent: token("--color-accent"),
  positive: token("--color-positive"),
  negative: token("--color-negative"),
};
const AL = C.accent, NL = C.ink;
const MONO = token("--font-body");

const MODEL_LABELS = {
  elo: "Elo baseline",
  lr: "Logistic regression",
  xgb: "XGBoost",
  ens: "Elo+LR stack",
  skl: "Poisson-Skellam",
};

/* Plotly mutates the layout object it is handed — the horizontal bar chart writes
   type:"category" into its yaxis. A shared axis literal would therefore leak a
   categorical y-axis into every later chart, so each call gets a fresh one. */
const axis = (extra = {}) => ({
  gridcolor: C.hairline,
  zerolinecolor: C.hairline,
  linecolor: C.hairline,
  ...extra,
});
const layout = ({ xaxis = {}, yaxis = {}, ...rest } = {}) => ({
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: C.muted, size: 11, family: MONO },
  colorway: [C.accent, C.ink, C.positive, C.negative, C.muted],
  margin: { l: 170, r: 20, t: 8, b: 40 },
  hoverlabel: { bgcolor: C.surface2, bordercolor: C.hairline, font: { family: MONO, color: C.ink } },
  legend: { font: { size: 10 } },
  ...rest,
  xaxis: axis(xaxis),
  yaxis: axis(yaxis),
});
const axisTitle = (text) => ({ text, font: { size: 10 } });
const PLOT_CONFIG = { displayModeBar: false, responsive: true };

const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
const json = (f) => fetch("data/" + f).then((r) => r.json());

document.getElementById("repo-link").href = REPO_URL;

/* ---- views: click or press 1-5. The <kbd> chips on the command row are only
   honest if the keys work, so they do. ---- */
const keys = [...document.querySelectorAll(".cmd__key")];
function show(tab) {
  keys.forEach((k) => k.setAttribute("aria-selected", String(k.dataset.tab === tab)));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("is-active", p.id === tab));
  // Plotly sizes to a hidden container as 0-width; nudge it once the pane is visible.
  document.querySelectorAll(`#${tab} .chart`).forEach((el) => {
    if (el.querySelector(".main-svg")) Plotly.Plots.resize(el);
  });
}
keys.forEach((k) => k.addEventListener("click", () => show(k.dataset.tab)));
document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const i = Number(e.key) - 1;
  if (Number.isInteger(i) && keys[i]) {
    show(keys[i].dataset.tab);
    keys[i].focus();
  }
});

/* ---- status strip ---- */
const readout = (label, value, tone) =>
  `<div class="readout"><dt>${label}</dt><dd${tone ? ` class="is-${tone}"` : ""}>${value}</dd></div>`;

Promise.all([json("meta.json"), json("accuracy.json"), json("odds.json")])
  .then(([meta, acc, odds]) => {
    const parts = [];
    const fav = [...odds].sort((a, b) => b.win_world_series - a.win_world_series)[0];
    if (fav) parts.push(readout("WS favourite", `${fav.abbrev || fav.team_name} ${pct(fav.win_world_series)}`, "accent"));

    const h2h = acc.head_to_head;
    if (h2h && h2h.models) {
      const best = Object.entries(h2h.models)
        .sort((a, b) => a[1].log_loss - b[1].log_loss)[0];
      parts.push(readout("Best model", MODEL_LABELS[best[0]] || best[0]));
      const d = best[1].delta_vs_elo;
      if (d != null) parts.push(readout("Δ vs Elo", d.toFixed(4), d < 0 ? "pos" : "neg"));
      parts.push(readout("Scored", `${h2h.n} games`));
    } else {
      parts.push(readout("Scored", `${acc.n_scored} games`));
    }
    parts.push(readout("Updated", meta.generated_at_utc));
    document.getElementById("status-readout").innerHTML = parts.join("");
  })
  .catch(() => {
    document.getElementById("status-readout").innerHTML = readout("Status", "data unavailable", "neg");
  });

/* ---- today + tomorrow ---- */
const gameCell = (g) => {
  const p = g.p_home_ens != null ? g.p_home_ens : g.p_home_lr;
  const probs = [
    ["Stack", g.p_home_ens], ["Elo", g.p_home_elo], ["LogReg", g.p_home_lr],
    ["XGB", g.p_home_xgb], ["Skellam", g.p_home_skl],
  ].filter(([, v]) => v != null).map(([n, v]) => `${n} ${pct(v, 0)}`).join(" · ");
  return `<article class="cell">
    <div class="game__time">${g.game_time_et} ET</div>
    <div class="game__matchup">${g.away_name} @ ${g.home_name}</div>
    <div class="game__pitchers">${g.away_pitcher} vs ${g.home_pitcher}</div>
    <div class="probbar" role="img" aria-label="Home win probability ${pct(p, 0)}">
      <div class="probbar__home" style="width:${(p * 100).toFixed(1)}%">${pct(p, 0)}</div>
      <div class="probbar__away" style="width:${((1 - p) * 100).toFixed(1)}%">${pct(1 - p, 0)}</div>
    </div>
    <div class="game__models">Home win — ${probs}</div>
  </article>`;
};

json("today.json").then((d) => {
  document.getElementById("today-title").textContent = `Predictions for ${d.date}`;
  document.getElementById("today-games").innerHTML = d.games.length
    ? d.games.map(gameCell).join("")
    : '<div class="cell cell--w4"><p class="empty">No regular-season games scheduled today.</p></div>';

  if (d.tomorrow && d.tomorrow.length) {
    document.getElementById("tomorrow-head").hidden = false;
    document.getElementById("tomorrow-title").textContent = `Tomorrow — ${d.tomorrow_date}`;
    document.getElementById("tomorrow-games").innerHTML = d.tomorrow.map(gameCell).join("");
  }
});

/* ---- odds ---- */
json("odds.json").then((rows) => {
  const sorted = [...rows].sort((a, b) => a.make_playoffs - b.make_playoffs);
  Plotly.newPlot("odds-chart", [{
    type: "bar", orientation: "h",
    x: sorted.map((r) => r.make_playoffs),
    y: sorted.map((r) => r.team_name),
    marker: { color: sorted.map((r) => (r.league === "American League" ? AL : NL)) },
    hovertemplate: "%{y}: %{x:.1%}<extra></extra>",
  }], layout({
    height: 720,
    xaxis: { tickformat: ".0%", title: axisTitle("P(make playoffs)") },
  }), PLOT_CONFIG);

  const cols = [
    ["team_name", "Team"], ["division", "Division"], ["current_wins", "W"],
    ["proj_wins", "Proj W"], ["make_playoffs", "Playoffs"], ["win_division", "Div"],
    ["first_round_bye", "Bye"], ["win_pennant", "Pennant"], ["win_world_series", "World Series"],
  ];
  const pctCols = new Set(["make_playoffs", "win_division", "first_round_bye", "win_pennant", "win_world_series"]);
  const body = [...rows]
    .sort((a, b) => b.win_world_series - a.win_world_series)
    .map((r) => "<tr>" + cols.map(([k]) =>
      pctCols.has(k)
        ? `<td class="num pct-cell"><div class="pct-fill" style="width:${(r[k] * 100).toFixed(1)}%"></div><span>${pct(r[k])}</span></td>`
        : `<td${typeof r[k] === "number" ? ' class="num"' : ""}>${r[k]}</td>`
    ).join("") + "</tr>").join("");
  document.getElementById("odds-table").innerHTML =
    "<thead><tr>" + cols.map(([, h]) => `<th>${h}</th>`).join("") + "</tr></thead><tbody>" + body + "</tbody>";
});

/* ---- tracker ---- */
json("accuracy.json").then((d) => {
  const cards = document.getElementById("tracker-cards");
  const notice = document.getElementById("tracker-notice");

  if (!d.n_scored) {
    cards.innerHTML = '<div class="cell cell--w4"><p class="empty">No scored predictions yet — check back after the first daily run.</p></div>';
    ["tracker-chart", "calibration-chart"].forEach((id) => { document.getElementById(id).style.display = "none"; });
    return;
  }

  // Sample-size honesty: below the reliability threshold realized accuracy is
  // mostly noise, so say so rather than letting a small number read as signal.
  if (!d.reliable) {
    const best = d.summary.ens || d.summary.lr || Object.values(d.summary)[0];
    const half = best ? Math.round(((best.acc_ci[1] - best.acc_ci[0]) / 2) * 100) : null;
    notice.hidden = false;
    notice.innerHTML = `<p class="notice"><span class="notice__tag">Small sample</span>
      Only <strong>${d.n_scored}</strong> games scored so far — accuracy needs about
      ${d.min_reliable_n} to mean anything. The 95% margin is still ±${half} points, so read
      log loss against the Elo baseline instead.</p>`;
  } else if (d.baselines && Object.keys(d.summary).length) {
    // Past the threshold the slot carries the yardsticks instead. Without them a
    // mid-fifties accuracy reads as a broken model; against the floor it is the
    // sport being close to a coin flip. Every number here comes from the export,
    // so the sentence stays true as the sample grows.
    const b = d.baselines;
    const [bk, bs] = Object.entries(d.summary).sort((x, y) => x[1].log_loss - y[1].log_loss)[0];
    const span = b.market_ceiling_accuracy - b.always_home_accuracy;
    const closed = span > 0 ? Math.round(((bs.accuracy - b.always_home_accuracy) / span) * 100) : null;
    const place = closed == null ? ""
      : closed >= 0 ? ` — about <strong>${closed}%</strong> of the way from that floor to the market`
      : " — <strong>below</strong> that floor on this sample";
    const ll = bs.log_loss < b.coinflip_log_loss
      ? `On log loss it does beat a coin flip (${bs.log_loss} vs ${b.coinflip_log_loss}), and the
         gap left to ${b.market_ceiling_log_loss} is what private information — injuries, confirmed
         lineups, weather, money flow — buys that public box scores cannot.`
      : `On log loss it is not yet separable from a coin flip (${bs.log_loss} vs
         ${b.coinflip_log_loss}).`;
    notice.hidden = false;
    // One decimal on all three, deliberately: rounded to whole percent the reader
    // recomputes the "% of the way" figure from 51/54/57 and gets a different
    // answer than the exact one quoted.
    notice.innerHTML = `<p class="notice notice--context"><span class="notice__tag">Reading the numbers</span>
      One MLB game is nearly a coin flip, so the ceiling for any public-data model is the closing
      line (~<strong>${pct(b.market_ceiling_accuracy)}</strong>) and the floor is "always pick the
      home team" (<strong>${pct(b.always_home_accuracy)}</strong> across these ${d.n_scored}
      games). ${MODEL_LABELS[bk]} (best of the ${Object.keys(d.summary).length} on log loss, the
      metric that actually separates them) sits at <strong>${pct(bs.accuracy)}</strong>${place}. ${ll}</p>`;
  }

  cards.innerHTML = Object.entries(MODEL_LABELS)
    .filter(([k]) => d.summary[k])
    .map(([k, label]) => {
      const s = d.summary[k];
      const ci = s.acc_ci ? `${pct(s.acc_ci[0], 0)}–${pct(s.acc_ci[1], 0)}` : "–";
      const exp = s.expected_accuracy != null ? pct(s.expected_accuracy) : "–";
      return `<article class="cell">
        <div class="stat__name">${label}</div>
        <div class="stat__figure">${s.log_loss}</div>
        <div class="stat__row">log loss · ${s.n} games</div>
        <div class="stat__row">accuracy ${pct(s.accuracy)} <span class="muted">(95% CI ${ci})</span></div>
        <div class="stat__row">expects ${exp} · Brier ${s.brier}</div>
      </article>`;
    }).join("");

  /* Head to head. The stat panes above score each model over its own non-null
     rows, which are different samples; this restricts to the intersection and
     puts a paired bootstrap interval on the gap to Elo. */
  const h2h = d.head_to_head;
  const h2hEl = document.getElementById("h2h");
  if (h2h && h2h.models) {
    const rows = Object.entries(MODEL_LABELS)
      .filter(([k]) => h2h.models[k])
      .map(([k, label]) => {
        const m = h2h.models[k];
        const head = `<td>${label}</td><td class="num">${m.log_loss}</td>` +
          `<td class="num">${m.brier}</td><td class="num">${pct(m.accuracy)}</td>`;
        if (k === "elo") return `<tr>${head}<td class="num">–</td><td class="num">–</td></tr>`;
        const cls = m.delta_vs_elo < 0 ? "hit" : "";
        return `<tr>${head}<td class="num ${cls}">${m.delta_vs_elo.toFixed(4)}</td>` +
          `<td class="num muted">[${m.delta_ci[0].toFixed(4)}, ${m.delta_ci[1].toFixed(4)}]</td></tr>`;
      }).join("");

    // Read the verdict out of the intervals rather than hardcoding it, so the
    // sentence stays true as the sample grows.
    const challengers = Object.entries(h2h.models).filter(([k]) => k !== "elo");
    const clearZero = challengers.filter(([, m]) => m.delta_ci[1] < 0).map(([k]) => MODEL_LABELS[k]);
    const verdict = clearZero.length
      ? `<strong>${clearZero.join(", ")}</strong> ${clearZero.length > 1 ? "clear" : "clears"} zero, so ` +
        `${clearZero.length > 1 ? "those models beat" : "that model beats"} the Elo baseline by more than noise.`
      : `Every interval spans zero, so on this sample the ML features are <strong>not
         distinguishable from the Elo baseline</strong> — the honest read, not a bug.`;

    h2hEl.innerHTML =
      `<div class="table-wrap"><table><thead><tr><th>Model</th><th class="num">Log loss</th>` +
      `<th class="num">Brier</th><th class="num">Accuracy</th><th class="num">Δ vs Elo</th>` +
      `<th class="num">95% CI (paired bootstrap)</th></tr></thead><tbody>${rows}</tbody></table></div>` +
      `<p class="hint hint--inset">All ${h2h.n} games that every
       model predicted. Negative Δ beats Elo. ${verdict} MLB games sit near a coin flip — home teams
       won ${pct(h2h.home_win_rate)} of these — and Elo already captures most of what pre-game team
       strength can tell you.</p>`;
  } else if (h2hEl) {
    h2hEl.innerHTML = '<p class="empty">Not enough overlapping predictions yet.</p>';
  }

  Plotly.newPlot("tracker-chart",
    Object.keys(MODEL_LABELS).filter((k) => d.daily[k]).map((k) => ({
      type: "scatter", mode: "lines", x: d.daily.dates, y: d.daily[k], name: MODEL_LABELS[k],
    })), layout({
      height: 320, margin: { l: 52, r: 16, t: 8, b: 36 },
      yaxis: { title: axisTitle("cumulative log loss") },
      legend: { font: { size: 9 }, orientation: "h", y: 1.16 },
    }), PLOT_CONFIG);

  /* ---- calibration / reliability curve ---- */
  const calChart = document.getElementById("calibration-chart");
  const calKeys = Object.keys(MODEL_LABELS).filter((k) => (d.calibration?.[k] || []).length);
  if (calKeys.length) {
    const traces = calKeys.map((k) => {
      const bins = d.calibration[k];
      return {
        type: "scatter", mode: "lines+markers",
        x: bins.map((b) => b.p_mean), y: bins.map((b) => b.win_rate), name: MODEL_LABELS[k],
        text: bins.map((b) => `n=${b.n}`),
        hovertemplate: "predicted %{x:.0%} → won %{y:.0%} (%{text})<extra></extra>",
      };
    });
    traces.push({
      type: "scatter", mode: "lines", x: [0.2, 0.8], y: [0.2, 0.8], name: "perfect",
      line: { dash: "dot", color: C.muted, width: 1 }, hoverinfo: "skip",
    });
    Plotly.newPlot("calibration-chart", traces, layout({
      height: 320, margin: { l: 52, r: 16, t: 8, b: 44 },
      xaxis: { title: axisTitle("predicted"), range: [0.15, 0.85], tickformat: ".0%" },
      yaxis: { title: axisTitle("actual"), range: [0, 1], tickformat: ".0%" },
      legend: { font: { size: 9 }, orientation: "h", y: 1.16 },
    }), PLOT_CONFIG);
  } else {
    calChart.style.display = "none";
  }

  const modelKeys = Object.keys(MODEL_LABELS).filter((k) => d.summary[k]);
  const recent = [...d.recent].reverse().map((g) => {
    const cells = modelKeys.map((k) => {
      const p = g["p_home_" + k];
      if (p == null) return '<td class="num">–</td>';
      const hit = (p > 0.5 ? 1 : 0) === g.home_win;
      return `<td class="num ${hit ? "hit" : "miss"}">${pct(p, 0)}</td>`;
    }).join("");
    return `<tr><td>${g.date}</td><td>${g.away_name} ${g.away_score} @ ${g.home_name} ${g.home_score}</td>${cells}</tr>`;
  }).join("");
  document.getElementById("recent-table").innerHTML =
    "<thead><tr><th>Date</th><th>Result</th>" +
    modelKeys.map((k) => `<th class="num">${MODEL_LABELS[k]}</th>`).join("") +
    '</tr></thead><tbody>' + recent + "</tbody>";
});

/* ---- roster moves ---- */
json("transactions.json").then((rows) => {
  const table = document.getElementById("moves-table");
  const filters = document.getElementById("moves-filters");
  const categories = ["All", ...new Set(rows.map((r) => r.category))];
  let active = "All";

  const badge = (c) => (c === "IL (injury)" ? "il" : c === "Call-up" ? "callup" : "other");

  const render = () => {
    const view = rows.filter((r) => active === "All" || r.category === active);
    table.innerHTML =
      "<thead><tr><th>Date</th><th>Team</th><th>Player</th><th>Type</th><th>Detail</th></tr></thead><tbody>" +
      (view.length
        ? view.map((r) =>
            `<tr><td>${r.date}</td><td>${r.team}</td><td>${r.player}</td>` +
            `<td><span class="badge badge--${badge(r.category)}">${r.category}</span></td>` +
            `<td style="white-space:normal">${r.description}</td></tr>`).join("")
        : '<tr><td colspan="5" class="empty">No moves in this category.</td></tr>') +
      "</tbody>";
  };

  filters.innerHTML = categories
    .map((c) => `<button type="button" data-cat="${c}" aria-pressed="${c === "All"}">${c}</button>`)
    .join("");
  filters.querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => {
      active = b.dataset.cat;
      filters.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", "false"));
      b.setAttribute("aria-pressed", "true");
      render();
    })
  );
  render();
});
