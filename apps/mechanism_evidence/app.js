"use strict";

const snapshot = JSON.parse(document.getElementById("evidence-data").textContent);
const state = { step: "1", model: "all" };
const steps = {
  "1": {
    title: "1 Attention creates a BOS component",
    prediction: "First-layer attention mixes a position-dependent BOS component with content. Its BOS weight changes with position.",
    path: ["BOS + content", "Attention", "Mixed component"],
    notes: [
      "Observed: the BOS weight is the mean over first-layer heads. The position axis is log₂(position + 1), so it starts at BOS, position 0.",
      "The mixed component also carries content, not only position.",
      "Causal: the saved artifacts contain no isolated test of this step. This is a gap, not a zero effect."
    ],
    edit: null
  
  },
  "2": {
    title: "2 The MLP changes the component",
    prediction: "The MLP corrects the projected component. Lower absolute spread can support stabilization, but does not alone establish signal-relative denoising.",
    path: ["Mixed component", "MLP", "Corrected component"],
    notes: [
      "Observed: the spread is the mean per-position SD of the projection across sequences, from position 64 to the end of the evidence context.",
      "The correction slopes are a separate measurement at fixed positions.",
    ],
    edit: "Flatten the MLP write along the BOS direction",
    experiment: [
      "Edit (mlp_mean): take the first-layer MLP output at every ordinary token, after BOS and any attention sink. Replace its component along u1 with the mean of that component over ordinary tokens. u1 is the unit direction of BOS's state after first-layer attention.",
      "This keeps the average MLP write along u1 and removes its token-by-token part. Everything after the MLP is then recomputed.",
      "Random control (mlp_mean_rand): the same edit along a random unit direction."
    ]
  },
  "3": {
    title: "3 Keys turn the share into attention",
    prediction: "Second-layer keys read the measured share. A graded share predicts attention, and a key-only shift tests its fitted coefficient.",
    path: ["Measured share", "Keys", "Attention weights"],
    notes: [
      "Observed: fitted and measured key coefficient B for a key-only shift of 0.2.",
      "Mean TV distance compares the graded attention prediction with measured attention. Lower is closer.",
      "For 12H, the numbers are ranges across the six selected heads, not confidence intervals.",
    ],
    edit: "Remove the BOS share from second-layer keys",
    experiment: [
      "Edit (keys_noshare): at the second-layer input, take d = x_BOS − μ, where μ is the mean input of ordinary tokens. Each token's share is its projection on d.",
      "Remove that projection from the key input of every token, for all 12 second-layer heads. Queries and values keep the clean input.",
      "Random control (keys_noshare_rand): remove the projection on a random direction with the same norm as d."
    ]
  },
  "4": {
    title: "4 Values expose a position readout",
    prediction: "Predict both attention weights and value projections from the measured share. Together, they predict the projected position readout.",
    path: ["Share + attention", "Values", "Projected readout"],
    notes: [
      "Observed: the value/share fit, and the graded readout fit that predicts both attention and values from the measured share.",
      "R² is 1 − Var(residual) / Var(measured), not mean squared error. A graded share is valid. Two levels are a special case.",
      "For 12H, the numbers are ranges across the six selected heads, not confidence intervals.",
    ],
    edit: "Remove the BOS share from second-layer values",
    experiment: [
      "Edit (values_noshare): at the second-layer input, take d = x_BOS − μ, where μ is the mean input of ordinary tokens. Each token's share is its projection on d.",
      "Remove that projection from the value input of every token, for all 12 second-layer heads. Queries and keys keep the clean input.",
      "Random control (values_noshare_rand): remove the projection on a random direction with the same norm as d."
    ]
  }
};
const causalNote = [
  "Baseline: the unedited model.",
  "Readout: position MAE in tokens at positions 16 through 127, lower is better. 1H and 12H use their frozen trained position heads. LM-BOS uses a frozen diagnostic position probe and also reports the increase in next-token loss.",
  "These tests show that the model uses this direction. They do not uniquely prove the full mechanism."
];

function visibleModels(data) {
  return data.models.filter(model => state.model === "all" || model.id === state.model);
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function svgElement(tag, attributes = {}, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}

function svgChart(label, height, className) {
  const svg = svgElement("svg", { viewBox: `0 0 344 ${height}`, role: "img", "aria-label": label, class: className });
  svg.append(svgElement("title", {}, label));
  return svg;
}

function svgText(svg, x, y, text, anchor = "middle") {
  svg.append(svgElement("text", { x, y, "text-anchor": anchor }, text));
}

function line(svg, x1, y1, x2, y2, className) {
  svg.append(svgElement("line", { x1, y1, x2, y2, class: className }));
}

function figure(svg, caption) {
  const fig = element("figure");
  fig.append(svg, element("figcaption", caption));
  return fig;
}

function format(value, digits = 3) {
  if (value > 0 && value < 10 ** -digits) return `<${10 ** -digits}`;
  return value.toFixed(digits).replace(/(\.\d*?[1-9])0+$|\.0+$/, "$1");
}

function rangeText(values) {
  return values[0] === values[1] ? format(values[0]) : `${format(values[0])} to ${format(values[1])}`;
}

function metrics(entries) {
  const list = element("dl", undefined, "metrics");
  for (const [label, value] of entries) {
    const row = element("div");
    row.append(element("dt", label), element("dd", value));
    list.append(row);
  }
  return list;
}

function attentionChart(observation) {
  const points = observation.attention_points;
  const chart = { left: 42, top: 16, width: 260, height: 140 };
  const maxPosition = Math.max(...points.map(point => point.position));
  // log2(position + 1) keeps position 0 on the axis, at the left edge.
  const plotX = position => chart.left + chart.width * Math.log2(position + 1) / Math.log2(maxPosition + 1);
  const plotY = weight => chart.top + chart.height * (1 - weight);
  const svg = svgChart("BOS attention probability against position, from position 0, on a log2(position + 1) axis", 220, "attention-chart");
  for (const weight of [0, 0.5, 1]) {
    line(svg, 42, plotY(weight), 302, plotY(weight), "grid");
    svgText(svg, 34, plotY(weight) + 4, format(weight), "end");
  }
  line(svg, 42, 16, 42, 156, "axis");
  line(svg, 42, 156, 302, 156, "axis");
  const ticks = [0, 1, 4, 16, 64, 256].filter(position => position < maxPosition && plotX(maxPosition) - plotX(position) > 32);
  ticks.push(maxPosition);
  for (const position of ticks) {
    line(svg, plotX(position), 156, plotX(position), 161, "axis");
    svgText(svg, plotX(position), 177, String(position));
  }
  svgText(svg, 172, 198, "Position (log₂ of position + 1)");
  const axisLabel = svgElement("text", { transform: "translate(12 86) rotate(-90)", "text-anchor": "middle" }, "BOS probability");
  svg.append(axisLabel);
  const path = points.map((point, index) => `${index === 0 ? "M" : "L"}${plotX(point.position)},${plotY(point.weight)}`).join(" ");
  svg.append(svgElement("path", { d: path, class: "attention-line" }));
  for (const point of points) svg.append(svgElement("circle", { cx: plotX(point.position), cy: plotY(point.weight), r: 2.5, class: "observed-mark" }));
  return figure(svg, "Starts at BOS, position 0, where the weight is 1.");
}

function horizontalBars(entries, axisLabel, className, logarithmic = false) {
  const values = entries.map(entry => entry.value);
  const max = Math.max(...values);
  // Log charts start below the smallest positive error, not at an implied zero.
  const low = logarithmic ? 10 ** Math.floor(Math.log10(Math.min(...values))) / 10 : 0;
  const high = logarithmic ? 10 ** Math.ceil(Math.log10(max)) : (max || 1) * 1.12;
  const plot = value => logarithmic
    ? 92 + 230 * (Math.log10(value) - Math.log10(low)) / (Math.log10(high) - Math.log10(low))
    : 92 + 230 * value / high;
  const bottom = 28 + entries.length * 32;
  const svg = svgChart(axisLabel, bottom + 56, className);
  const ticks = logarithmic
    ? Array.from({ length: Math.round(Math.log10(high / low)) + 1 }, (_, i) => low * 10 ** i)
    : [0, high / 2, high];
  for (const tick of ticks) {
    const x = plot(tick);
    line(svg, x, 16, x, bottom, "grid");
    svgText(svg, x, bottom + 19, format(tick));
  }
  entries.forEach((entry, index) => {
    const y = 22 + index * 32;
    svgText(svg, 82, y + 14, entry.label, "end");
    svg.append(svgElement("rect", { x: 92, y, width: plot(entry.value) - 92, height: 20, class: `bar ${entry.className}` }));
  });
  line(svg, 92, bottom, 322, bottom, "axis");
  svgText(svg, 207, bottom + 42, axisLabel);
  return svg;
}

function headChart(observation, fields, axisLabel, fixedDomain) {
  const values = observation.head_metrics.flatMap(head => fields.map(field => head[field.key]));
  const low = Math.min(0, ...values);
  const high = fixedDomain ? Math.max(1, ...values) : Math.max(...values, 1) * 1.1;
  const plot = value => 48 + 260 * (value - low) / (high - low);
  const bottom = 42 + observation.head_metrics.length * 27;
  const svg = svgChart(axisLabel, bottom + 54, "head-chart");
  // Shape and text distinguish the two measurements without color.
  svg.append(svgElement("circle", { cx: 52, cy: 13, r: 4, class: "observed-mark" }));
  svgText(svg, 62, 17, fields[0].label, "start");
  svg.append(svgElement("rect", { x: 181, y: 9, width: 8, height: 8, class: "alternate-mark" }));
  svgText(svg, 195, 17, fields[1].label, "start");
  for (const tick of [low, (low + high) / 2, high]) {
    line(svg, plot(tick), 30, plot(tick), bottom, "grid");
    svgText(svg, plot(tick), bottom + 19, format(tick));
  }
  observation.head_metrics.forEach((head, index) => {
    const y = 42 + index * 27;
    svgText(svg, 34, y + 4, `h${head.head}`, "end");
    line(svg, plot(head[fields[0].key]), y - 3, plot(head[fields[1].key]), y + 3, "axis");
    svg.append(svgElement("circle", { cx: plot(head[fields[0].key]), cy: y - 3, r: 4, class: "observed-mark" }));
    svg.append(svgElement("rect", { x: plot(head[fields[1].key]) - 4, y: y - 1, width: 8, height: 8, class: "alternate-mark" }));
  });
  line(svg, 48, bottom, 308, bottom, "axis");
  svgText(svg, 178, bottom + 42, axisLabel);
  return svg;
}

function slopeTable(slopes) {
  const table = element("table", undefined, "slopes");
  table.append(element("caption", "Correction slopes, separate from SD"));
  const header = element("tr");
  for (const label of ["Position", "Slope", "R²"]) {
    const cell = element("th", label);
    cell.scope = "col";
    header.append(cell);
  }
  const head = element("thead");
  head.append(header);
  const body = element("tbody");
  for (const slope of slopes) {
    const row = element("tr");
    for (const value of [slope.j, format(slope.slope_correction_on_pre), format(slope.r2)]) row.append(element("td", String(value)));
    body.append(row);
  }
  table.append(head, body);
  return table;
}

function observedPanel(model) {
  const observation = model.observations[state.step];
  const panel = element("section", undefined, "evidence observed");
  panel.append(element("h4", "Observed evidence"));
  if (state.step === "1") {
    panel.append(element("p", "BOS attention, mean over first-layer heads."), attentionChart(observation));
    const points = observation.attention_points;
    panel.append(metrics([points[0], points[1], points[2], points[points.length - 1]].map(point => [`Position ${point.position}`, format(point.weight)])));
  } else if (state.step === "2") {
    panel.append(element("p", `Absolute projection SD, positions ${observation.spread_positions[0]} through ${observation.spread_positions[1]}.`));
    panel.append(figure(horizontalBars([
      { label: "Before MLP", value: observation.sd_before, className: "observed-mark" },
      { label: "After MLP", value: observation.sd_after, className: "alternate-mark" }
    ], "Projection SD (absolute units)", "spread-chart"), observation.spread_definition));
    panel.append(metrics([["Before MLP", format(observation.sd_before)], ["After MLP", format(observation.sd_after)]]));
    panel.append(element("p", observation.interpretation, model.id === "lmbos_s9" ? "status-note" : "note"));
    panel.append(slopeTable(observation.slopes));
  } else if (state.step === "3") {
    panel.append(element("p", `Key-only shift, delta = ${format(observation.key_shift_delta)}.`));
    panel.append(figure(headChart(observation, [
      { key: "B_fitted", label: "Fitted B" }, { key: "B_measured", label: "Measured B" }
    ], "Key coefficient B (unitless)", false), `Query positions 1 through ${observation.query_positions[1]}.`));
    panel.append(metrics([
      ["Fitted B", rangeText(observation.ranges.B_fitted)],
      ["Measured B", rangeText(observation.ranges.B_measured)],
      ["Attention distance (mean TV)", rangeText(observation.ranges.attention_tv)]
    ]));
  } else {
    panel.append(figure(headChart(observation, [
      { key: "value_r2", label: "Value/share" }, { key: "graded_readout_r2", label: "Graded readout" }
    ], "Fit R² (residual variance)", true), `Queries 1 through ${observation.query_positions[1]}. Value keys 0 through ${observation.value_key_positions[1]}, calibrated without key 0.`));
    panel.append(metrics([["Value/share R²", rangeText(observation.ranges.value_r2)], ["Graded readout R²", rangeText(observation.ranges.graded_readout_r2)]]));
  }
  return panel;
}

const armLabels = { baseline: "Baseline", targeted: "Targeted edit", random: "Random control" };

function causalPanel(model) {
  const causal = model.causal[state.step];
  const panel = element("section", undefined, "evidence causal");
  panel.append(element("h4", "Causal evidence"));
  if (causal.status === "not_tested") {
    panel.append(element("p", "No isolated causal test", "status-note"));
    return panel;
  }
  const output = causal.position_output.replaceAll("_", " ");
  const where = causal.edited_positions === "all"
    ? "at every position"
    : `from position ${causal.first_ordinary_position} on`;
  panel.append(element("p", `Edit: ${steps[state.step].edit[0].toLowerCase() + steps[state.step].edit.slice(1)}, ${where}.`, "edit-line"));
  panel.append(element("p", `Position MAE from a ${output}.`));
  const entries = Object.keys(armLabels).map(role => ({ label: armLabels[role], value: causal[role].mae_middle, className: role }));
  const values = entries.map(entry => entry.value);
  const logarithmic = Math.min(...values) > 0 && Math.max(...values) / Math.min(...values) > 20;
  // Short chart labels keep the complete control labels readable in the numeric summary.
  const chartEntries = entries.map((entry, index) => ({ ...entry, label: ["Baseline", "Targeted", "Random"][index] }));
  panel.append(figure(horizontalBars(chartEntries, logarithmic ? "Position MAE (tokens, log scale)" : "Position MAE (tokens)", "causal-chart", logarithmic), `Query positions ${causal.positions[0]} through ${causal.positions[1]}.`));
  panel.append(metrics(entries.map(entry => [entry.label, `${entry.value.toFixed(2)} tokens`])));
  if (causal.baseline.loss_increase !== null) {
    const loss = element("div", undefined, "loss-panel");
    loss.append(element("h4", "Next-token loss increase"));
    loss.append(metrics(Object.keys(armLabels).map(role => {
      const value = causal[role].loss_increase;
      return [armLabels[role], value > 0 && value < 0.001 ? "<0.001" : value.toFixed(3)];
    })));
    panel.append(loss);
  }
  return panel;
}

function jsonDetails(label, document) {
  const details = element("details");
  details.append(element("summary", label), element("pre", JSON.stringify(document, null, 2)));
  return details;
}

function sourceDetails(model) {
  const details = element("details", undefined, "source-details");
  details.append(element("summary", "Measurements and source records"));
  details.append(element("p", `Iteration ${model.iteration}. Training context ${model.trained_context_length}. Evidence context ${model.evidence_context_length}. Softmax T_used ${model.softmax_T_used}.`));
  details.append(element("p", model.checkpoint_path, "source-path"));
  details.append(element("p", `Selected second-layer heads: ${model.heads.join(", ")}. Recorded artifact heads: ${model.recorded_heads.join(", ")}.`));
  details.append(element("p", `Checkpoint verification: ${model.checkpoint_verification.source}.`));
  details.append(jsonDetails("Observed measurements and exact JSON field references", model.observations[state.step]));
  details.append(jsonDetails("Causal arms and exact JSON field references", model.causal[state.step]));
  details.append(jsonDetails("Context source references", model.source_refs));
  for (const [name, source] of Object.entries(model.sources)) {
    const record = element("details");
    record.append(element("summary", `${name} source`));
    record.append(element("p", source.path, "source-path"), element("p", `SHA-256 ${source.sha256}`, "source-hash"));
    record.append(element("pre", JSON.stringify(source.document, null, 2)));
    details.append(record);
  }
  return details;
}

function modelCard(model) {
  const card = element("article", undefined, "model-card");
  const header = element("header", undefined, "card-header");
  header.append(element("h3", `${model.name}, seed ${model.seed}`));
  header.append(element("p", `Iteration ${model.iteration} · ${model.heads.length === 1 ? `Second-layer head ${model.heads[0]}` : `${model.heads.length} BOS-focused second-layer heads`}`, "note"));
  card.append(header, observedPanel(model), causalPanel(model), sourceDetails(model));
  return card;
}

function render() {
  const step = steps[state.step];
  document.getElementById("step-title").textContent = step.title;
  document.getElementById("prediction").textContent = step.prediction;
  const illustration = document.getElementById("illustration");
  illustration.replaceChildren();
  step.path.forEach((label, index) => {
    if (index) illustration.append(element("b", "→"));
    illustration.append(element("span", label));
  });
  const notes = document.getElementById("step-notes");
  notes.replaceChildren(element("h3", "How to read this step"));
  const list = element("ul");
  for (const note of step.notes) list.append(element("li", note));
  notes.append(list);
  if (step.experiment) {
    notes.append(element("h3", "Causal experiment"));
    const experiment = element("ul");
    for (const note of [...step.experiment, ...causalNote]) experiment.append(element("li", note));
    notes.append(experiment);
  }
  for (const tab of document.querySelectorAll("[data-step]")) {
    const selected = tab.dataset.step === state.step;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  }
  for (const button of document.querySelectorAll("[data-model]")) button.setAttribute("aria-pressed", String(button.dataset.model === state.model));
  const models = visibleModels(snapshot);
  const panel = document.getElementById("evidence-panel");
  panel.classList.toggle("single", models.length === 1);
  panel.setAttribute("aria-labelledby", `tab-${state.step}`);
  panel.replaceChildren(...models.map(modelCard));
  document.getElementById("selection-status").textContent = `${step.title}. Showing ${models.length} ${models.length === 1 ? "model" : "models"}.`;
}

const tabs = [...document.querySelectorAll("[data-step]")];
for (const [index, tab] of tabs.entries()) {
  tab.addEventListener("click", () => { state.step = tab.dataset.step; render(); });
  tab.addEventListener("keydown", event => {
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    else if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = tabs.length - 1;
    else return;
    event.preventDefault();
    state.step = tabs[next].dataset.step;
    render();
    tabs[next].focus();
  });
}
// One filter button per model in the data, after "All models".
const filter = document.querySelector(".model-filter");
for (const model of snapshot.models) {
  const button = element("button", `${model.name}, seed ${model.seed}`);
  button.dataset.model = model.id;
  button.setAttribute("aria-pressed", "false");
  filter.append(button);
}
for (const button of document.querySelectorAll("[data-model]")) {
  button.addEventListener("click", () => { state.model = button.dataset.model; render(); });
}

const protocol = document.getElementById("protocol");
for (const key of ["selected_examples", "graded_share", "spread_note", "count_note", "flat_share_note"]) protocol.append(element("p", snapshot.protocol[key]));
for (const [name, count] of Object.entries(snapshot.protocol.counts)) {
  protocol.append(element("p", `${name}: ${count.count} sequences, verified from ${count.constant} at script line ${count.line}.`));
  const record = jsonDetails(`${name} count source`, count.source);
  protocol.append(record);
}
protocol.append(jsonDetails("Metric definitions", snapshot.protocol.metric_definitions));
render();
