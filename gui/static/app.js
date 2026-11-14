"use strict";
// Bridge XAI front end over gui/server.py's JSON API: the game in the middle (the table, a model
// chooser at every seat, the auction, the bidding box) and, beside it, only what the system
// produces for the selected call (Why, Weights, Cards, Ask, Models).

const SEATS = ["N", "E", "S", "W"];
const PAD_ORDER = ["W", "N", "E", "S"];  // as a bridge column prints the auction
const SEAT_NAME = { N: "North", E: "East", S: "South", W: "West" };
const SUIT_SYM = { S: "♠", H: "♥", D: "♦", C: "♣" };
const STRAINS = ["C", "D", "H", "S", "NT"];
const $ = (id) => document.getElementById(id);

let state = { id: null, game: null, selected: null, recIndex: null, record: null, models: [], dialogs: {}, meanings: {},
              version: 0, llm: null, explore: false, explainBy: null, tab: "why",
              vs: null, openConcept: null, allBids: false, allConcepts: false };  // the Weights tab

async function api(method, path, body) {
  const r = await fetch(path, { method, headers: { "Content-Type": "application/json" },
                                body: body ? JSON.stringify(body) : undefined });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}
function esc(s) { return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function status(msg) { $("status").textContent = msg || ""; }
function pct(p) { return p >= 0.995 ? "100%" : p < 0.005 ? "<1%" : `${Math.round(p * 100)}%`; }
function plainCall(c) { return c === "P" ? "Pass" : c === "X" ? "Double" : c === "XX" ? "Redouble" : c; }
function cap(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }
function modelLabel(id) { const m = state.models.find((x) => x.id === id); return m ? m.short : id; }

// ------------------------------------------------------------------ calls and cards
function callInner(c) {
  if (c === "P") return "Pass";
  if (c === "X") return "Dbl";
  if (c === "XX") return "Rdbl";
  const st = c.slice(1);
  return st === "NT" ? `${c[0]}NT` : `${c[0]}<span class="s-${st}">${SUIT_SYM[st]}</span>`;
}
function callClass(c) { return c === "P" ? "pass" : c === "X" ? "dbl" : c === "XX" ? "rdbl" : ""; }
// a call as set text (the pad, the sheet)
function tile(c, extra = "", attrs = "") { return `<span class="call ${callClass(c)} ${extra}" ${attrs}>${callInner(c)}</span>`; }
function cardText(card) { return SUIT_SYM[card[0]] + (card[1] === "T" ? "10" : card[1]); }
function cardSpan(card) { return `<b class="c-${card[0]}">${cardText(card)}</b>`; }

// "1S" / "2NT" / "Pass" in running text -> suit symbols in the deck's colours
function prettyText(t) {
  return esc(t)
    .replace(/\(([AKQJT2-9](?: [AKQJT2-9])*)\)/g, (m, cards) => `(${cards.replace(/T/g, "10")})`)
    .replace(/\b([1-7])(NT|C|D|H|S)\b/g, (m, l, st) => st === "NT" ? `${l}NT` : `${l}<span class="s-${st}">${SUIT_SYM[st]}</span>`)
    .replace(/([♠♥♦♣])(10|[2-9AKQJ])/g, (m, s, r) => `<b class="c-${{ "♠": "S", "♥": "H", "♦": "D", "♣": "C" }[s]}">${s}${r}</b>`);
}

function parseHand(handStr) {
  const parts = Object.fromEntries(handStr.split(" ").map((p) => p.split(":")));
  const out = {};
  for (const s of ["S", "H", "D", "C"]) out[s] = parts[s] && parts[s] !== "-" ? parts[s].split("") : [];
  return out;
}
function hcpOf(handStr) {
  const v = { A: 4, K: 3, Q: 2, J: 1 };
  return handStr.replace(/[SHDC]:/g, "").split("").reduce((t, c) => t + (v[c] || 0), 0);
}

function cardHtml(s, r, size, heat) {
  const rank = r === "T" ? "10" : r;
  const h = heat ? `<span class="heat"><i style="width:${Math.round(heat.share * 100)}%"></i></span>` : "";
  const key = heat && heat.share >= 0.25 ? " key" : "";
  const title = heat ? ` title="${esc(heat.title)}"` : "";
  return `<div class="card ${size} c-${s}${key}" data-card="${s}${r}"${title}><span class="r">${rank}</span><span class="s">${SUIT_SYM[s]}</span>${h}</div>`;
}
function handHtml(handStr, size, heatMap, twoRows) {
  const h = parseHand(handStr);
  const group = (s) => !h[s].length ? `<span class="void">— ${SUIT_SYM[s]}</span>`
    : `<div class="suit-group">` + h[s].map((r) => cardHtml(s, r, size, heatMap && heatMap[s + r])).join("") + `</div>`;
  // alternate colours: ♠ ♥ ♣ ♦; the side seats in two rows (majors, minors)
  if (twoRows) return `<div class="hand two-rows"><div class="hrow">${group("S")}${group("H")}</div><div class="hrow">${group("C")}${group("D")}</div></div>`;
  return `<div class="hand">${["S", "H", "C", "D"].map(group).join("")}</div>`;
}
function backsHtml(size) {
  return `<div class="hand hidden-hand"><div class="suit-group">` +
         Array.from({ length: 13 }, () => `<div class="card ${size} back"></div>`).join("") + `</div></div>`;
}
function modelOptions(selected, withHuman) {
  return (withHuman ? [`<option value="human" ${selected === "human" ? "selected" : ""}>you</option>`] : [])
    .concat(state.models.map((x) => `<option value="${x.id}" title="${esc(x.label)}" ${x.id === selected ? "selected" : ""}>${esc(x.short)}</option>`)).join("");
}

// ------------------------------------------------------------------ setup
async function init() {
  const m = await api("GET", "/api/models");
  state.models = m.models;
  const has = (id) => m.models.some((x) => x.id === id);
  const ours = has("independent-sl") ? "independent-sl" : m.default;
  const theirs = has("brl-model-sl") ? "brl-model-sl" : m.default;
  for (const s of SEATS) $(`seat-sel-${s}`).innerHTML = modelOptions((s === "N" || s === "S") ? ours : theirs, true);
  $("advisor-sel").innerHTML = modelOptions(ours, false);
  $("btn-new").onclick = () => { $("seed").value = ""; newDeal(false); };  // always a random deal (Options picks one by number)
  $("btn-deal").onclick = () => { closeSetup(); newDeal(true); };
  $("btn-setup").onclick = openSetup;
  $("btn-example").onclick = () => {  // a Funbridge deal used to compare explanations
    $("pbn").value = "N:Q6.A76.JT32.KT64 AT74.K9.KQ96.953 52.83.A854.AQ872 KJ983.QJT542.7.J";
    $("calls").value = "1D 2C 2H 3C P";
    $("dealer").value = "E";
    $("vul").value = "None";
    $("seat-sel-S").value = "human";
  };
  $("setup-close").onclick = closeSetup;
  $("setup").onclick = (ev) => { if (ev.target === $("setup")) closeSetup(); };
  $("btn-next").onclick = () => move("next");
  $("btn-auto").onclick = () => move("auto");
  $("btn-box").onclick = () => { state.explore = !state.explore; render(); };
  $("show-all").onchange = () => render();
  $("llm-status").onclick = async () => {
    if (state.llm && state.llm.loaded && confirm("Unload Qwen and free its GPU memory?")) { await api("POST", "/api/llm/unload"); pollLLM(); }
  };
  document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") { hidePopup(); closeSetup(); } });
  document.addEventListener("click", (ev) => { if (!ev.target.closest(".call, .bcard, .popup")) hidePopup(); });
  window.addEventListener("scroll", () => hidePopup(), { passive: true });
  pollLLM();
  setInterval(pollLLM, 8000);
  // URL options (sharing a deal / screenshots): ?seed=42&N=human&vul=NS&dealer=N&auto=1&play=1NT&select=3&tab=cards
  // &pbn=N:...&calls=1D,2C,2H,3C,P&ask=...&popup=3D&by=<model id>&vs=X|alt&open=<concept name>
  const q = new URLSearchParams(location.search);
  for (const s of SEATS) if (q.get(s)) $(`seat-sel-${s}`).value = q.get(s);
  if (q.get("advisor")) $("advisor-sel").value = q.get("advisor");
  if (q.get("seed")) $("seed").value = q.get("seed");
  if (q.get("vul")) $("vul").value = q.get("vul");
  if (q.get("dealer")) $("dealer").value = q.get("dealer");
  if (q.get("pbn")) $("pbn").value = q.get("pbn");
  if (q.get("calls")) $("calls").value = q.get("calls").replace(/,/g, " ");
  if (q.get("tab")) state.tab = q.get("tab");
  const keepVs = q.get("vs"), keepOpen = q.get("open");
  await newDeal(Boolean(q.get("pbn") || q.get("calls")));
  // the link's deal number, dealer and vulnerability are for its first deal only: New deal is random afterwards
  if (q.get("seed")) $("seed").value = "";
  if (q.get("dealer")) $("dealer").value = "you";
  if (q.get("vul")) $("vul").value = "random";
  if (q.get("auto")) await move("auto");
  for (const c of (q.get("play") || "").split(",").filter(Boolean)) await humanCall(c);
  if (q.get("by")) state.explainBy = q.get("by");
  if (q.get("select") !== null && state.game && state.game.auction.length) {
    const i = Math.min(Number(q.get("select")), state.game.auction.length - 1);
    state.selected = state.game.auction[i].record;  // so that selecting it keeps the URL's vs and open
    state.vs = keepVs; state.openConcept = keepOpen;
    await selectRecord(state.game.auction[i].record);
  } else if (q.get("by") && state.selected !== null) await selectRecord(state.selected);
  for (const qq of q.getAll("ask")) await askQuestion(qq);
  if (q.get("explore")) { state.explore = true; render(); }
  if (q.get("popup_at") !== null) {  // the meaning card of the call at auction position k (screenshots)
    const el = document.querySelector(`#auction .call[data-k="${Number(q.get("popup_at"))}"]`);
    if (el) await showMeaning(el, Number(el.dataset.k), el.dataset.call);
  }
  if (q.get("popup")) {
    await loadMeanings(null);
    const el = document.querySelector(`#bidbox .bcard[data-call="${q.get("popup")}"]`);
    if (el) showMeaning(el, null, q.get("popup"));
  }
}
// at a human's turn: the advisor's view of the position, before the call (the hint)
async function adviseIfHumanTurn() {
  const g = state.game;
  if (!g || g.finished || g.seats[g.to_act].id !== "human") return;
  loadMeanings(null).then(() => render()).catch(() => {});
  if (!$("hints").checked) return;
  try {
    const r = await api("GET", `/api/game/${state.id}/advice`);
    await selectRecord(r.record);
  } catch (e) { status(e.message); }
}
function openSetup() { $("setup").classList.remove("hidden"); }
function closeSetup() { $("setup").classList.add("hidden"); }

async function pollLLM() {
  try {
    const s = await api("GET", "/api/llm");
    const el = $("llm-status");
    if (s.loaded) el.textContent = "Qwen loaded";
    else if (s.loading) el.textContent = "Qwen loading…";
    else if (s.error) el.textContent = "Qwen unavailable";
    else el.textContent = s.available ? "Qwen idle" : "No Qwen adapter";
    el.classList.toggle("clickable", Boolean(s.loaded));
    el.title = (s.error || s.adapter || "") + (s.loaded ? " (click to free the GPU)" : "");
    state.llm = s;
  } catch (e) { /* server restarting */ }
}

async function newDeal(withEntered) {
  const seats = {};
  for (const s of SEATS) seats[s] = $(`seat-sel-${s}`).value;
  // dealer "you" (the default): a human seat deals, so a new deal is bid from its first bid instead of
  // starting wherever the models' bids before the human's turn leave it; without a human seat, random
  const humans = SEATS.filter((s) => seats[s] === "human");
  const dealer = $("dealer").value === "you" ? (humans.includes("S") ? "S" : humans[0] || null) : $("dealer").value || null;
  const body = { seats, dealer, vulnerability: $("vul").value,
                 seed: $("seed").value === "" ? null : Number($("seed").value), advisor: $("advisor-sel").value };
  if (withEntered) {
    if ($("pbn").value.trim()) body.pbn = $("pbn").value.trim();
    const calls = $("calls").value.trim().split(/[\s,]+/).filter(Boolean);
    if (calls.length) body.calls = calls;
  }
  status("dealing…");
  hidePopup();
  try {
    const r = await api("POST", "/api/game", body);
    state.id = r.id; state.game = r.game; state.selected = null; state.recIndex = null; state.record = null; state.dialogs = {};
    state.meanings = {}; state.version++; state.animated = 0;
    $("seed").placeholder = `Random (last: ${r.game.seed})`;
    render(); clearAdvisor();
    status("");
    if (humanSeats().length && !state.game.finished && state.game.seats[state.game.to_act].id !== "human") await move("auto");
    else if (!state.game.finished && state.game.seats[state.game.to_act].id === "human") await adviseIfHumanTurn();
    else if (state.game.auction.length) await selectRecord(state.game.auction[state.game.auction.length - 1].record);
  } catch (e) { status(e.message); }
}

async function move(kind) {
  if (!state.id) return;
  status(kind === "auto" ? "bidding…" : "thinking…");
  hidePopup();
  try {
    const r = await api("POST", `/api/game/${state.id}/${kind}`);
    state.game = r.game; state.version++; render();
    const last = state.game.auction[state.game.auction.length - 1];
    if (!state.game.finished && state.game.seats[state.game.to_act].id === "human") await adviseIfHumanTurn();
    else if (last) await selectRecord(last.record);
    status("");
  } catch (e) { status(e.message); }
}

async function humanCall(call) {
  status("…");
  hidePopup();
  try {
    let r = await api("POST", `/api/game/${state.id}/call`, { call });
    state.game = r.game; state.version++; render();
    const mine = state.game.auction[state.game.auction.length - 1];
    if (!state.game.finished && state.game.seats[state.game.to_act].id !== "human") {
      r = await api("POST", `/api/game/${state.id}/auto`);
      state.game = r.game; state.version++; render();
    }
    if (!state.game.finished && state.game.seats[state.game.to_act].id === "human") await adviseIfHumanTurn();
    else await selectRecord(mine.record);
    status("");
  } catch (e) { status(e.message); }
}

// switching who plays a seat in the middle of a deal (from its name plate)
async function switchSeat(seat, player) {
  try {
    const r = await api("POST", `/api/game/${state.id}/seat`, { seat, player });
    state.game = r.game; state.version++;
    $(`seat-sel-${seat}`).value = player;  // the next deal keeps it
    render();
    status(`${SEAT_NAME[seat]}: ${player === "human" ? "you" : modelLabel(player)}`);
    if (!state.game.finished && state.game.to_act === seat && player === "human") await adviseIfHumanTurn();
  } catch (e) { status(e.message); }
}
// the model that explains the selected call (sticky); at a human's turn also the advisor
async function switchExplainer(model) {
  state.explainBy = model;
  const rec = state.record;
  if (rec && (rec.advice || rec.human_call !== undefined)) {
    try {
      const r = await api("POST", `/api/game/${state.id}/advisor`, { model });
      state.game = r.game; $("advisor-sel").value = model;
    } catch (e) { status(e.message); }
  }
  if (state.selected !== null) await selectRecord(state.selected);
}

// ------------------------------------------------------------------ the table
function humanSeats() { return SEATS.filter((s) => state.game.seats[s].id === "human"); }
function vulSeats(g) { return { None: [], NS: ["N", "S"], EW: ["E", "W"], Both: SEATS }[g.vulnerability]; }

function heatFor(seat) {
  const rec = state.record;
  if (!rec || rec.seat !== seat || !rec.card_stats || !rec.card_stats.cards) return null;
  const map = {};
  for (const c of rec.card_stats.cards) {
    const to = c.to.length ? ` (mostly to ${plainCall(c.to[0].call)})` : "";
    map[c.card] = { share: c.share, title: `Replacing ${cardText(c.card)} by another card changes ${plainCall(rec.call)} in ${c.changed} of ${c.of} cases${to}` };
  }
  return map;
}

function render() {
  const g = state.game;
  if (!g) return;
  const humans = humanSeats();
  const showAll = $("show-all").checked || humans.length === 0 || g.finished;
  const vul = vulSeats(g);
  // a single human sits at the bottom of the table, as on the bridge sites
  const rot = humans.length === 1 ? (SEATS.indexOf(humans[0]) - 2 + 4) % 4 : 0;
  const bottom = SEATS[(2 + rot) % 4];
  for (const s of SEATS) {
    const pos = SEATS[(SEATS.indexOf(s) - rot + 4) % 4];  // where it is drawn
    const el = $(`seat-${pos}`);
    el.dataset.seat = s;
    const big = s === bottom;
    const size = big ? "" : "sm";
    const shown = showAll || humans.includes(s);
    const plate = `<div class="plate ${g.to_act === s && !g.finished ? "turn" : ""}">` +
      `<span class="letter ${vul.includes(s) ? "vul" : ""}" title="${SEAT_NAME[s]}${vul.includes(s) ? ", vulnerable" : ""}">${s}</span>` +
      `<select data-seat="${s}" title="Who plays ${SEAT_NAME[s]} (switch at any time)">${modelOptions(g.seats[s].id, true)}</select>` +
      (shown ? `<span class="hcp">${hcpOf(g.hands[s])} HCP</span>` : "") + `</div>`;
    const side = (pos === "E" || pos === "W") && !big;
    const hand = shown ? handHtml(g.hands[s], size, heatFor(s), side) : backsHtml(size);
    el.innerHTML = s === bottom ? hand + plate : plate + hand;
    el.querySelector("select").onchange = (ev) => switchSeat(s, ev.target.value);
  }
  renderAuction();
  const humanTurn = !g.finished && g.seats[g.to_act] && g.seats[g.to_act].id === "human";
  $("btn-next").disabled = g.finished || humanTurn;
  $("btn-auto").disabled = g.finished || humanTurn;
  $("btn-box").classList.toggle("on", state.explore);
  $("btn-box").disabled = g.finished;
  renderBidbox(humanTurn, (humanTurn || state.explore) && !g.finished);
  renderResult();
}

function renderAuction() {
  const g = state.game, vul = vulSeats(g);
  const vulText = g.vulnerability === "None" ? "None vulnerable" : g.vulnerability === "Both" ? "Both vulnerable" : `${g.vulnerability.split("").join("-")} vulnerable`;
  $("board").textContent = `${g.custom ? "Entered deal" : `Deal ${g.seed}`} · Dealer ${SEAT_NAME[g.dealer]} · ${vulText}`;
  let h = PAD_ORDER.map((s) => `<div class="hd ${vul.includes(s) ? "vul" : ""}">${SEAT_NAME[s]}</div>`).join("");
  const cells = Array(PAD_ORDER.indexOf(g.dealer)).fill(null).concat(g.auction.map((a, k) => ({ ...a, k })));
  if (!g.finished) cells.push("?");
  for (const a of cells) {
    if (a === null) { h += `<div class="slot"></div>`; continue; }
    if (a === "?") { h += `<div class="slot"><span class="call q">?</span></div>`; continue; }
    const sel = (state.selected === a.record ? " sel" : "") + (a.k >= (state.animated || 0) ? " new" : "");
    const mark = a.unusual ? `<span class="mark" title="unusual for the system">!</span>`
      : (a.human && a.advisor_call !== a.call) ? `<span class="mark diff" title="the advisor would bid ${esc(plainCall(a.advisor_call))}">≠</span>` : "";
    h += `<div class="slot"><span class="call ${callClass(a.call)}${sel}${a.human && !a.entered ? " human" : ""}" data-rec="${a.record}" data-k="${a.k}" data-call="${a.call}">${callInner(a.call)}${mark}</span></div>`;
  }
  $("auction").innerHTML = h;
  state.animated = g.auction.length;
  document.querySelectorAll("#auction .call[data-rec]").forEach((el) => {
    el.onclick = (ev) => {
      ev.stopPropagation();
      const touch = matchMedia("(hover: none)").matches;
      selectRecord(Number(el.dataset.rec), touch);
      if (!touch) showMeaning(el, Number(el.dataset.k), el.dataset.call);
    };
    el.onmouseenter = () => showMeaning(el, Number(el.dataset.k), el.dataset.call);
    el.onmouseleave = () => scheduleHide();
  });
  $("turn-line").textContent = "";
}

function bcard(c, legal, p) {
  const st = c.length > 1 && /^[1-7]/.test(c) ? c.slice(1) : c;
  if (!legal) return `<span class="bcard ${st === "NT" ? "nt" : st} disabled">${callInner(c)}</span>`;
  const pr = p ? `<span class="pr" style="width:${Math.max(4, Math.round(p * 100))}%"></span>${p >= 0.05 ? `<span class="pv">${pct(p)}</span>` : ""}` : "";
  return `<span class="bcard ${st === "NT" ? "nt" : st}" data-call="${c}">${callInner(c)}${pr}</span>`;
}
function renderBidbox(humanTurn, visible) {
  const box = $("bidbox"), g = state.game;
  if (!visible) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  $("bidbox-title").textContent = humanTurn ? "" : `Meanings for ${SEAT_NAME[g.to_act]}`;
  $("bidbox-title").classList.toggle("hidden", humanTurn);
  const legal = new Set(g.legal);
  const m = state.meanings[meaningKey(null)];
  const pOf = {};
  if (m) for (const d of m.calls) pOf[d.call] = d.p;
  const firstLegal = Math.min(...g.legal.filter((c) => /^[1-7]/.test(c)).map((c) => Number(c[0])), 8);
  let grid = "";
  for (let lvl = Math.min(firstLegal, 7); lvl <= 7; lvl++) for (const st of STRAINS) grid += bcard(`${lvl}${st}`, legal.has(`${lvl}${st}`), pOf[`${lvl}${st}`]);
  $("bidbox-grid").innerHTML = grid;
  $("bidbox-special").innerHTML = ["P", "X", "XX"].map((c) => bcard(c, legal.has(c), pOf[c])).join("");
  box.querySelectorAll(".bcard[data-call]").forEach((el) => {
    el.onmouseenter = () => showMeaning(el, null, el.dataset.call);
    el.onmouseleave = () => scheduleHide();
    el.onclick = (ev) => {
      ev.stopPropagation();
      if (humanTurn && !matchMedia("(hover: none)").matches) { hidePopup(); humanCall(el.dataset.call); return; }
      if (humanTurn && $("popup").dataset.call === el.dataset.call && !$("popup").classList.contains("hidden")) { hidePopup(); humanCall(el.dataset.call); return; }
      showMeaning(el, null, el.dataset.call);  // touch: the first tap shows the meaning, the second bids
    };
  });
}

function renderResult() {
  const g = state.game, el = $("result");
  if (!g.finished) { el.classList.add("hidden"); return; }
  el.classList.remove("hidden");
  const r = g.result;
  let head;
  if (r.contract === "passed out") head = "Passed out.";
  else {
    const c = r.contract.replace(/X+$/, ""), dbl = r.contract.endsWith("XX") ? " redoubled" : r.contract.endsWith("X") ? " doubled" : "";
    head = `${tile(c)}${dbl} by ${SEAT_NAME[r.declarer]}`;
    if (r.tricks !== null && r.tricks !== undefined) {
      const outcome = r.made ? (r.overtricks ? `made with ${r.overtricks} overtrick${r.overtricks > 1 ? "s" : ""}` : "made exactly") : `down ${r.down}`;
      head += ` — ${r.tricks} tricks double-dummy, ${outcome}. North-South ${r.score_ns > 0 ? "+" : ""}${r.score_ns}.`;
    } else head += " (an entered deal: no double-dummy result).";
  }
  let dd = "";
  if (g.dd_table) {
    dd = `<table class="dd"><tr><th></th>` +
         ["NT", "S", "H", "D", "C"].map((s) => `<th>${s === "NT" ? "NT" : `<span class="c-${s}">${SUIT_SYM[s]}</span>`}</th>`).join("") + `</tr>`;
    for (const s of SEATS) {
      dd += `<tr><th>${s}</th>` + ["NT", "S", "H", "D", "C"].map((st) => {
        const hit = r.declarer === s && r.contract.replace(/X+$/, "").slice(1) === st;
        return `<td class="${hit ? "hit" : ""}">${g.dd_table[s][st]}</td>`;
      }).join("") + "</tr>";
    }
    dd += `</table><div class="hint">double-dummy tricks for each declarer and suit</div>`;
  }
  el.innerHTML = `<div class="result-head">${head}</div>${dd}`;
}

// ------------------------------------------------------------------ meanings (the index card)
// whose meaning: the model sitting at the seat that calls at position `at`, else (a human's
// seat) the advisor
function modelAt(at) {
  const g = state.game;
  const seat = at === null || at >= g.auction.length ? g.to_act : g.auction[at].seat;
  const id = seat && g.seats[seat] ? g.seats[seat].id : null;
  return id && id !== "human" ? id : g.advisor.id;
}
function meaningKey(at) { return `${state.version}:${at === null ? "now" : at}:${modelAt(at)}`; }
async function loadMeanings(at) {
  const key = meaningKey(at);
  if (state.meanings[key]) return state.meanings[key];
  if (state.meanings[key + ":p"]) return state.meanings[key + ":p"];
  const p = api("GET", `/api/game/${state.id}/meanings?at=${at === null ? "current" : at}&model=${encodeURIComponent(modelAt(at))}`)
    .then((m) => { state.meanings[key] = m; delete state.meanings[key + ":p"]; return m; })
    .catch((e) => { delete state.meanings[key + ":p"]; throw e; });
  state.meanings[key + ":p"] = p;
  return p;
}

let hideTimer = null;
function scheduleHide() { clearTimeout(hideTimer); hideTimer = setTimeout(hidePopup, 260); }
function hidePopup() { clearTimeout(hideTimer); const p = $("popup"); p.classList.add("hidden"); p.dataset.call = ""; }

async function showMeaning(el, at, call) {
  clearTimeout(hideTimer);
  const pop = $("popup");
  pop.dataset.call = call;
  pop.innerHTML = `<div class="ph">${tile(call)}</div><div class="none">measuring…</div>`;
  pop.classList.remove("hidden");
  placePopup(el);
  pop.onmouseenter = () => clearTimeout(hideTimer);
  pop.onmouseleave = () => scheduleHide();
  let m;
  try { m = await loadMeanings(at); } catch (e) { pop.innerHTML = `<div class="ph">${tile(call)}</div><div class="none">${esc(e.message)}</div>`; return; }
  if (pop.dataset.call !== call) return;
  pop.innerHTML = meaningHtml(m.calls.find((x) => x.call === call), m);
  placePopup(el);
}

function placePopup(el) {
  const pop = $("popup"), r = el.getBoundingClientRect(), w = pop.offsetWidth, h = pop.offsetHeight;
  const left = Math.min(window.innerWidth - w - 12, Math.max(12, r.left + r.width / 2 - w / 2));
  let top = r.top - h - 10;
  if (top < 60) top = Math.min(window.innerHeight - h - 12, r.bottom + 10);
  pop.style.left = `${left}px`;
  pop.style.top = `${Math.max(12, top)}px`;
}

function forcingPill(f) {
  if (!f) return "";
  return `<span class="forcing" title="partner passed it ${Math.round(f.rate * 100)}% of ${f.n.toLocaleString()} times when the next opponent passed">${esc(f.label)}</span>`;
}
function suitCells(src, other) {
  return ["S", "H", "D", "C"].map((s) => {
    const x = src.suits.find((y) => y.suit === s);
    const y = other ? other.suits.find((z) => z.suit === s) : null;
    const t = x.min === x.max ? `${x.min}` : x.max >= 13 ? `${x.min}+` : `${x.min}–${x.max}`;
    return `<div class="row"><span class="sym c-${s}">${SUIT_SYM[s]}</span><span>${t}</span>${suitScale(x, y)}</div>`;
  }).join("");
}
// a range on a fixed scale: the 5th-95th percentile band and the median tick, optionally the
// model's own band under it (thinner), so the numbers in the card can be read at a glance
function rangeBand(r, max, cls = "") {
  const at = (v) => `${(Math.min(v, max) / max) * 100}%`;
  const w = (Math.min(r[2], max) - Math.min(r[0], max)) / max * 100;
  return `<div class="band ${cls}" style="left:${at(r[0])};width:${Math.max(w, 1.5)}%"></div>` +
         `<div class="tick ${cls}" style="left:${at(r[1])}"></div>`;
}
function hcpScale(main, other) {
  const top = Math.max(main.hcp[2], other ? other.hcp[2] : 0);
  const max = Math.max(25, Math.ceil(top / 5) * 5);
  let ticks = "";
  for (let v = 0; v <= max; v += 5) ticks += `<span style="left:${(v / max) * 100}%">${v}</span>`;
  return `<div class="scale${other ? " two" : ""}"><div class="track">${rangeBand(main.hcp, max)}${other ? rangeBand(other.hcp, max, "alt") : ""}</div>` +
         `<div class="ticks">${ticks}</div></div>`;
}
const SUIT_MAX = 10;
function suitScale(x, y) {
  const r = (v) => [v.min, v.median ?? v.min, v.max];
  return `<div class="track mini${y ? " two" : ""}">${rangeBand(r(x), SUIT_MAX)}${y ? rangeBand(r(y), SUIT_MAX, "alt") : ""}</div>`;
}
// "bid it 1,370 times (10% of 13,361)": the count alone has no scale
function wb5Count(w) {
  const n = w.n.toLocaleString("en-US");
  return w.of ? `bid it ${n} times (${pct(w.n / w.of)} of ${w.of.toLocaleString("en-US")})` : `bid it ${n} times`;
}

function meaningHtml(d, m) {
  if (!d) return `<div class="none">No data for this bid.</div>`;
  const main = d.wb5 || d.model;
  let h = `<div class="ph">${tile(d.call)}<span>${main ? esc(main.hcp_text) : ""}</span>${d.wb5 && d.wb5.forcing ? forcingPill(d.wb5.forcing) : ""}</div>`;
  const other = d.wb5 && d.model ? d.model : null;  // the model's own hands, drawn under WBridge5's
  if (main) h += hcpScale(main, other) + `<div class="rows">${suitCells(main, other)}</div>`;
  else h += `<div class="none">WBridge5 never reached this exact auction 20 times, and ${esc(m.model)} makes this bid with too few hands here to say what it shows.</div>`;
  if (d.phrase) h += `<div class="desc">${prettyText(cap(d.phrase))}</div>`;
  let mm = d.p !== undefined ? `${esc(m.model)} bids it ${pct(d.p)} here` : "";
  if (d.model && d.wb5) {
    mm += ` · its own hands for it: ${esc(d.model.hcp_text)}, ` + ["S", "H", "D", "C"].map((s) => {
      const x = d.model.suits.find((y) => y.suit === s); return `<span class="c-${s}">${SUIT_SYM[s]}</span>${x.min}–${x.max}`; }).join(" ");
  }
  if (mm) h += `<div class="model">${mm}</div>`;
  const src = d.wb5
    ? `WBridge5, ${d.wb5.match === "exact" ? "this exact auction" : "this auction (leading passes aside)"}: ${wb5Count(d.wb5)}, 5th–95th percentile`
    : d.model ? `No WBridge5 data for this auction: the hands ${esc(m.model)} itself makes the bid with (${Math.round(d.model.n_eff)} effective)` : "";
  if (src) h += `<div class="src">${src}</div>`;
  if (other) h += `<div class="src key"><i></i>WBridge5 <i class="alt"></i>${esc(m.model.replace(/\s*\(.*\)/, ""))}</div>`;
  return h;
}

// ------------------------------------------------------------------ the side: the selected call
function clearAdvisor() {
  $("side-empty").classList.remove("hidden");
  $("side-body").classList.add("hidden");
}

async function selectRecord(i, scroll = false) {
  if (i !== state.selected) { state.vs = null; state.openConcept = null; state.allBids = false; state.allConcepts = false; }
  state.selected = i;
  const by = state.explainBy ? `?model=${encodeURIComponent(state.explainBy)}` : "";
  try { state.record = await api("GET", `/api/game/${state.id}/record/${i}${by}`); } catch (e) { status(e.message); return; }
  state.recIndex = state.record.index ?? i;
  render();
  renderAdvisor(state.record);
  // only when the player asked for it (a tap on a call): on a narrow window the side is below the table
  if (scroll && window.innerWidth <= 1150) $("side").scrollIntoView({ behavior: "smooth" });
}

// the auction position a record explains: its call's index, or the current position for an advice record
function auctionIndexOfRecord(i) {
  const k = state.game.auction.findIndex((a) => a.record === i);
  return k < 0 ? null : k;
}

const TABS = [["why", "Why"], ["weights", "Weights"], ["cards", "Cards"], ["ask", "Ask"], ["models", "Models"]];
const OLD_TABS = { meaning: "why", concepts: "why", more: "models" };  // older tab names, still accepted in links
function currentTab() { const t = OLD_TABS[state.tab] || state.tab; return TABS.some(([k]) => k === t) ? t : "why"; }

function renderAdvisor(rec) {
  $("side-empty").classList.add("hidden");
  const el = $("side-body");
  el.classList.remove("hidden");
  const human = rec.human_call !== undefined;
  const made = human ? rec.human_call : rec.call;
  const entered = human && (state.game.auction.find((a) => a.record === state.selected) || {}).entered;
  const role = rec.advice ? `advice from ${esc(who(rec))}` : entered ? "entered bid" : human ? "your bid"
    : rec.view_of !== undefined ? `as ${esc(who(rec))} would bid` : esc(who(rec));
  const shown = rec.view_of !== undefined && !human && !rec.advice ? rec.call : made;
  let h = `<div class="head"><div class="big-call ${callClass(shown)}">${callInner(shown)}</div><div>` +
       `<div class="who">${SEAT_NAME[rec.seat]} · ${role}</div>` +
       `<div class="title">${prettyText(cap(rec.situation).replace(/\b([1-7])(nt|c|d|h|s)\b/gi, (m) => m.toUpperCase()))}</div>` +
       `<div class="meta">${rec.hcp} HCP · ${rec.shape.split("-").map((n, j) => `<span class="c-${"SHDC"[j]}">${SUIT_SYM["SHDC"[j]]}</span>${n}`).join(" ")}</div></div></div>`;
  if (human) h += rec.human_call === rec.call ? `<div class="note">${esc(who(rec))} would make the same bid.</div>`
    : `<div class="note">${esc(who(rec))} would bid ${tile(rec.call)} instead; its reasoning below.</div>`;
  if (rec.actual_call && rec.actual_call !== rec.call) h += `<div class="note">At the table ${esc(state.game.seats[rec.seat].label)} bid ${tile(rec.actual_call)}.</div>`;
  if (state.explainBy) h += `<div class="note">Explained by ${esc(modelLabel(state.explainBy))}. <button class="btn link" id="by-reset">Back to the model at the seat</button></div>`;
  const tab = currentTab();
  h += `<div class="tabs">${TABS.map(([k, l]) => `<button class="tab ${k === tab ? "on" : ""}" data-tab="${k}">${l}</button>`).join("")}</div>`;
  h += `<div id="tabpane" class="tabpane"></div>`;
  el.innerHTML = h;
  el.querySelectorAll(".tab").forEach((b) => { b.onclick = () => { state.tab = b.dataset.tab; el.querySelectorAll(".tab").forEach((x) => x.classList.toggle("on", x === b)); fillTab(rec); }; });
  if ($("by-reset")) $("by-reset").onclick = () => { state.explainBy = null; selectRecord(state.selected); };
  fillTab(rec);
}
function who(rec) { return rec.model_id ? modelLabel(rec.model_id) : rec.model; }
function modelIdOf(rec) {  // the record's model id, from its label
  const m = state.models.find((x) => x.label === rec.model || x.short === rec.model);
  return m ? m.id : state.game.advisor.id;
}

function fillTab(rec) {
  const pane = $("tabpane");
  const tab = currentTab();
  const human = rec.human_call !== undefined;
  const made = human ? rec.human_call : rec.call;
  const k = auctionIndexOfRecord(state.selected);
  if (tab === "why") { pane.innerHTML = whyTab(rec); fillMeaning(k, made); }
  else if (tab === "weights") { pane.innerHTML = `<div id="weights" class="hint">computing…</div>`; loadWeights(rec); }
  else if (tab === "cards") pane.innerHTML = cardsStatTab(rec);
  else if (tab === "ask") pane.innerHTML = askBoxHtml(rec);
  else pane.innerHTML = `<h3>What each model would bid here</h3><div id="compare" class="hint">loading…</div>`;
  // wiring
  const lb = $("btn-llm");
  if (lb) lb.onclick = () => askLLM(rec);
  if ($("ask-form")) $("ask-form").onsubmit = (ev) => { ev.preventDefault(); askQuestion($("ask-input").value); };
  pane.querySelectorAll(".chip").forEach((b) => { b.onclick = () => askQuestion(b.dataset.q); });
  pane.querySelectorAll(".call").forEach((t) => { t.style.cursor = "default"; });
  pane.querySelectorAll("[data-cards]").forEach((x) => {  // hovering lifts those cards in the hand on the table
    const cards = x.dataset.cards.split(",");
    const sel = () => cards.map((c) => `.seat[data-seat="${rec.seat}"] .card[data-card="${c}"]`).join(",");
    x.onmouseenter = () => document.querySelectorAll(sel()).forEach((c) => c.classList.add("pulse"));
    x.onmouseleave = () => document.querySelectorAll(".card.pulse").forEach((c) => c.classList.remove("pulse"));
  });
  pane.querySelectorAll(".call[data-alt]").forEach((t) => {  // what each alternative would have meant here
    t.style.cursor = "help";
    t.onmouseenter = () => showMeaning(t, k, t.dataset.alt);
    t.onmouseleave = () => scheduleHide();
    t.onclick = (ev) => { ev.stopPropagation(); showMeaning(t, k, t.dataset.alt); };
  });
  if (tab === "models") loadCompare();
}

// Why: the explanation, the reasons it rests on, what the call shows here, and the bids weighed
function whyTab(rec) {
  let h = `<div class="ex-text">${prettyText(rec.explanation.split(" This is unusual: ")[0])}</div>`;
  h += `<div class="ex-src">${esc(rec.explanation_source.replace(/\s*\(.*\)/, ""))} · ${rec.checks_passed ? `<span class="ok">checked</span>` : `<span class="bad">failed a check: ${esc(rec.check_problems.join("; "))}</span>`}</div>`;
  for (const n of rec.unusual) h += `<div class="note warn">Unusual. ${prettyText(n)}</div>`;
  h += `<div id="llm-box" style="margin-top:10px">${rec.llm ? llmHtml(rec.llm) : `<button id="btn-llm" class="btn">In Qwen's words</button> <span class="hint" id="llm-note"></span>`}</div>`;
  // each reason the model used, with its weight and the hand's evidence for it
  const rc = rec.reason_clauses || [];
  if (rc.length) {
    const contrib = Object.fromEntries(rec.concepts.map((c) => [c.name, c]));
    const behind = Object.fromEntries(((rec.card_stats || {}).concept_cards || []).map((x) => [x.name, x]));
    h += `<h3>Reasons<span class="aside">${rec.model_kind === "concept" ? "weight in the decision" : "the rules' reading (no concept layer)"}</span></h3><div class="why-list">`;
    for (const r of rc) {
      const c = contrib[r.name];
      const val = c && c.contribution != null ? `<span class="w num ${c.contribution >= 0 ? "pos" : "neg"}">${c.contribution > 0 ? "+" : ""}${c.contribution.toFixed(2)}</span>` : "";
      h += `<div><div class="why-head">${esc(cap(r.gloss))}${val}</div>` +
           (r.clause ? `<div class="why-ev">${prettyText(cap(r.clause))}.</div>`
            : r.hand_based ? "" : `<div class="why-ev dim">A fact about the bid itself.</div>`) +
           (c && c.note ? `<div class="why-ev dim">${esc(cap(c.note))}</div>` : "") +
           (behind[r.name] && behind[r.name].cards.length
             ? `<div class="why-cards" data-cards="${behind[r.name].cards.map((x) => x.card).join(",")}">Switched off by replacing ` +
               behind[r.name].cards.map((x) => `${cardSpan(x.card)} <span class="num">${x.n}/${x.of}</span>`).join(", ") + `</div>` : "") + `</div>`;
    }
    h += `</div>`;
    // what the model also relies on that the rules reject for this hand: where the concept account is wrong
    for (const c of rec.concepts.filter((c) => c.contribution > 0 && !c.true && c.activation >= 0.5).sort((a, b) => b.contribution - a.contribution).slice(0, 2)) {
      h += `<div class="note warn">The model also relies on “${esc(c.gloss)}” (+${c.contribution.toFixed(2)}), which the rules reject for this hand.${c.note ? ` ${esc(cap(c.note))}` : ""}</div>`;
    }
  }
  h += `<div id="adv-meaning"></div>`;
  h += `<h3>Bids weighed${rec.trust && rec.trust.agree === false ? `<span class="aside">the concepts alone would bid ${esc(plainCall(rec.trust.concept_only_bid))}</span>` : ""}</h3><div class="probs">`;
  rec.probabilities.slice(0, 5).forEach((a, j) => {
    h += `<div class="prob-row ${j === 0 ? "top" : ""}">${tile(a.call, "", `data-alt="${a.call}"`)}<div class="prob-bar"><i style="width:${Math.max(1, a.p * 100)}%"></i></div><div class="val">${pct(a.p)}</div></div>`;
  });
  return h + `</div>`;
}

// what the call shows in this auction (WBridge5's measured ranges), one line under the reasons
async function fillMeaning(k, call) {
  try {
    const m = await loadMeanings(k);
    const box = $("adv-meaning");
    if (!box) return;
    const d = m.calls.find((x) => x.call === call);
    const main = d && (d.wb5 || d.model);
    if (!main) return;
    const suits = ["S", "H", "D", "C"].map((s) => {
      const x = main.suits.find((y) => y.suit === s);
      return `<span class="c-${s}">${SUIT_SYM[s]}</span>${x.min === x.max ? x.min : x.max >= 13 ? `${x.min}+` : `${x.min}–${x.max}`}`; }).join(" ");
    const src = d.wb5 ? `WBridge5, ${d.wb5.match === "exact" ? "this exact auction" : "this auction (leading passes aside)"}: ${wb5Count(d.wb5)}` : `${esc(m.model)}'s own hands for it`;
    box.innerHTML = `<h3>What ${tile(call)} shows here${d.wb5 && d.wb5.forcing ? `<span class="aside">${esc(d.wb5.forcing.label)}</span>` : ""}</h3>` +
      `<div class="mean-line"><b>${esc(main.hcp_text)}</b> · ${suits}</div><div class="ex-src">${src}</div>`;
  } catch (e) { /* the meaning is optional here */ }
}

function cardsStatTab(rec) {
  const st = rec.card_stats;
  if (!st || !st.cards) return `<div class="hint">No card statistics for this decision.</div>`;
  let h = `<h3>Which cards carry ${tile(st.call)}<span class="aside">${st.kept} of ${st.n_swaps} changes keep it</span></h3>`;
  h += `<div class="keycards">` + ["S", "H", "D", "C"].flatMap((s) => st.cards.filter((c) => c.card[0] === s)).map((c) =>
    `<div class="kc" data-cards="${c.card}" title="Replacing ${cardText(c.card)} changes the bid in ${c.changed} of ${c.of} cases${c.to.length ? ", mostly to " + plainCall(c.to[0].call) : ""}">` +
    `<div class="face c-${c.card[0]}">${c.card[1] === "T" ? "10" : c.card[1]}<br>${SUIT_SYM[c.card[0]]}</div>` +
    `<div class="bar"><i style="height:${Math.round(c.share * 100)}%"></i></div><div class="n">${c.changed}</div></div>`).join("") + `</div>`;
  h += `<div class="hint">Each bar: how many of the card's ${st.cards[0] ? st.cards[0].of : 39} replacements change the bid.</div>`;
  if (st.calls.length) {
    h += `<h3>What would change the bid</h3><div class="change-list">`;
    for (const c of st.calls) {
      const sw = c.simplest;
      h += `<div class="change" data-cards="${c.cards.map((x) => x.card).join(",")}">${tile(c.call)}<div>${c.n} of ${st.n_swaps} changes; the smallest: ${cardSpan(sw.in)} instead of ${cardSpan(sw.out)}` +
           `<div class="c2">${sw.hcp} HCP, ${sw.shape}</div></div></div>`;
    }
    h += `</div>`;
  } else h += `<div class="hint" style="margin-top:8px">No one-card change leads to another call.</div>`;
  return h;
}

// ------------------------------------------------------------------ Weights: the policy layer at work
// every bid's score and probability, and the bid made against any other bid (or all the alternatives),
// concept by concept: value × (weight for the bid − weight for the other), as nlg.decision_record.score_breakdown
function signed(x) { return (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(2); }
function defaultVs(rec) { return rec.human_call !== undefined && rec.human_call !== rec.call ? rec.human_call : null; }

async function loadWeights(rec) {
  const vs = state.vs || defaultVs(rec);
  const i = state.recIndex;
  let w;
  try { w = await api("GET", `/api/game/${state.id}/record/${i}/weights${vs ? `?vs=${encodeURIComponent(vs)}` : ""}`); }
  catch (e) { if ($("weights")) $("weights").textContent = e.message; return; }
  const el = $("weights");
  if (!el || state.recIndex !== i) return;
  el.classList.remove("hint");
  el.innerHTML = weightsHtml(w, rec);
  loadWeightsWiring(el, w, rec);
}
function loadWeightsWiring(el, w, rec) {
  el.querySelectorAll("[data-vs]").forEach((x) => {
    x.onclick = () => { if (x.dataset.vs !== w.call) { state.vs = x.dataset.vs; loadWeights(rec); } };
  });
  el.querySelectorAll(".ct-row[data-k]").forEach((x) => {
    x.onclick = () => { const n = w.rows[Number(x.dataset.k)].name; state.openConcept = state.openConcept === n ? null : n;
                        el.innerHTML = weightsHtml(w, rec); loadWeightsWiring(el, w, rec); };
  });
  if ($("all-bids")) $("all-bids").onclick = () => { state.allBids = true; el.innerHTML = weightsHtml(w, rec); loadWeightsWiring(el, w, rec); };
  if ($("all-concepts")) $("all-concepts").onclick = () => { state.allConcepts = true; el.innerHTML = weightsHtml(w, rec); loadWeightsWiring(el, w, rec); };
}

function weightsHtml(w, rec) {
  const vsTile = (v) => v === "alt" ? "all the alternatives" : tile(v);
  // 1. the scores: every legal bid, most likely first (the rare ones folded away)
  const shown = state.allBids ? w.bids : w.bids.filter((b, j) => j < 3 || b.p >= 0.01 || b.call === w.vs);
  let h = `<h3>Every bid's score<span class="aside">${esc(w.model.replace(/\s*\(.*\)/, ""))}</span></h3><div class="scores">`;
  for (const b of shown) {
    const cls = b.call === w.call ? " made" : b.call === w.vs ? " vs" : "";
    h += `<div class="score-row${cls}" ${b.call === w.call ? "" : `data-vs="${b.call}" title="compare ${plainCall(w.call)} with ${plainCall(b.call)}"`}>` +
         `${tile(b.call)}<span class="num sc">${b.score.toFixed(2)}</span><div class="prob-bar"><i style="width:${Math.max(0.5, b.p * 100)}%"></i></div>` +
         `<div class="val">${pct(b.p)}</div></div>`;
  }
  if (shown.length < w.bids.length) h += `<button class="btn link more" id="all-bids">${w.bids.length - shown.length} more bids</button>`;
  h += `</div><div class="hint">The highest score is bid. A score one point higher makes a bid e ≈ 2.7 times as likely.</div>`;
  if (!w.rows) return h + `<div class="note">${esc(w.model)} has no concept layer: its scores come straight from its hidden layer, so they cannot be split into concepts.</div>`;
  // 2. the bid made against the compared bid, concept by concept
  const others = w.bids.filter((b) => b.call !== w.call);
  const ru = others.length ? others[0].call : null;
  h += `<h3>${tile(w.call)} against ${vsTile(w.vs)}</h3>`;
  h += `<div class="chips">` + (ru ? `<button class="chip ${w.vs === ru ? "on" : ""}" data-vs="${ru}">the runner-up, ${plainCall(ru)}</button>` : "") +
       `<button class="chip ${w.vs === "alt" ? "on" : ""}" data-vs="alt">all the alternatives</button></div>`;
  // the margin in score points, said in probabilities: against one bid it is exactly the log of their odds
  const pts = `<b>${w.margin.toFixed(2)} points</b>`, times = Math.exp(w.margin);
  h += `<div class="margin-line">` + (w.vs === "alt"
    ? `${tile(w.call)} ${pct(w.p)} against the other bids, ${pct(w.p_vs)} together: its score is ${pts} above their ` +
      `average score, each bid counted by how likely it is.`
    : `${tile(w.call)} ${pct(w.p)} against ${tile(w.vs)} ${pct(w.p_vs)}: its score is ${pts} higher, which makes it ` +
      `${times < 10 ? times.toFixed(1) : times < 1000 ? Math.round(times) : "over 1000"} times as likely.`) +
       ` The table splits these points concept by concept.</div>`;
  // the eight largest contributions first; the server sends twenty, the rest is one row
  const rows = state.allConcepts ? w.rows : w.rows.slice(0, 8);
  const folded = w.rows.slice(rows.length);
  const rest = { n: w.rest.n + folded.length, contribution: w.rest.contribution + folded.reduce((t, r) => t + r.contribution, 0) };
  const parts = [rest.contribution, w.constant, w.hidden || 0, w.margin].concat(rows.map((r) => r.contribution));
  const scale = Math.max(...parts.map(Math.abs), 0.01);
  const bar = (v) => {
    const width = Math.abs(v) / scale * 50;
    return `<div class="cbar"><i class="${v >= 0 ? "pos" : "neg"}" style="left:${v >= 0 ? 50 : 50 - width}%;width:${Math.max(width, 0.4)}%"></i></div>`;
  };
  const reasons = new Set(rec.reasons || []);
  h += `<div class="ct"><div class="ct-head"><span>concept</span><span class="num">value</span><span class="num span2">contribution</span></div>`;
  rows.forEach((r, k) => {
    // not true: ruled out by the bid and auction alone, or by this hand (a condition on the hand fails)
    const byBid = r.rule && r.rule.groups.every((g) => g.some((c) => !c.holds && !c.hand));
    const tags = (reasons.has(r.name) ? `<span class="tag reason">reason</span>` : "") +
                 (r.descriptor ? `<span class="tag">names the suit</span>` : r.true === false
                   ? `<span class="tag">${byBid ? prettyText(`not true for ${plainCall(w.call)}`) : "not true for this hand"}</span>` : "");
    h += `<div class="ct-row${state.openConcept === r.name ? " open" : ""}" data-k="${k}" title="click: the rule that defines this concept and what it adds to every bid's score">` +
         `<div class="nm">${esc(cap(r.gloss))}${tags}</div><span class="num cv">${r.value.toFixed(2)}</span>${bar(r.contribution)}` +
         `<span class="num cn ${r.contribution >= 0 ? "pos" : "neg"}">${signed(r.contribution)}</span></div>`;
    if (state.openConcept === r.name) h += (r.rule ? ruleHtml(w, r) : "") + stripHtml(w, r);
  });
  const total = (label, v, cls = "", title = "") => `<div class="ct-row sum ${cls}"${title ? ` title="${esc(title)}"` : ""}><div class="nm">${label}</div><span></span>${bar(v)}<span class="num cn ${v >= 0 ? "pos" : "neg"}">${signed(v)}</span></div>`;
  h += total(`the other ${rest.n} concepts` + (folded.length ? ` <button class="btn link" id="all-concepts">show ${folded.length} more</button>` : ""), rest.contribution);
  h += total("each bid's starting score (bias)", w.constant, "",
             `A fixed number for each bid, added to its score whatever the hand and the auction, learned in training ` +
             `together with the weights: a starting lean for or against the bid before any concept counts. This row is ` +
             `${plainCall(w.call)}'s minus ${w.vs === "alt" ? "the alternatives' average" : plainCall(w.vs) + "'s"}.`);
  if (w.hidden !== null) h += total("the hidden layer (joint model)", w.hidden);
  h += total(`score difference`, w.margin, "margin") + `</div>`;
  h += `<div class="hint">${prettyText(`A concept adds value × weight to each bid's score, so it counts for ${plainCall(w.call)} by value × (its weight for ` +
       `${plainCall(w.call)} − its weight for ${w.vs === "alt" ? "the alternatives" : plainCall(w.vs)}). Click a concept to see the rule that defines it, ` +
       `checked for ${plainCall(w.call)} with this hand, and what it adds to every bid.`)}</div>`;
  if (w.hidden !== null) h += `<div class="note">The joint model's policy layer also reads the network's hidden layer, shown as one row. The weights in the Why tab come from its concepts-only layer, so they differ from these.</div>`;
  return h;
}

// a concept's rule (concepts/rules.py, explain.rule_justification.rule_conditions) for the bid made with this hand: the
// concept is true when every condition holds (bid_is_splinter: every condition of either part); hover shows the code
function ruleHtml(w, r) {
  const ru = r.rule;
  if (ru.retired) return `<div class="rule"><div class="rule-head">A retired concept: its rule is never true. It is kept so that trained networks keep their columns.</div></div>`;
  let h = `<div class="rule"><div class="rule-head"><span>True when ${ru.groups.length > 1 ? "every condition of either part holds" : "every condition holds"}</span>` +
          `<span class="${ru.holds ? "ok" : "bad"}">${ru.holds ? "true" : "not true"} for ${tile(w.call)}</span></div>`;
  ru.groups.forEach((g, j) => {
    if (ru.groups.length > 1) h += `<div class="rule-or">${j ? "or" : "either"}</div>`;
    for (const c of g) {
      h += `<div class="cond" title="${esc(c.code)}"><span>${prettyText(cap(c.text))}${c.hand ? `<span class="tag">about the hand</span>` : ""}</span>` +
           `<span class="${c.holds ? "ok" : "bad"}">${c.holds ? "holds" : "fails"}</span></div>`;
    }
  });
  return h + `</div>`;
}

// one concept's push on every bid's score, as a line (thesis Figure 3.9): the bid made, the compared bid,
// the other bids sized by their probability and, against all the alternatives, their weighted average
function stripHtml(w, r) {
  const vals = r.points, lo = Math.min(...vals), hi = Math.max(...vals), span = Math.max(hi - lo, 1e-6);
  const at = (v) => `${4 + (v - lo) / span * 92}%`;
  const iCall = w.bids.findIndex((b) => b.call === w.call);
  const rest = w.bids.map((b, j) => j).filter((j) => j !== iCall);
  const pRest = rest.reduce((t, j) => t + w.bids[j].p, 0) || 1;
  const avg = rest.reduce((t, j) => t + vals[j] * w.bids[j].p / pRest, 0);
  let dots = "";
  rest.slice().reverse().forEach((j) => {
    const b = w.bids[j], d = 5 + 16 * Math.sqrt(b.p);
    const cls = b.call === w.vs ? "vs" : "";
    dots += `<i class="dot ${cls}" style="left:${at(vals[j])};width:${d}px;height:${d}px" title="${plainCall(b.call)} ${signed(vals[j])} (${pct(b.p)})"></i>`;
  });
  dots += `<i class="dot made" style="left:${at(vals[iCall])}" title="${plainCall(w.call)} ${signed(vals[iCall])}"></i>`;
  if (w.vs === "alt") dots += `<i class="dia" style="left:${at(avg)}" title="the alternatives on average ${signed(avg)}"></i>`;
  const lab = (v, html, cls) => `<span class="lab ${cls}" style="left:${at(v)}">${html}</span>`;
  const jVs = w.vs === "alt" ? -1 : w.bids.findIndex((b) => b.call === w.vs);
  const labels = lab(vals[iCall], tile(w.call), "made") + (jVs >= 0 ? lab(vals[jVs], tile(w.vs), "vs") : lab(avg, "average", "avg"));
  const below = rest.filter((j) => j !== jVs).slice(0, 2).map((j) => lab(vals[j], tile(w.bids[j].call), "other")).join("");
  return `<div class="strip"><div class="strip-labels">${labels}</div><div class="strip-track">${dots}</div>` +
         `<div class="strip-labels below">${below}</div>` +
         `<div class="strip-ends"><span>${signed(lo)}</span><span>what it adds to each bid's score</span><span>${signed(hi)}</span></div></div>`;
}

// ------------------------------------------------------------------ follow-up questions
// the calls a player would actually consider here: the model's real alternatives (2%+), else the
// natural candidates of the auction -- a raise of our suit, 3NT, a cuebid of theirs -- rather than a
// runner-up the model gives 0.2% (4S on the example deal)
function candidateCalls(rec) {
  const real = rec.probabilities.filter((a) => a.call !== rec.call && a.p >= 0.02).map((a) => a.call);
  if (real.length) return real;
  const legal = new Set(rec.legal_calls || rec.probabilities.map((a) => a.call));
  const k = auctionIndexOfRecord(state.selected);
  const before = k === null ? state.game.auction : state.game.auction.slice(0, k);
  const seat = rec.seat, partner = SEATS[(SEATS.indexOf(seat) + 2) % 4];
  const ours = before.filter((a) => (a.seat === seat || a.seat === partner) && /^[1-7][CDHS]$/.test(a.call)).map((a) => a.call[1]);
  const theirs = before.filter((a) => a.seat !== seat && a.seat !== partner && /^[1-7][CDHS]$/.test(a.call)).map((a) => a.call[1]);
  const last = before.filter((a) => /^[1-7]/.test(a.call)).pop();
  const rank = (c) => (Number(c[0]) - 1) * 5 + STRAINS.indexOf(c.slice(1));
  const cheapest = (st) => { for (let l = 1; l <= 7; l++) { const c = `${l}${st}`; if (!last || rank(c) > rank(last.call)) return c; } return null; };
  const out = [];
  if (ours.length) out.push(cheapest(ours[ours.length - 1]));
  out.push("3NT");
  if (theirs.length) out.push(cheapest(theirs[0]));
  if (ours.length) out.push({ C: "5C", D: "5D", H: "4H", S: "4S" }[ours[ours.length - 1]]);
  return [...new Set(out.filter((c) => c && c !== rec.call && legal.has(c)))];
}

function suggestions(rec) {
  const out = [];
  if (rec.human_call !== undefined && rec.human_call !== rec.call) out.push(`Why not ${plainCall(rec.human_call)}?`);
  const alts = candidateCalls(rec).filter((c) => c !== rec.human_call);
  if (alts.length) out.push(`Why not ${plainCall(alts[0])}?`);
  const k = auctionIndexOfRecord(state.selected);
  const before = k === null ? state.game.auction : state.game.auction.slice(0, k);
  const other = before.filter((a) => a.seat !== rec.seat && a.call !== "P").pop();
  if (other) out.push(`What does ${SEAT_NAME[other.seat]}'s ${plainCall(other.call)} show?`);
  const game = alts.find((c) => ["3NT", "4H", "4S", "5C", "5D"].includes(c)) || alts[1];
  out.push(game ? `What would I need for ${plainCall(game)}?` : "What would change your bid?");
  out.push("How sure are you?");
  return out.slice(0, 4);
}
function qaHtml(x) {
  if (x.pending) return `<div class="qa"><div class="q">${esc(x.question)}</div><div class="a hint">thinking…</div></div>`;
  if (x.error) return `<div class="qa"><div class="q">${esc(x.question)}</div><div class="note warn">${esc(x.error)}</div></div>`;
  return `<div class="qa"><div class="q">${esc(x.question)}</div><div class="a">${prettyText(x.answer)}</div>` +
    `<div class="ex-src">read as “${esc(x.understood_as)}” (${esc(x.parsed_by)}) · answer: ${esc(x.source)}` +
    `${x.passed_checks ? " · <span class='ok'>checked</span>" : " · <span class='bad'>" + esc(x.problems.join("; ")) + "</span>"}${x.note ? " · " + esc(x.note) : ""}</div></div>`;
}
function askBoxHtml(rec) {
  const log = (state.dialogs[state.recIndex] || []).map(qaHtml).join("");
  return `<div class="chips">${suggestions(rec).map((q) => `<button class="chip" data-q="${esc(q)}">${prettyText(q)}</button>`).join("")}</div>` +
    `<form id="ask-form" class="ask-form"><input id="ask-input" autocomplete="off" placeholder="why not 3NT? · what if I had the ♠A instead of the ♦8?"><button class="btn" type="submit">Ask</button></form>` +
    `<div id="ask-log">${log}</div>`;
}
async function askQuestion(question) {
  question = (question || "").trim();
  if (!question || state.recIndex === null) return;
  if (state.tab !== "ask" && state.record) { state.tab = "ask"; renderAdvisor(state.record); }
  const i = state.recIndex, list = (state.dialogs[i] = state.dialogs[i] || []);
  const entry = { question, pending: true };
  list.push(entry);
  const redraw = () => { if (state.recIndex === i && $("ask-log")) $("ask-log").innerHTML = list.map(qaHtml).join(""); };
  redraw();
  if ($("ask-input")) $("ask-input").value = "";
  try { Object.assign(entry, await api("POST", `/api/game/${state.id}/record/${i}/ask`, { question, llm: true }), { pending: false }); }
  catch (e) { Object.assign(entry, { pending: false, error: e.message }); }
  redraw();
  pollLLM();
}
function llmHtml(l) {
  return `<div class="sub">In Qwen's words</div><div class="ex-text" style="font-size:16px">${prettyText(l.text)}</div>` +
         `<div class="ex-src">${l.passed_checks ? "<span class='ok'>checked</span>" : "<span class='bad'>failed a check</span>"}${l.attempts > 1 ? ` after ${l.attempts} attempts` : ""}</div>`;
}
async function askLLM(rec) {
  const note = $("llm-note"), btn = $("btn-llm");
  btn.disabled = true;
  note.textContent = state.llm && state.llm.loaded ? "writing…" : "loading Qwen (about a minute the first time)…";
  try { rec.llm = await api("POST", `/api/game/${state.id}/record/${state.recIndex}/llm`); if ($("llm-box")) $("llm-box").innerHTML = llmHtml(rec.llm); }
  catch (e) { note.textContent = e.message; btn.disabled = false; }
  pollLLM();
}

// ------------------------------------------------------------------ the other models
async function loadCompare() {
  const el = $("compare");
  if (!el) return;
  try {
    const r = await api("GET", `/api/game/${state.id}/record/${state.recIndex}/compare`);
    if (!$("compare")) return;
    const cur = state.record.model_id || modelIdOf(state.record);
    $("compare").classList.remove("hint");
    $("compare").innerHTML = r.models.map((m) =>
      `<div class="cmp-row ${m.id === cur ? "cur" : ""}"><div>${esc(m.label)}${m.kind === "upstream" ? '<span class="tag">original brl</span>' : ""}` +
      (m.id === cur ? "" : `<button class="btn link use" data-use="${m.id}">explain</button>`) + `</div>${tile(m.call)}<div class="val">${pct(m.p)}</div></div>`).join("");
    $("compare").querySelectorAll("[data-use]").forEach((b) => { b.onclick = () => switchExplainer(b.dataset.use); });
  } catch (e) { el.textContent = e.message; }
}

// ?shot=MS: hold the page's load event MS milliseconds (a delayed image), so a
// headless browser's --screenshot captures the table after the API calls finished
{ const ms = new URLSearchParams(location.search).get("shot");
  if (ms) { const img = new Image(); img.src = `/api/wait?ms=${Number(ms)}`; img.style.display = "none"; document.body.appendChild(img); } }
init().catch((e) => status(e.message));
