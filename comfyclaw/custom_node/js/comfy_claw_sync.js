/**
 * ComfyClaw Sync Extension  v2.0
 *
 * Connects to the ComfyClaw Python sync server (ws://127.0.0.1:8765 by default)
 * and reloads the ComfyUI canvas in real time whenever the agent modifies the
 * workflow topology.
 *
 * Protocol — the Python SyncServer sends two message types:
 *
 *   Full snapshot (initial load / reconnect):
 *   { "type": "workflow_update", "workflow": { "<nodeId>": { class_type, inputs, … }, … } }
 *
 *   Incremental diff (subsequent mutations):
 *   { "type": "workflow_diff", "ops": [ {op, id, data?}, … ], "full": {…} }
 *
 * Configuration (persisted in localStorage):
 *   localStorage.setItem('comfyclaw_ws_url', 'ws://127.0.0.1:8765');
 *   localStorage.setItem('comfyclaw_op_delay', '400');   // ms between ops
 *
 * Status badge:
 *   🔄 connecting  |  🟢 live  |  ✨ updated (flashes 2 s)  |  🔴 disconnected
 */

import { app } from "../../scripts/app.js";

const DEFAULT_WS_URL         = `ws://${window.location.hostname}:8765`;
const RECONNECT_DELAY_MS     = 3000;
const MAX_RECONNECT_ATTEMPTS = 20;
const DEFAULT_OP_DELAY_MS    = 400;

const NODE_W = 220;
const NODE_H = 180;
const GAP_X  = 60;
const GAP_Y  = 40;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function getOpDelay() {
  const stored = localStorage.getItem("comfyclaw_op_delay");
  if (stored !== null) {
    const n = parseInt(stored, 10);
    if (!isNaN(n) && n >= 0) return n;
  }
  return DEFAULT_OP_DELAY_MS;
}

// ─────────────────────────────────────────────────────────────────────────────
// Status badge
// ─────────────────────────────────────────────────────────────────────────────

let statusEl = null;

function createStatusBadge() {
  const el = document.createElement("span");
  el.id = "comfyclaw-status";
  el.title = "ComfyClaw Sync — click to reconfigure URL";
  Object.assign(el.style, {
    position:   "fixed",
    bottom:     "12px",
    right:      "12px",
    zIndex:     "9999",
    padding:    "4px 10px",
    borderRadius: "12px",
    fontSize:   "12px",
    fontFamily: "monospace",
    fontWeight: "bold",
    cursor:     "pointer",
    userSelect: "none",
    transition: "background 0.3s",
  });
  el.addEventListener("click", promptConfig);
  document.body.appendChild(el);
  return el;
}

const STATUS = {
  connecting:   { bg: "#555",    fg: "#fff", label: "🔄 ComfyClaw: connecting…"    },
  connected:    { bg: "#1a7a3f", fg: "#fff", label: "🟢 ComfyClaw: live"           },
  disconnected: { bg: "#7a1a1a", fg: "#fff", label: "🔴 ComfyClaw: disconnected"   },
  updated:      { bg: "#1a4a7a", fg: "#fff", label: "✨ ComfyClaw: graph updated"  },
};

function setStatus(state, extra) {
  if (!statusEl) return;
  const s = STATUS[state] || STATUS.disconnected;
  statusEl.style.background = s.bg;
  statusEl.style.color = s.fg;
  statusEl.textContent = extra ? `${s.label} — ${extra}` : s.label;
  if (state === "updated") {
    setTimeout(() => setStatus("connected"), 2000);
  }
}

function promptConfig() {
  const current = localStorage.getItem("comfyclaw_ws_url") || DEFAULT_WS_URL;
  const val = window.prompt("ComfyClaw WebSocket URL:", current);
  if (val !== null) {
    localStorage.setItem("comfyclaw_ws_url", val.trim());
    window.location.reload();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// API-format detection & conversion (for full-reload fallback)
// ─────────────────────────────────────────────────────────────────────────────

function isApiFormat(data) {
  if (typeof data !== "object" || data === null || Array.isArray(data)) return false;
  const keys = Object.keys(data);
  if (keys.length === 0) return false;
  return keys.every(k => /^\d+$/.test(k) && data[k] && data[k].class_type);
}

function apiToLitegraph(apiWf) {
  const nodes  = [];
  const links  = [];
  let linkCounter = 0;
  const linkMap   = {};

  const ids  = Object.keys(apiWf).sort((a, b) => parseInt(a) - parseInt(b));
  const COLS = 5;

  const posMap = {};
  ids.forEach((nid, idx) => {
    const col = idx % COLS;
    const row = Math.floor(idx / COLS);
    posMap[nid] = [col * (NODE_W + GAP_X) + 60, row * (NODE_H + GAP_Y) + 60];
  });

  ids.forEach(nid => {
    const apiNode       = apiWf[nid];
    const inputs_meta   = [];
    const widgets_values = [];

    for (const [key, val] of Object.entries(apiNode.inputs || {})) {
      if (Array.isArray(val) && val.length === 2 && typeof val[0] === "string") {
        const [srcId, srcIdx] = val;
        const linkKey = `${srcId}:${srcIdx}`;
        let lid;
        if (linkMap[linkKey] !== undefined) {
          lid = linkMap[linkKey];
        } else {
          lid = linkCounter++;
          linkMap[linkKey] = lid;
          links.push([lid, parseInt(srcId), srcIdx, parseInt(nid), inputs_meta.length, "*"]);
        }
        inputs_meta.push({ name: key, type: "*", link: lid });
      } else {
        widgets_values.push(val);
      }
    }

    nodes.push({
      id:             parseInt(nid),
      type:           apiNode.class_type,
      pos:            posMap[nid],
      size:           [NODE_W, NODE_H],
      flags:          {},
      order:          parseInt(nid),
      mode:           0,
      inputs:         inputs_meta,
      outputs:        [],
      title:          apiNode._meta?.title || apiNode.class_type,
      properties:     { "Node name for S&R": apiNode.class_type },
      widgets_values,
    });
  });

  const maxId = ids.reduce((m, k) => Math.max(m, parseInt(k)), 0);
  return {
    last_node_id:  maxId,
    last_link_id:  linkCounter - 1,
    nodes, links,
    groups:  [],
    config:  {},
    extra:   { comfyclaw: true },
    version: 0.4,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Full workflow loading (used for initial load / reconnect)
// ─────────────────────────────────────────────────────────────────────────────

async function loadWorkflowIntoCanvas(data) {
  try {
    if (isApiFormat(data) && typeof app.loadApiJson === "function") {
      await app.loadApiJson(data);
      console.log("[ComfyClaw] Loaded via app.loadApiJson");
      return true;
    }

    const graphData = isApiFormat(data) ? apiToLitegraph(data) : data;
    if (typeof app.loadGraphData === "function") {
      await app.loadGraphData(graphData);
      console.log("[ComfyClaw] Loaded via app.loadGraphData");
      return true;
    }

    if (app.graph && typeof app.graph.configure === "function") {
      app.graph.configure(isApiFormat(data) ? apiToLitegraph(data) : data);
      app.graph.setDirtyCanvas?.(true, true);
      console.log("[ComfyClaw] Loaded via app.graph.configure");
      return true;
    }

    console.warn("[ComfyClaw] No suitable canvas load method found.");
    return false;
  } catch (err) {
    console.error("[ComfyClaw] Error loading workflow into canvas:", err);
    return false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Incremental diff application — node-by-node canvas updates
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Accumulated API-format workflow the client knows about.
 * Updated on every op so we can reload the full graph at each step.
 */
let _currentApiWorkflow = {};

/**
 * Temporarily highlight a node using LiteGraph's native color system.
 */
function highlightNode(nodeId, durationMs = 1500) {
  const lgNode = app.graph?.getNodeById(parseInt(nodeId));
  if (!lgNode) return;

  const origColor   = lgNode.color;
  const origBgcolor = lgNode.bgcolor;

  lgNode.color   = "#4a9eff";
  lgNode.bgcolor = "#1a3a5a";
  app.graph?.setDirtyCanvas?.(true, true);

  setTimeout(() => {
    lgNode.color   = origColor;
    lgNode.bgcolor = origBgcolor;
    app.graph?.setDirtyCanvas?.(true, true);
  }, durationMs);
}

/**
 * Apply a single diff op:
 *  1. Update ``_currentApiWorkflow`` (the accumulated state).
 *  2. Reload the full graph via ComfyUI's native loader (handles layout).
 *  3. Highlight the affected node so the user can see what changed.
 */
async function applyOp(op) {
  switch (op.op) {
    case "add_node":
      _currentApiWorkflow[op.id] = op.data;
      await loadWorkflowIntoCanvas(_currentApiWorkflow);
      highlightNode(op.id);
      console.log(`[ComfyClaw] +node ${op.id} (${op.data.class_type})`);
      break;

    case "remove_node":
      delete _currentApiWorkflow[op.id];
      await loadWorkflowIntoCanvas(_currentApiWorkflow);
      console.log(`[ComfyClaw] -node ${op.id}`);
      break;

    case "update_node":
      _currentApiWorkflow[op.id] = op.data;
      await loadWorkflowIntoCanvas(_currentApiWorkflow);
      highlightNode(op.id, 800);
      console.log(`[ComfyClaw] ~node ${op.id} (updated)`);
      break;

    default:
      console.warn(`[ComfyClaw] Unknown op: ${op.op}`);
  }
}

/**
 * Process an array of diff ops sequentially with a delay between each op
 * for a smooth visual build-up effect.
 */
async function applyDiffOps(ops) {
  const delayMs = getOpDelay();
  for (let i = 0; i < ops.length; i++) {
    await applyOp(ops[i]);
    if (delayMs > 0 && i < ops.length - 1) {
      await sleep(delayMs);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// WebSocket client with auto-reconnect
// ─────────────────────────────────────────────────────────────────────────────

class SyncClient {
  constructor() {
    this.ws = null;
    this.reconnectAttempts = 0;
    this.destroyed = false;
    this._processing = false;
    this._queue = [];
  }

  connect() {
    const url = localStorage.getItem("comfyclaw_ws_url") || DEFAULT_WS_URL;
    setStatus("connecting");
    try {
      this.ws = new WebSocket(url);
    } catch (err) {
      console.warn("[ComfyClaw] WebSocket construction failed:", err);
      this._scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      console.log(`[ComfyClaw] Connected to ${url}`);
      this.reconnectAttempts = 0;
      setStatus("connected");
    };

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        this._queue.push(msg);
        this._processQueue();
      } catch (err) {
        console.error("[ComfyClaw] Message parse error:", err);
      }
    };

    this.ws.onerror = () => {};

    this.ws.onclose = () => {
      if (!this.destroyed) {
        setStatus("disconnected");
        this._scheduleReconnect();
      }
    };
  }

  async _processQueue() {
    if (this._processing) return;
    this._processing = true;
    try {
      while (this._queue.length > 0) {
        const msg = this._queue.shift();
        await this._handleMessage(msg);
        if (this._queue.length > 0 && msg.type === "workflow_diff") {
          await sleep(getOpDelay());
        }
      }
    } finally {
      this._processing = false;
    }
  }

  async _handleMessage(msg) {
    if (msg.type === "workflow_update" && msg.workflow) {
      _currentApiWorkflow = JSON.parse(JSON.stringify(msg.workflow));
      const nodeCount = Object.keys(msg.workflow).length;
      const ok = await loadWorkflowIntoCanvas(msg.workflow);
      if (ok) {
        setStatus("updated", `${nodeCount} nodes`);
      }
    } else if (msg.type === "workflow_diff" && Array.isArray(msg.ops)) {
      const addCount = msg.ops.filter(o => o.op === "add_node").length;
      const rmCount  = msg.ops.filter(o => o.op === "remove_node").length;
      const updCount = msg.ops.filter(o => o.op === "update_node").length;
      await applyDiffOps(msg.ops);
      const total = Object.keys(_currentApiWorkflow).length;
      const parts = [];
      if (addCount)  parts.push(`+${addCount}`);
      if (rmCount)   parts.push(`-${rmCount}`);
      if (updCount)  parts.push(`~${updCount}`);
      setStatus("updated", `${total} nodes (${parts.join(", ")})`);
    }
  }

  _scheduleReconnect() {
    if (this.destroyed) return;
    if (this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      console.warn("[ComfyClaw] Max reconnect attempts reached. Giving up.");
      setStatus("disconnected", "max retries");
      return;
    }
    this.reconnectAttempts++;
    setTimeout(() => this.connect(), RECONNECT_DELAY_MS);
  }

  destroy() {
    this.destroyed = true;
    this.ws?.close();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// ComfyUI extension registration
// ─────────────────────────────────────────────────────────────────────────────

app.registerExtension({
  name: "ComfyClaw.SyncBridge",

  async setup() {
    console.log("[ComfyClaw] Extension loaded — ComfyClaw Sync Bridge v2.0");
    statusEl = createStatusBadge();
    setTimeout(() => new SyncClient().connect(), 500);
  },
});
