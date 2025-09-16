import { app } from "../../scripts/app.js";

// Color math helpers
function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }
function hsvToRgb(h, s, v) {
  let c = v * s;
  let x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  let m = v - c;
  let r = 0, g = 0, b = 0;
  if (0 <= h && h < 60) { r = c; g = x; b = 0; }
  else if (60 <= h && h < 120) { r = x; g = c; b = 0; }
  else if (120 <= h && h < 180) { r = 0; g = c; b = x; }
  else if (180 <= h && h < 240) { r = 0; g = x; b = c; }
  else if (240 <= h && h < 300) { r = x; g = 0; b = c; }
  else { r = c; g = 0; b = x; }
  r = Math.round((r + m) * 255);
  g = Math.round((g + m) * 255);
  b = Math.round((b + m) * 255);
  return [r, g, b];
}
function rgbToHex(r, g, b) { return "#" + [r, g, b].map(v => v.toString(16).padStart(2, "0")).join("").toUpperCase(); }
function hexToRgb(hex) {
  const v = (hex || "").trim();
  const m = /^#?([0-9a-fA-F]{6})$/.exec(v);
  if (!m) return null;
  const intVal = parseInt(m[1], 16);
  const r = (intVal >> 16) & 255;
  const g = (intVal >> 8) & 255;
  const b = intVal & 255;
  return [r, g, b];
}
function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const d = max - min;
  let h = 0;
  if (d === 0) h = 0;
  else if (max === r) h = 60 * (((g - b) / d) % 6);
  else if (max === g) h = 60 * (((b - r) / d) + 2);
  else h = 60 * (((r - g) / d) + 4);
  if (h < 0) h += 360;
  const s = max === 0 ? 0 : d / max;
  const v = max;
  return [h, s, v];
}

// Convert RGB (0-255) to HSL. Returns [h, s%, l%]
function rgbToHsl(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  let h = 0, s = 0;
  const l = (max + min) / 2;
  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r:
        h = ((g - b) / d + (g < b ? 6 : 0));
        break;
      case g:
        h = (b - r) / d + 2; break;
      case b:
        h = (r - g) / d + 4; break;
    }
    h *= 60;
  }
  return [Math.round(h), Math.round(s * 100), Math.round(l * 100)];
}

// Convert HSL (h:0-360, s:0-1, l:0-1) to RGB (0-255)
function hslToRgb(h, s, l) {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let r1 = 0, g1 = 0, b1 = 0;
  if (0 <= h && h < 60) { r1 = c; g1 = x; b1 = 0; }
  else if (60 <= h && h < 120) { r1 = x; g1 = c; b1 = 0; }
  else if (120 <= h && h < 180) { r1 = 0; g1 = c; b1 = x; }
  else if (180 <= h && h < 240) { r1 = 0; g1 = x; b1 = c; }
  else if (240 <= h && h < 300) { r1 = x; g1 = 0; b1 = c; }
  else { r1 = c; g1 = 0; b1 = x; }
  const r = Math.round((r1 + m) * 255);
  const g = Math.round((g1 + m) * 255);
  const b = Math.round((b1 + m) * 255);
  return [r, g, b];
}

// Overlay palette builder and controller
const S4Palette = (() => {
  let overlayEl = null; // #palette element
  let panel, sv, hue, hexInput, rgbInputs, hslInputs, cancelBtn, okBtn;
  let state = { h: 0, s: 0, v: 1 };
  let commitCb = null;
  let visible = false;
  let apiObj = null;

  function ensure() {
    if (apiObj) return apiObj;
    overlayEl = document.createElement("div");
    overlayEl.id = "palette";
    overlayEl.style.position = "fixed";
    overlayEl.style.inset = "0";
    overlayEl.style.display = "none";
    overlayEl.style.zIndex = 10000;
    overlayEl.style.background = "rgba(0,0,0,0.45)";

    panel = document.createElement("div");
    panel.style.position = "absolute";
    panel.style.left = "50%";
    panel.style.top = "50%";
    panel.style.transform = "translate(-50%, -50%)";
    panel.style.minWidth = "320px";
    panel.style.padding = "14px";
    panel.style.background = "#1f1f1f";
    panel.style.border = "1px solid #3a3a3a";
    panel.style.borderRadius = "10px";
    panel.style.boxShadow = "0 12px 40px rgba(0,0,0,0.5)";
    panel.style.color = "#ddd";

    const title = document.createElement("div");
    title.textContent = "💀S4Color Palette";
    title.style.fontWeight = "600";
    title.style.marginBottom = "10px";

    sv = document.createElement("canvas");
    hue = document.createElement("canvas");
    sv.width = 300; sv.height = 170;
    hue.width = 300; hue.height = 16;
    sv.style.borderRadius = "8px";
    hue.style.borderRadius = "6px";
    sv.style.display = "block";
    hue.style.display = "block";
    sv.style.marginBottom = "10px";
    hue.style.margin = "8px 0 10px 0";

    // HEX row with label
    const hexRow = document.createElement("div");
    hexRow.style.display = "grid";
    hexRow.style.gridTemplateColumns = "56px 1fr";
    hexRow.style.gap = "8px";
    hexRow.style.marginTop = "6px";
    const hexLab = document.createElement("div");
    hexLab.textContent = "HEX";
    hexLab.style.alignSelf = "center";
    hexLab.style.fontSize = "12px";
    hexLab.style.opacity = "0.9";
    hexLab.style.minWidth = "56px";
    hexLab.style.width = "56px";
    hexInput = document.createElement("input");
    hexInput.type = "text";
    hexInput.placeholder = "#FFFFFF";
    hexInput.style.width = "100%";
    hexInput.style.padding = "6px 8px";
    hexInput.style.border = "1px solid #444";
    hexInput.style.borderRadius = "6px";
    hexInput.style.background = "#111";
    hexInput.style.color = "#fff";
    hexInput.style.boxSizing = "border-box";
    hexRow.appendChild(hexLab);
    hexRow.appendChild(hexInput);

    // RGB row (three inputs)
    const rgbRow = document.createElement("div");
    rgbRow.style.display = "grid";
    rgbRow.style.gridTemplateColumns = "56px 1fr 1fr 1fr";
    rgbRow.style.gap = "6px";
    rgbRow.style.marginTop = "6px";
    const rgbLab = document.createElement("div");
    rgbLab.textContent = "RGB";
    rgbLab.style.alignSelf = "center";
    rgbLab.style.fontSize = "12px";
    rgbLab.style.opacity = "0.9";
    rgbLab.style.minWidth = "56px";
    rgbLab.style.width = "56px";
    const rIn = document.createElement("input");
    const gIn = document.createElement("input");
    const bIn = document.createElement("input");
    [rIn, gIn, bIn].forEach((el, i) => {
      el.type = "number"; el.min = "0"; el.max = "255";
      el.placeholder = ["R", "G", "B"][i];
      el.style.padding = "6px 8px";
      el.style.border = "1px solid #444";
      el.style.borderRadius = "6px";
      el.style.background = "#111";
      el.style.color = "#fff";
      el.style.width = "100%";
      el.style.boxSizing = "border-box";
    });
    rgbRow.appendChild(rgbLab);
    rgbRow.appendChild(rIn); rgbRow.appendChild(gIn); rgbRow.appendChild(bIn);
    rgbInputs = { rIn, gIn, bIn };

    // HSL row (three inputs)
    const hslRow = document.createElement("div");
    hslRow.style.display = "grid";
    hslRow.style.gridTemplateColumns = "56px 1fr 1fr 1fr";
    hslRow.style.gap = "6px";
    hslRow.style.marginTop = "6px";
    const hslLab = document.createElement("div");
    hslLab.textContent = "HSL";
    hslLab.style.alignSelf = "center";
    hslLab.style.fontSize = "12px";
    hslLab.style.opacity = "0.9";
    hslLab.style.minWidth = "56px";
    hslLab.style.width = "56px";
    const hIn = document.createElement("input");
    const sIn = document.createElement("input");
    const lIn = document.createElement("input");
    hIn.type = "number"; hIn.min = "0"; hIn.max = "360"; hIn.placeholder = "H";
    sIn.type = "number"; sIn.min = "0"; sIn.max = "100"; sIn.placeholder = "S%";
    lIn.type = "number"; lIn.min = "0"; lIn.max = "100"; lIn.placeholder = "L%";
    [hIn, sIn, lIn].forEach((el) => {
      el.style.padding = "6px 8px";
      el.style.border = "1px solid #444";
      el.style.borderRadius = "6px";
      el.style.background = "#111";
      el.style.color = "#fff";
      el.style.width = "100%";
      el.style.boxSizing = "border-box";
    });
    hslRow.appendChild(hslLab);
    hslRow.appendChild(hIn); hslRow.appendChild(sIn); hslRow.appendChild(lIn);
    hslInputs = { hIn, sIn, lIn };

    const actions = document.createElement("div");
    actions.style.display = "flex";
    actions.style.gap = "8px";
    actions.style.marginTop = "8px";
    cancelBtn = document.createElement("button");
    cancelBtn.textContent = "Cancel";
    cancelBtn.style.padding = "6px 12px";
    cancelBtn.style.border = "1px solid #555";
    cancelBtn.style.background = "#222";
    cancelBtn.style.color = "#ddd";
    cancelBtn.style.borderRadius = "6px";
    okBtn = document.createElement("button");
    okBtn.textContent = "OK";
    okBtn.style.padding = "6px 12px";
    okBtn.style.border = "1px solid #5a8";
    okBtn.style.background = "#2b2";
    okBtn.style.color = "#fff";
    okBtn.style.borderRadius = "6px";
    actions.appendChild(cancelBtn);
    actions.appendChild(okBtn);

    // Wrap content to enforce same width as canvases
    const contentWrap = document.createElement("div");
    contentWrap.style.width = `${sv.width}px`;
    contentWrap.style.margin = "0 auto";
    contentWrap.style.boxSizing = "border-box";
    contentWrap.appendChild(sv);
    contentWrap.appendChild(hue);
    contentWrap.appendChild(hexRow);
    contentWrap.appendChild(rgbRow);
    contentWrap.appendChild(hslRow);

    // Center actions and match width
    actions.style.justifyContent = "center";
    actions.style.maxWidth = `${sv.width}px`;
    actions.style.marginLeft = "auto";
    actions.style.marginRight = "auto";

    panel.appendChild(title);
    panel.appendChild(contentWrap);
    panel.appendChild(actions);
    overlayEl.appendChild(panel);
    document.body.appendChild(overlayEl);

    // Draw helpers
    function drawHue() {
      const ctx = hue.getContext("2d");
      const w = hue.width, h = hue.height;
      const grd = ctx.createLinearGradient(0, 0, w, 0);
      for (let i = 0; i <= 360; i += 60) {
        const [r, g, b] = hsvToRgb(i, 1, 1);
        grd.addColorStop(i / 360, `rgb(${r},${g},${b})`);
      }
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = grd;
      ctx.fillRect(0, 0, w, h);
      // Draw selected hue marker (ring) similar to OS pickers
      const cx = clamp((state.h / 360) * w, 0, w);
      const cy = h / 2;
      const radius = Math.max(5, cy - 2);
      const [mr, mg, mb] = hsvToRgb(state.h, 1, 1);
      ctx.save();
      ctx.fillStyle = `rgb(${mr},${mg},${mb})`;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(cx, cy, radius + 1.5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
    function drawSV() {
      const ctx = sv.getContext("2d");
      const w = sv.width, h = sv.height;
      const [r, g, b] = hsvToRgb(state.h, 1, 1);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(0, 0, w, h);
      const grdWhite = ctx.createLinearGradient(0, 0, w, 0);
      grdWhite.addColorStop(0, "#fff");
      grdWhite.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = grdWhite;
      ctx.fillRect(0, 0, w, h);
      const grdBlack = ctx.createLinearGradient(0, 0, 0, h);
      grdBlack.addColorStop(0, "rgba(0,0,0,0)");
      grdBlack.addColorStop(1, "#000");
      ctx.fillStyle = grdBlack;
      ctx.fillRect(0, 0, w, h);
      // crosshair
      const x = state.s * w; const y = (1 - state.v) * h;
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI*2); ctx.stroke();
      ctx.strokeStyle = "#000"; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(x, y, 7.5, 0, Math.PI*2); ctx.stroke();
    }

    function updateUI() {
      const [r, g, b] = hsvToRgb(state.h, state.s, state.v);
      const hex = rgbToHex(r, g, b);
      hexInput.value = hex;
      rgbInputs.rIn.value = String(r);
      rgbInputs.gIn.value = String(g);
      rgbInputs.bIn.value = String(b);
      const [hh, ss, ll] = rgbToHsl(r, g, b);
      hslInputs.hIn.value = String(hh);
      hslInputs.sIn.value = String(ss);
      hslInputs.lIn.value = String(ll);
      return hex;
    }

    // Interactions
    hue.addEventListener("mousedown", (e) => {
      const rect = hue.getBoundingClientRect();
      const move = (ev) => {
        const x = clamp(ev.clientX - rect.left, 0, rect.width);
        state.h = (x / rect.width) * 360;
        drawHue();
        drawSV();
        updateUI();
      };
      const up = () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
      move(e);
    });
    sv.addEventListener("mousedown", (e) => {
      const rect = sv.getBoundingClientRect();
      const move = (ev) => {
        const x = clamp(ev.clientX - rect.left, 0, rect.width);
        const y = clamp(ev.clientY - rect.top, 0, rect.height);
        state.s = x / rect.width; state.v = 1 - (y / rect.height);
        drawSV();
        updateUI();
      };
      const up = () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
      move(e);
    });
    hexInput.addEventListener("input", () => {
      const rgb = hexToRgb(hexInput.value);
      if (!rgb) return;
      const [h, s, v] = rgbToHsv(rgb[0], rgb[1], rgb[2]);
      state = { h, s, v };
      drawHue(); drawSV(); updateUI();
    });

    // Sync from RGB inputs -> state
    [rIn, gIn, bIn].forEach((el) => {
      el.addEventListener("input", () => {
        const r = clamp(parseInt(rIn.value || "0", 10), 0, 255);
        const g = clamp(parseInt(gIn.value || "0", 10), 0, 255);
        const b = clamp(parseInt(bIn.value || "0", 10), 0, 255);
        const [h, s, v] = rgbToHsv(r, g, b);
        state = { h, s, v };
        drawHue(); drawSV(); updateUI();
      });
    });

    // Sync from HSL inputs -> state
    [hIn, sIn, lIn].forEach((el) => {
      el.addEventListener("input", () => {
        const h = clamp(parseInt(hIn.value || "0", 10), 0, 360);
        const s = clamp(parseInt(sIn.value || "0", 10), 0, 100) / 100;
        const l = clamp(parseInt(lIn.value || "0", 10), 0, 100) / 100;
        const [r, g, b] = hslToRgb(h, s, l);
        const [hh, ss, vv] = rgbToHsv(r, g, b);
        state = { h: hh, s: ss, v: vv };
        drawHue(); drawSV(); updateUI();
      });
    });
    cancelBtn.addEventListener("click", () => { hide(); });
    okBtn.addEventListener("click", () => {
      const hex = updateUI();
      if (typeof commitCb === "function") commitCb(hex);
      hide();
    });

    function show(initialHex, onCommit) {
      commitCb = onCommit;
      const rgb = hexToRgb(initialHex) || [255,255,255];
      const [h, s, v] = rgbToHsv(rgb[0], rgb[1], rgb[2]);
      state = { h, s, v };
      drawHue(); drawSV(); updateUI();
      overlayEl.style.display = "block";
      visible = true;
    }
    function hide() { overlayEl.style.display = "none"; visible = false; }

    overlayEl.addEventListener("click", (e) => { if (e.target === overlayEl) hide(); });

    apiObj = { show, hide };
    return apiObj;
  }

  return {
    show(initialHex, onCommit) { const api = ensure(); return api.show(initialHex, onCommit); },
    hide() { const api = ensure(); return api.hide(); },
    isOpen() { return visible; }
  };
})();

app.registerExtension({
  name: "com.s4color.ui",
  async setup() {},
  nodeCreated(node) {
    if (node?.comfyClass !== "ImageColorPicker") return;
    const hexWidget = node.widgets?.find((w) => w.name === "hex");

    // Ensure node height can accommodate preview area at default size
    (function ensureNodeHeight() {
      try {
        const margin = 12;
        const header = 26;
        const perWidget = 26;
        const widgetsCount = (node.widgets && node.widgets.length) ? node.widgets.length : 0;
        const desiredPreview = 140;
        const baseline = header + widgetsCount * perWidget + margin + desiredPreview + margin;
        if (node.size[1] < baseline) node.size[1] = baseline;
      } catch {}
    })();

    function resolveLinkedHex() {
      let val = (hexWidget?.value || "#FFFFFF").toString();
      try {
        const input = node.inputs?.find((i) => i && i.name === "hex" && typeof i.link === "number");
        if (input && node.graph && node.graph.links) {
          const link = node.graph.links[input.link];
          if (link) {
            const origin = node.graph.getNodeById(link.origin_id);
            if (origin) {
              // Try to read first string-like widget value from origin
              const w = origin.widgets?.find((w) => typeof w?.value === "string");
              if (w && typeof w.value === "string") val = w.value;
            }
          }
        }
      } catch {}
      return val;
    }

    // Draw preview area inside node, adaptive to node size
    const prevOnDraw = node.onDrawBackground;
    node.onDrawBackground = function(ctx) {
      try {
        const margin = 12;
        const widgetsCount = (this.widgets && this.widgets.length) ? this.widgets.length : 0;
        const header = 26;
        const perWidget = 26;
        const x = margin;
        const y = header + widgetsCount * perWidget + margin;
        const w = this.size[0] - margin * 2;
        const availableH = this.size[1] - y - margin;
        const isCollapsed = !!(this.flags && this.flags.collapsed);
        if (isCollapsed || availableH <= 4 || w <= 4) {
          node.__s4_previewRect = null;
        } else {
          const h = availableH; // draw strictly inside node bounds
          node.__s4_previewRect = { x, y, w, h };
          const value = resolveLinkedHex();
          ctx.save();
          ctx.fillStyle = /^#[0-9A-F]{6}$/i.test(value) ? value : "#FFFFFF";
          ctx.fillRect(x, y, w, h);
          ctx.restore();
        }
      } catch {}
      if (prevOnDraw) return prevOnDraw.apply(this, arguments);
    };

    // Open overlay when clicking inside preview area
    const prevMouseDown = node.onMouseDown;
    node.onMouseDown = function(e, localPos) {
      try {
        const rect = node.__s4_previewRect;
        if (!rect) return prevMouseDown ? prevMouseDown.apply(this, arguments) : undefined;
        let lx, ly;
        if (localPos) { lx = localPos[0]; ly = localPos[1]; }
        else {
          const ds = app?.canvas?.ds || { scale: 1, offset: [0, 0] };
          const scale = ds.scale || 1; const off = ds.offset || [0, 0];
          lx = (e.clientX - off[0]) / scale - this.pos[0];
          ly = (e.clientY - off[1]) / scale - this.pos[1];
        }
        if (rect && lx >= rect.x && lx <= rect.x + rect.w && ly >= rect.y && ly <= rect.y + rect.h) {
          const current = resolveLinkedHex();
          // Always hide then reopen in next tick to reset DOM state
          if (S4Palette.isOpen()) {
            S4Palette.hide();
            setTimeout(() => {
              S4Palette.show(current, (hex) => applyHex(hex));
            }, 0);
          } else {
            S4Palette.show(current, (hex) => applyHex(hex));
          }
          return true;
        }
      } catch {}
      if (prevMouseDown) return prevMouseDown.apply(this, arguments);
    };

    function applyHex(hex) {
      if (hexWidget) {
        hexWidget.value = hex;
        if (typeof hexWidget.callback === "function") { try { hexWidget.callback(hex); } catch {} }
      }
      if (node?.graph) node.graph.setDirtyCanvas(true, true);
    }

    // When connections change on input 'hex', refresh preview immediately
    const prevConnChange = node.onConnectionsChange;
    node.onConnectionsChange = function(type, slot, wasConnected, link_info, input) {
      try {
        if (type === LiteGraph.INPUT) {
          if (node?.graph) node.graph.setDirtyCanvas(true, true);
        }
      } catch {}
      if (prevConnChange) return prevConnChange.apply(this, arguments);
    };

    // Hide palette when node collapses or is removed
    const prevCollapse = node.onCollapse;
    node.onCollapse = function() {
      try { if (S4Palette.isOpen()) S4Palette.hide(); } catch {}
      if (prevCollapse) return prevCollapse.apply(this, arguments);
    };

    const prevRemoved = node.onRemoved;
    node.onRemoved = function() {
      try { if (S4Palette.isOpen()) S4Palette.hide(); } catch {}
      if (prevRemoved) return prevRemoved.apply(this, arguments);
    };
  }
});


