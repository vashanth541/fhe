/**
 * charts.js — Lightweight Canvas-based Charts
 * No external dependencies — pure Canvas 2D API
 */

'use strict';

const Charts = (() => {

  const COLORS = {
    cyan:   '#00c8ff',
    amber:  '#ffb700',
    green:  '#00ff88',
    purple: '#a855f7',
    red:    '#ff3c5c',
    bg:     '#040c14',
    bgCard: '#0b1f33',
    border: 'rgba(0,200,255,0.12)',
    text:   '#7aa8c4',
    textLt: '#e0f4ff',
  };

  function setupCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width  = rect.width  * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    return { ctx, w: rect.width, h: rect.height };
  }

  function drawGrid(ctx, x0, y0, w, h, steps = 5) {
    ctx.strokeStyle = COLORS.border;
    ctx.lineWidth   = 0.5;
    for (let i = 0; i <= steps; i++) {
      const y = y0 + (h / steps) * i;
      ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x0 + w, y); ctx.stroke();
    }
  }

  function drawAxes(ctx, x0, y0, w, h) {
    ctx.strokeStyle = 'rgba(0,200,255,0.25)';
    ctx.lineWidth   = 1;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, y0 + h); ctx.lineTo(x0 + w, y0 + h); ctx.stroke();
  }

  function label(ctx, text, x, y, color = COLORS.text, size = 10, align = 'center') {
    ctx.fillStyle  = color;
    ctx.font       = `${size}px 'Share Tech Mono', monospace`;
    ctx.textAlign  = align;
    ctx.fillText(text, x, y);
  }

  /* ── Bar Chart ── */
  function barChart(canvas, { labels, values, colors, title, unit = 'ms' }) {
    const { ctx, w, h } = setupCanvas(canvas);
    const pad = { top: 28, right: 16, bottom: 36, left: 44 };
    const gw  = w - pad.left - pad.right;
    const gh  = h - pad.top  - pad.bottom;
    const x0  = pad.left, y0 = pad.top;

    // Background
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, w, h);

    // Title
    label(ctx, title || '', w / 2, 18, COLORS.cyan, 11);

    const maxVal  = Math.max(...values, 1);
    const barW    = gw / labels.length;
    const barPad  = barW * 0.2;

    drawGrid(ctx, x0, y0, gw, gh);
    drawAxes(ctx, x0, y0, gw, gh);

    values.forEach((val, i) => {
      const bh   = (val / maxVal) * gh;
      const bx   = x0 + i * barW + barPad;
      const by   = y0 + gh - bh;
      const bw   = barW - barPad * 2;
      const col  = colors ? colors[i] : COLORS.cyan;

      // Bar glow
      ctx.shadowColor = col;
      ctx.shadowBlur  = 8;
      const grad = ctx.createLinearGradient(bx, by + bh, bx, by);
      grad.addColorStop(0, col + '22');
      grad.addColorStop(1, col);
      ctx.fillStyle = grad;
      ctx.fillRect(bx, by, bw, bh);
      ctx.shadowBlur = 0;

      // Value on top
      label(ctx, `${val}${unit}`, bx + bw / 2, by - 4, col, 9);

      // Label
      const lbl = labels[i].length > 8 ? labels[i].slice(0, 7) + '…' : labels[i];
      label(ctx, lbl, bx + bw / 2, y0 + gh + 18, COLORS.text, 8);
    });

    // Y-axis labels
    for (let i = 0; i <= 4; i++) {
      const v = Math.round((maxVal / 4) * (4 - i));
      label(ctx, v, x0 - 6, y0 + (gh / 4) * i + 4, COLORS.text, 8, 'right');
    }
  }

  /* ── Line Chart ── */
  function lineChart(canvas, { labels, datasets, title }) {
    const { ctx, w, h } = setupCanvas(canvas);
    const pad = { top: 28, right: 16, bottom: 36, left: 44 };
    const gw  = w - pad.left - pad.right;
    const gh  = h - pad.top  - pad.bottom;
    const x0  = pad.left, y0 = pad.top;

    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, w, h);

    label(ctx, title || '', w / 2, 18, COLORS.cyan, 11);

    const allVals = datasets.flatMap(d => d.data);
    const maxVal  = Math.max(...allVals, 1);
    const minVal  = Math.min(...allVals, 0);
    const range   = maxVal - minVal || 1;

    drawGrid(ctx, x0, y0, gw, gh);
    drawAxes(ctx, x0, y0, gw, gh);

    datasets.forEach(ds => {
      const pts = ds.data.map((v, i) => ({
        x: x0 + (i / (ds.data.length - 1)) * gw,
        y: y0 + gh - ((v - minVal) / range) * gh,
      }));

      // Gradient fill
      const grad = ctx.createLinearGradient(0, y0, 0, y0 + gh);
      grad.addColorStop(0, ds.color + '44');
      grad.addColorStop(1, ds.color + '00');
      ctx.beginPath();
      ctx.moveTo(pts[0].x, y0 + gh);
      pts.forEach(p => ctx.lineTo(p.x, p.y));
      ctx.lineTo(pts[pts.length - 1].x, y0 + gh);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();

      // Line
      ctx.beginPath();
      ctx.strokeStyle = ds.color;
      ctx.lineWidth   = 2;
      ctx.shadowColor = ds.color;
      ctx.shadowBlur  = 6;
      pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Dots
      pts.forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
        ctx.fillStyle = ds.color;
        ctx.fill();
      });
    });

    // X labels
    labels.forEach((lbl, i) => {
      const x = x0 + (i / (labels.length - 1)) * gw;
      const l = lbl.length > 6 ? lbl.slice(0, 5) + '…' : lbl;
      label(ctx, l, x, y0 + gh + 18, COLORS.text, 8);
    });

    // Y labels
    for (let i = 0; i <= 4; i++) {
      const v = (minVal + (range / 4) * (4 - i)).toFixed(0);
      label(ctx, v, x0 - 6, y0 + (gh / 4) * i + 4, COLORS.text, 8, 'right');
    }
  }

  /* ── Donut/Progress Chart ── */
  function noiseGauge(canvas, { value, max, title }) {
    const { ctx, w, h } = setupCanvas(canvas);
    const cx = w / 2, cy = h * 0.55, r = Math.min(w, h) * 0.34;
    const frac = Math.min(1, value / max);

    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, w, h);

    label(ctx, title || '', cx, 18, COLORS.cyan, 11);

    // Background arc
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI * 0.75, Math.PI * 2.25);
    ctx.strokeStyle = COLORS.border;
    ctx.lineWidth   = 14;
    ctx.lineCap     = 'round';
    ctx.stroke();

    // Filled arc (noise consumed)
    if (frac > 0) {
      const col = frac < 0.4 ? COLORS.green : frac < 0.7 ? COLORS.amber : COLORS.red;
      const endAngle = Math.PI * 0.75 + frac * Math.PI * 1.5;
      ctx.beginPath();
      ctx.arc(cx, cy, r, Math.PI * 0.75, endAngle);
      ctx.strokeStyle = col;
      ctx.lineWidth   = 14;
      ctx.shadowColor = col;
      ctx.shadowBlur  = 10;
      ctx.stroke();
      ctx.shadowBlur  = 0;
    }

    // Center text
    label(ctx, `${Math.round(frac * 100)}%`, cx, cy + 8, COLORS.textLt, 22, 'center');
    label(ctx, 'consumed', cx, cy + 24, COLORS.text, 9, 'center');

    // Labels
    label(ctx, '0%',  cx - r - 4, cy + r * 0.45, COLORS.text, 8, 'right');
    label(ctx, '100%', cx + r + 4, cy + r * 0.45, COLORS.text, 8, 'left');
  }

  return { barChart, lineChart, noiseGauge };
})();

window.Charts = Charts;
