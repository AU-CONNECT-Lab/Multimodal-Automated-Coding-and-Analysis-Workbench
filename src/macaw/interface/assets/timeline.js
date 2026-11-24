// The time strip under a player: play, the time, speed, zoom, and lanes for the analysed part, segments or marks, listed
// moments, and lanes a page adds (stacked channels, lines and coded spans).
(function () {
  const STEPS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
  const strips = new Map();
  // each strip's zoom by its id, so one drawn again for new settings keeps where it was
  const views = new Map();
  // each stack's gain by strip and lane, kept the same way
  const gains = new Map();
  // px per row of a stack
  const ROW_PX = 14;

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
  const send = (id, data) => window.dash_clientside && window.dash_clientside.set_props && id && window.dash_clientside.set_props(id, { data });

  function clock(t, long, tenths) {
    const sign = t < 0 ? '-' : '';
    t = Math.abs(t);
    const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
    const sec = tenths ? s.toFixed(tenths).padStart(3 + tenths, '0') : String(Math.floor(s)).padStart(2, '0');
    return sign + (long || h ? `${h}:${String(m).padStart(2, '0')}:${sec}` : `${m}:${sec}`);
  }

  function parse(text) {
    const parts = String(text).trim().split(':').map(Number);
    if (!parts.length || parts.some((v) => !Number.isFinite(v))) return null;
    return parts.reduce((total, v) => total * 60 + v, 0);
  }

  function make(root) {
    const cfg = JSON.parse(root.dataset.strip);
    const video = cfg.video ? document.getElementById(cfg.video) : null;
    const length = Math.max(cfg.length || (video && video.duration) || 1, 0.1);
    const long = (cfg.offset || 0) + length * (cfg.scale || 1) >= 3600;
    const noun = cfg.noun || 'segment', editWindow = cfg.editWindow === undefined ? cfg.edit : cfg.editWindow;
    const kept = root.id && views.get(root.id);
    const S = {
      own: cfg.start || 0, view: kept && kept[1] <= length ? [...kept] : [0, length], segs: (cfg.segments || []).map((s) => ({ ...s })), sel: null,
      win: cfg.window ? [...cfg.window] : null, oldWin: null, said: '', drag: null, ticks: [], n: 0, raf: 0, tell: 0, drawn: '', last: null,
    };
    const extraLane = (lane, k) => {
      const name = `<span class="strip-name" title="${esc(lane.title || lane.name)}">${esc(lane.name)}</span>`;
      if (lane.kind === 'stack') {
        const n = lane.rows.length, labels = lane.rows.map((row, i) => `<i style="top:${((i + 0.5) / n) * 100}%">${esc(row.name)}</i>`).join('');
        const series = [...new Map(lane.rows.flatMap((row) => row.series).map((s) => [s.name, s.colour])).entries()];
        const legend = series.length > 1 ? series.map(([label, colour]) => `<b style="color:${colour}">${esc(label)}</b>`).join(' ') : '';
        const gain = lane.own ? '' : `<span class="strip-gain"><button type="button" data-g="0.5" data-x="${k}" title="Smaller">−</button><button type="button" data-g="2" data-x="${k}" title="Larger">+</button></span>`;
        return `<div class="strip-lane"><span class="strip-name" title="${esc(lane.title || lane.name)}">${esc(lane.name)}${gain}</span>`
          + `<div class="strip-track strip-stack" data-x="${k}" style="height:${Math.max(n * ROW_PX, 28)}px"><svg preserveAspectRatio="none" viewBox="0 0 100 ${n * 10}"></svg>`
          + `<span class="strip-rows">${labels}</span><span class="strip-legend">${legend} <i class="strip-scale"></i></span></div></div>`;
      }
      if (lane.kind === 'lines') {
        const legend = lane.series.length > 1 || lane.scaled ? lane.series.map((s) => `<b style="color:${s.colour}">${esc(s.name)}</b>`).join(' ') : '';
        const range = lane.scaled ? '' : `<span class="strip-range"><i>${+lane.high.toPrecision(3)}${esc(lane.unit || '')}</i><i>${+lane.low.toPrecision(3)}${esc(lane.unit || '')}</i></span>`;
        return `<div class="strip-lane">${name}<div class="strip-track strip-lines" data-x="${k}"><svg preserveAspectRatio="none" viewBox="0 0 100 100">`
          + lane.series.map((s) => `<path stroke="${s.colour}"/>`).join('') + `</svg>${range}<span class="strip-legend">${legend}</span></div></div>`;
      }
      return `<div class="strip-lane">${name}<div class="strip-track thin strip-bars" data-x="${k}"></div></div>`;
    };
    const rec = (own) => own * (cfg.scale || 1) + (cfg.offset || 0);
    const ownOf = (t) => (t - (cfg.offset || 0)) / (cfg.scale || 1);
    const lanes = [
      `<div class="strip-lane"><span></span><div class="strip-ruler"><b class="strip-hover"></b></div></div>`,
      `<div class="strip-lane"><span class="strip-name">Analysed</span><div class="strip-track thin" data-lane="win"></div></div>`,
      cfg.edit ? `<div class="strip-lane"><span class="strip-name">${esc(cfg.editName || 'Segments')}</span><div class="strip-track" data-lane="segs"><svg class="strip-wave" preserveAspectRatio="none"><path/></svg></div></div>` : '',
      cfg.ticks ? `<div class="strip-lane strip-ticks-lane" hidden><span class="strip-name">On this page</span><div class="strip-track thin" data-lane="ticks"></div></div>` : '',
      ...(cfg.lanes || []).map(extraLane),
    ];
    root.innerHTML = `
      <div class="strip-bar">
        ${video ? '<button type="button" class="strip-play" title="Play or pause (space)"></button>' : ''}
        <input class="strip-time" title="Type a time, as 1:43 or 103">
        <span class="strip-spacer"></span>
        ${video ? `<select class="strip-speed" title="Speed">${[0.5, 1, 1.5, 2].map((r) => `<option value="${r}"${r === 1 ? ' selected' : ''}>${r}×</option>`).join('')}</select>
        <button type="button" class="strip-mute" title="Sound on or off"></button>` : ''}
        <button type="button" class="strip-zoom" data-z="out" title="Zoom out (ctrl and scroll)">−</button>
        <button type="button" class="strip-zoom" data-z="in" title="Zoom in (ctrl and scroll)">+</button>
      </div>
      <div class="strip-lanes">${lanes.join('')}<div class="strip-head"></div></div>
      ${cfg.edit ? '<div class="strip-chosen"></div>' : ''}`;
    const $ = (sel) => root.querySelector(sel);
    const ruler = $('.strip-ruler'), hover = $('.strip-hover'), head = $('.strip-head'), timeBox = $('.strip-time');
    const winTrack = $('[data-lane=win]'), segTrack = $('[data-lane=segs]'), tickTrack = $('[data-lane=ticks]');
    const chosen = $('.strip-chosen'), lanesBox = $('.strip-lanes');
    const extraTracks = [...root.querySelectorAll('.strip-track[data-x]')];

    const span = () => S.view[1] - S.view[0];
    const pct = (own) => `${((own - S.view[0]) / span()) * 100}%`;
    const width = (a, b) => `${((b - a) / span()) * 100}%`;
    const ownAt = (x) => { const r = ruler.getBoundingClientRect(); return clamp(S.view[0] + ((x - r.left) / r.width) * span(), 0, length); };
    const now = () => (video && video.readyState ? video.currentTime : S.own);
    const playing = () => video && !video.paused;
    const snap = (t, skip) => {
      const near = (6 / ruler.clientWidth) * span();
      for (const s of S.segs) if (s !== skip) for (const e of [s.a, s.b]) if (Math.abs(e - t) < near) return e;
      return Math.round(t * 100) / 100;
    };

    function tell() {
      clearTimeout(S.tell);
      S.tell = setTimeout(() => {
        const own = Math.round(now() * 100) / 100;
        send(cfg.timeStore, { camera: cfg.camera, own, t: Math.round(rec(own) * 100) / 100 });
      }, 120);
    }

    function seek(own, play) {
      own = clamp(own, 0, length);
      S.own = own;
      if (video && video.readyState) video.currentTime = own; else tell();
      if (play && video) video.play();
      follow(true);
      draw();
    }

    function follow(centre) {
      const t = now(), w = span();
      if (w >= length) return;
      if (t < S.view[0] || t > S.view[1]) {
        const start = centre ? t - w / 2 : t - w * 0.1;
        S.view = [clamp(start, 0, length - w), 0];
        S.view[1] = S.view[0] + w;
      }
    }

    function zoom(factor, around) {
      const w = clamp(span() * factor, Math.min(2, length), length);
      const at = around === undefined ? now() : around;
      const share = (at - S.view[0]) / span();
      const start = clamp(at - share * w, 0, length - w);
      S.view = [start, start + w];
      draw();
    }

    function drawRuler() {
      const w = span(), px = ruler.clientWidth || 600;
      const step = STEPS.find((s) => (s / w) * px >= 64) || 3600;
      const tenths = step < 1 ? 1 : 0;
      let html = '';
      for (let t = Math.ceil(rec(S.view[0]) / step) * step; t <= rec(S.view[1]) + 1e-9; t += step) {
        html += `<span style="left:${pct(ownOf(t))}">${clock(t, long, tenths)}</span>`;
      }
      ruler.querySelectorAll('span').forEach((el) => el.remove());
      ruler.insertAdjacentHTML('afterbegin', html);
    }

    function drawWave() {
      const svg = segTrack && segTrack.querySelector('svg');
      if (!svg || !cfg.loudness) return;
      const { step, values } = cfg.loudness, px = Math.max(segTrack.clientWidth, 1);
      const first = Math.max(0, Math.floor(S.view[0] / step)), last = Math.min(values.length - 1, Math.ceil(S.view[1] / step));
      const every = Math.max(1, Math.floor((last - first) / px));
      const top = [], bottom = [];
      for (let i = first; i <= last; i += every) {
        let v = 0;
        for (let j = i; j < Math.min(i + every, values.length); j++) v = Math.max(v, values[j]);
        const x = (((i * step) - S.view[0]) / span()) * 100;
        top.push(`${x.toFixed(3)} ${(50 - 46 * v).toFixed(2)}`);
        bottom.push(`${x.toFixed(3)} ${(50 + 46 * v).toFixed(2)}`);
      }
      svg.setAttribute('viewBox', '0 0 100 100');
      svg.querySelector('path').setAttribute('d', top.length ? `M${top.join(' L')} L${bottom.reverse().join(' L')} Z` : '');
    }

    function drawWindow() {
      if (!S.win) { winTrack.innerHTML = ''; return; }
      const [a, b] = S.win;
      winTrack.innerHTML = `<i class="strip-out" style="left:0;width:${pct(a)}"></i><i class="strip-out" style="left:${pct(b)};right:0"></i>`
        + `<i class="strip-window" style="left:${pct(a)};width:${width(a, b)}"></i>`
        + (editWindow ? `<i class="strip-handle" data-h="0" style="left:${pct(a)}"></i><i class="strip-handle" data-h="1" style="left:${pct(b)}"></i>` : '');
    }

    function drawSegments() {
      if (!segTrack) return;
      segTrack.querySelectorAll('.strip-block, .strip-out').forEach((el) => el.remove());
      if (S.win) segTrack.insertAdjacentHTML('beforeend', `<i class="strip-out" style="left:0;width:${pct(S.win[0])}"></i><i class="strip-out" style="left:${pct(S.win[1])};right:0"></i>`);
      S.segs.forEach((s, i) => segTrack.insertAdjacentHTML('beforeend',
        `<div class="strip-block${s === S.sel ? ' chosen' : ''}${s.done ? '' : ' pending'}" data-i="${i}" style="left:${pct(s.a)};width:${width(s.a, s.b)}" title="${esc(s.name)}: ${s.a.toFixed(2)} to ${s.b.toFixed(2)} s">`
        + `<i class="grip l"></i>${esc(s.name)}<small>${(s.b - s.a).toFixed(1)} s</small><i class="grip r"></i></div>`));
    }

    function drawTicks() {
      if (!tickTrack) return;
      tickTrack.closest('.strip-lane').hidden = !S.ticks.length;
      tickTrack.innerHTML = S.ticks.filter(([t]) => t >= S.view[0] && t <= S.view[1])
        .map(([t, what]) => `<i class="strip-tick" style="left:${pct(t)}" data-t="${t}" title="${esc(clock(rec(t), long, 1))}: ${esc(what)}"></i>`).join('');
    }

    function drawExtra() {
      const key = `${S.view[0]},${S.view[1]},${ruler.clientWidth}`;
      if (key === S.drawn) return;
      S.drawn = key;
      extraTracks.forEach((track) => {
        const lane = cfg.lanes[+track.dataset.x];
        if (lane.kind === 'stack') {
          drawStack(track, lane, +track.dataset.x);
        } else if (lane.kind === 'lines') {
          const size = lane.series[0].values.length, low = lane.low, high = lane.high;
          const first = Math.max(0, Math.floor(S.view[0] / lane.step) - 1), last = Math.min(size - 1, Math.ceil(S.view[1] / lane.step) + 1);
          track.querySelectorAll('path').forEach((path, i) => {
            const values = lane.series[i].values;
            let d = '', pen = false;
            for (let j = first; j <= last; j++) {
              const v = values[j];
              if (v === null) { pen = false; continue; }
              const x = (((j + 0.5) * lane.step - S.view[0]) / span()) * 100;
              const y = 96 - ((clamp(v, low, high) - low) / (high - low)) * 92;
              d += `${pen ? 'L' : 'M'}${x.toFixed(3)} ${y.toFixed(2)}`;
              pen = true;
            }
            path.setAttribute('d', d);
          });
        } else {
          track.innerHTML = lane.spans.filter((b) => b.b >= S.view[0] && b.a <= S.view[1]).map((b) =>
            `<i class="strip-code" data-t="${b.a}" style="left:${pct(b.a)};width:${width(b.a, b.b)}" title="${esc(b.label)}: ${clock(rec(b.a), long, 1)} to ${clock(rec(b.b), long, 1)}">${esc(b.label)}</i>`).join('');
        }
      });
    }

    function drawStack(track, lane, k) {
      const size = lane.rows[0].series[0].values.length, gain = gains.get(`${root.id}|${k}`) || 1, scale = lane.scale / gain;
      // values come in units of lane.scale
      const first = Math.max(0, Math.floor(S.view[0] / lane.step) - 1), last = Math.min(size - 1, Math.ceil(S.view[1] / lane.step) + 1);
      const x = (t) => (((t - S.view[0]) / span()) * 100).toFixed(3);
      let html = lane.events.filter((t) => t >= S.view[0] && t <= S.view[1])
        .map((t) => `<path class="strip-event" d="M${x(t)} 0V${lane.rows.length * 10}"/>`).join('');
      lane.rows.forEach((row, r) => {
        const centre = r * 10 + 5;
        row.series.forEach((s) => {
          let d = '', pen = false;
          for (let j = first; j <= last; j++) {
            const v = s.values[j];
            if (v === null) { pen = false; continue; }
            d += `${pen ? 'L' : 'M'}${x((j + 0.5) * lane.step)} ${(centre - clamp(v * gain, -1, 1) * 4.7).toFixed(2)}`;
            pen = true;
          }
          html += `<path stroke="${s.colour}" d="${d}"/>`;
        });
      });
      track.querySelector('svg').innerHTML = html;
      const said = track.querySelector('.strip-scale');
      if (said) said.textContent = lane.own ? 'each row to its own range' : `row ±${+scale.toPrecision(2)}${lane.unit ? ` ${lane.unit}` : ''}`;
    }

    function drawChosen() {
      if (!chosen || (document.activeElement && chosen.contains(document.activeElement))) return;
      const s = S.sel;
      if (!s) {
        chosen.innerHTML = `<span class="strip-said">${esc(S.said || `Drag on an empty stretch to make a ${noun}.`)}</span>`
          + (S.oldWin ? ' <button type="button" class="link" data-act="undo-window">Undo</button>' : '');
        return;
      }
      const box = (label, key, value, type) => `<label class="field"><span class="field-label">${label}</span><input data-k="${key}" value="${esc(value)}"${type ? ` type="${type}" step="0.01" min="0"` : ''}></label>`;
      const said = S.said || (cfg.notes ? '' : s.done ? 'summarised' : 'summarised after the next run');
      // names the page suggests, such as the events a signal file marked
      const listId = `${root.id || 'strip'}-names`;
      const nameBox = (cfg.names || []).length
        ? box('Name', 'name', s.name).replace('data-k="name"', `data-k="name" list="${listId}"`) + `<datalist id="${listId}">${cfg.names.map((n) => `<option value="${esc(n)}">`).join('')}</datalist>`
        : box('Name', 'name', s.name);
      chosen.innerHTML = `${nameBox}${box('From (s)', 'a', s.a.toFixed(2), 'number')}<button type="button" class="button small" data-act="a">Start here</button>`
        + `${box('To (s)', 'b', s.b.toFixed(2), 'number')}<button type="button" class="button small" data-act="b">End here</button>`
        + (cfg.notes ? box('Note', 'note', s.note || '') : '')
        + `<span class="strip-said">${esc(said)}</span>`
        + `<button type="button" class="button small" data-act="play">Play it</button><button type="button" class="button small" data-act="remove">Remove</button>`;
    }

    function draw() {
      if (!root.isConnected) return;
      const t = now();
      if (document.activeElement !== timeBox) timeBox.value = clock(rec(t), long, 2);
      if (video) {
        root.querySelector('.strip-play').classList.toggle('on', !!playing());
        root.querySelector('.strip-mute').classList.toggle('off', video.muted);
      }
      const box = lanesBox.getBoundingClientRect(), r = ruler.getBoundingClientRect();
      const x = ((t - S.view[0]) / span()) * r.width;
      head.style.left = `${r.left - box.left + x}px`;
      head.hidden = x < 0 || x > r.width;
      drawRuler(); drawWindow(); drawSegments(); drawTicks(); drawWave(); drawExtra(); drawChosen();
      if (root.id) views.set(root.id, [...S.view]);
    }

    function loop() {
      cancelAnimationFrame(S.raf);
      if (!playing() || !root.isConnected) return;
      follow(false);
      draw();
      S.raf = requestAnimationFrame(loop);
    }

    function save(op) {
      S.n += 1;
      send(cfg.editStore, { ...op, n: S.n });
    }

    // what a person changed and what it was before, so a mark among several of one name is found again
    const setOp = (s, was) => ({ op: 'set', name: s.name, old: was ? was.name : null, start: s.a, end: s.b, note: s.note || '', was });
    const before = (s) => ({ name: s.name, a: s.a, b: s.b, note: s.note || '' });

    function readTicks() {
      if (!cfg.ticks) return;
      const found = [];
      document.querySelectorAll(cfg.ticks).forEach((button) => {
        if (button.closest('details:not([open])')) return;
        let id;
        try { id = JSON.parse(button.id); } catch (e) { return; }
        const card = button.closest('.card'), row = button.closest('tr, figure, .clip');
        const title = card ? card.querySelector('.card-head h2') : null;
        const what = [title ? title.textContent : '', row ? row.textContent.replace(/\s+/g, ' ').trim().slice(0, 70) : ''].filter(Boolean).join(': ');
        found.push([ownOf(parseFloat(id.t)), what]);
      });
      S.ticks = found;
      drawTicks();
    }

    timeBox.addEventListener('keydown', (e) => { if (e.key === 'Enter') timeBox.blur(); });
    timeBox.addEventListener('change', () => { const t = parse(timeBox.value); timeBox.blur(); if (t !== null) seek(ownOf(t)); else draw(); });
    root.querySelectorAll('.strip-zoom').forEach((b) => b.addEventListener('click', () => zoom(b.dataset.z === 'in' ? 0.5 : 2)));
    if (video) {
      video.controls = false;
      root.querySelector('.strip-play').addEventListener('click', () => { if (video.paused) video.play(); else video.pause(); });
      root.querySelector('.strip-speed').addEventListener('change', (e) => { video.playbackRate = parseFloat(e.target.value); });
      root.querySelector('.strip-mute').addEventListener('click', () => { video.muted = !video.muted; draw(); });
      video.addEventListener('click', () => { if (video.paused) video.play(); else video.pause(); });
      video.addEventListener('dblclick', () => { if (video.requestFullscreen) video.requestFullscreen(); });
      video.addEventListener('play', loop);
      video.addEventListener('pause', () => { draw(); tell(); });
      video.addEventListener('seeked', () => { draw(); if (video.paused) tell(); });
      video.addEventListener('loadedmetadata', () => {
        if (cfg.start && !video.currentTime) video.currentTime = Math.min(cfg.start, video.duration || cfg.start);
        else tell();
        draw();
      });
      if (video.readyState && cfg.start && !video.currentTime) video.currentTime = Math.min(cfg.start, video.duration || cfg.start);
      else if (video.readyState) tell();
    } else {
      tell();
    }

    ruler.parentElement.parentElement.addEventListener('wheel', (e) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        zoom(e.deltaY > 0 ? 1.25 : 0.8, ownAt(e.clientX));
      } else if ((e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) && span() < length) {
        e.preventDefault();
        const d = ((e.deltaX || e.deltaY) / ruler.clientWidth) * span();
        const start = clamp(S.view[0] + d, 0, length - span());
        S.view = [start, start + span()];
        draw();
      }
    }, { passive: false });

    const move = (e) => {
      const t = ownAt(e.clientX);
      const track = e.currentTarget, lane = track.dataset && track.dataset.x !== undefined ? cfg.lanes[+track.dataset.x] : null;
      let row = '';
      if (lane && lane.kind === 'stack') {
        const r = track.getBoundingClientRect();
        row = lane.rows[clamp(Math.floor(((e.clientY - r.top) / r.height) * lane.rows.length), 0, lane.rows.length - 1)].name;
      }
      hover.textContent = clock(rec(t), long, 1) + (row ? ` · ${row}` : '');
      hover.style.left = pct(t);
      hover.style.opacity = 1;
      const d = S.drag;
      if (!d) return;
      if (d.scrub) { S.own = t; if (video && video.readyState) video.currentTime = t; }
      else if (d.win !== undefined) { S.win[d.win] = Math.round(t * 10) / 10; if (S.win[0] > S.win[1]) { S.win.reverse(); d.win = 1 - d.win; } }
      else if (d.part === 'move') {
        const w = d.b0 - d.a0;
        const a = snap(clamp(d.a0 + t - d.t0, 0, length - w), d.s);
        d.s.a = a; d.s.b = Math.round((a + w) * 100) / 100;
      } else {
        d.s[d.part] = snap(t, d.s);
        if (d.s.a > d.s.b) { [d.s.a, d.s.b] = [d.s.b, d.s.a]; d.part = d.part === 'a' ? 'b' : 'a'; }
      }
      d.moved = true;
      draw();
    };
    const up = () => {
      const d = S.drag;
      S.drag = null;
      if (!d) return;
      if (d.scrub) { if (!video || !video.readyState) tell(); }
      else if (d.win !== undefined && d.moved) { save({ op: 'window', start: S.win[0], end: S.win[1] }); }
      else if (d.s && d.fresh && d.s.b - d.s.a < 0.5) { S.segs.splice(S.segs.indexOf(d.s), 1); S.sel = null; }
      else if (d.s && d.moved) { S.said = 'saving'; save(setOp(d.s, d.fresh ? null : d.was)); }
      draw();
    };
    ruler.addEventListener('pointerdown', (e) => { S.drag = { scrub: true }; ruler.setPointerCapture(e.pointerId); move(e); });
    if (editWindow) {
      winTrack.addEventListener('pointerdown', (e) => {
        const h = e.target.closest('.strip-handle');
        if (!h) return;
        S.oldWin = S.oldWin || [...S.win];
        S.drag = { win: +h.dataset.h };
        winTrack.setPointerCapture(e.pointerId);
      });
    }
    if (cfg.edit) {
      segTrack.addEventListener('pointerdown', (e) => {
        const t = ownAt(e.clientX), block = e.target.closest('.strip-block');
        S.said = '';
        if (block) {
          const s = S.segs[+block.dataset.i];
          S.sel = s;
          S.last = s.name;
          const part = e.target.classList.contains('l') ? 'a' : e.target.classList.contains('r') ? 'b' : 'move';
          S.drag = { s, part, t0: t, a0: s.a, b0: s.b, was: before(s) };
        } else {
          let k = S.segs.length + 1;
          while (S.segs.some((s) => s.name === `${noun}_${k}`)) k += 1;
          // marks may share a name, so a new one takes the last one's, to mark the same thing again quickly
          const s = { name: cfg.notes && S.last ? S.last : `${noun}_${k}`, a: snap(t), b: snap(t), done: false, note: '' };
          S.segs.push(s); S.sel = s;
          S.drag = { s, part: 'b', t0: t, a0: s.a, b0: s.b, fresh: true };
        }
        segTrack.setPointerCapture(e.pointerId);
        draw();
      });
      chosen.addEventListener('change', (e) => {
        const s = S.sel, key = e.target.dataset.k;
        if (!s || !key) return;
        const was = before(s);
        if (key === 'name') s.name = e.target.value.trim() || was.name;
        else if (key === 'note') s.note = e.target.value.trim();
        else s[key] = parseFloat(e.target.value);
        S.last = s.name;
        e.target.blur();
        S.said = 'saving';
        save(setOp(s, was));
        draw();
      });
      chosen.addEventListener('click', (e) => {
        const act = e.target.dataset.act, s = S.sel;
        if (act === 'undo-window' && S.oldWin) { S.win = S.oldWin; S.oldWin = null; save({ op: 'window', start: S.win[0], end: S.win[1] }); }
        if (!s) return draw();
        if (act === 'a' || act === 'b') { const was = before(s); s[act] = Math.round(now() * 100) / 100; S.said = 'saving'; save(setOp(s, was)); }
        if (act === 'play') seek(s.a, true);
        if (act === 'remove') { S.segs.splice(S.segs.indexOf(s), 1); S.sel = null; save({ op: 'remove', name: s.name, start: s.a, end: s.b }); }
        draw();
      });
    }
    root.addEventListener('click', (e) => {
      const button = e.target.closest('[data-g]');
      if (!button) return;
      const key = `${root.id}|${button.dataset.x}`;
      gains.set(key, clamp((gains.get(key) || 1) * +button.dataset.g, 1 / 64, 64));
      S.drawn = '';
      draw();
    });
    for (const track of extraTracks) {
      track.addEventListener('pointerdown', (e) => {
        const bar = e.target.closest('.strip-code');
        if (bar) { seek(+bar.dataset.t, true); return; }
        S.drag = { scrub: true };
        track.setPointerCapture(e.pointerId);
        move(e);
      });
    }
    for (const track of [ruler, winTrack, segTrack, tickTrack, ...extraTracks].filter(Boolean)) {
      track.addEventListener('pointermove', move);
      track.addEventListener('pointerup', up);
      track.addEventListener('pointerleave', () => { hover.style.opacity = 0; });
    }
    if (tickTrack) tickTrack.addEventListener('click', (e) => { const k = e.target.closest('.strip-tick'); if (k) seek(+k.dataset.t, true); });

    const api = {
      root, seek, draw, readTicks,
      seekRecording: (t, play) => seek(ownOf(t), play),
      random: () => { const [a, b] = S.win || [0, length]; seek(a + Math.random() * (b - a)); },
      shown: (data) => {
        const was = S.sel && before(S.sel);
        S.segs = (data.segments || []).map((s) => ({ ...s }));
        S.sel = was ? S.segs.find((s) => s.name === was.name && (!cfg.notes || Math.abs(s.a - was.a) < 0.01)) || null : null;
        if (data.window !== undefined) S.win = data.window;
        S.said = data.said || '';
        if (data.op !== 'window') S.oldWin = null;
        draw();
      },
    };
    new ResizeObserver(draw).observe(ruler);
    draw();
    readTicks();
    return api;
  }

  function scan() {
    for (const [root, api] of strips) if (!root.isConnected) strips.delete(root);
    document.querySelectorAll('.time-strip').forEach((root) => {
      const api = strips.get(root);
      if (api && api.config === root.dataset.strip) return;
      const made = make(root);
      made.config = root.dataset.strip;
      strips.set(root, made);
    });
    for (const api of strips.values()) api.readTicks();
    pin();
  }

  // a player slot whose video is mostly under the tabs pins its card small there, keeping its height in the page
  function pin() {
    const bar = document.getElementById('topbar');
    const top = bar ? bar.getBoundingClientRect().bottom : 0;
    document.querySelectorAll('.player-slot').forEach((slot) => {
      const card = slot.firstElementChild, was = slot.classList.contains('away');
      if (!card) return;
      if (!was) slot.dataset.full = slot.offsetHeight;
      const full = +slot.dataset.full || 0, r = slot.getBoundingClientRect();
      const away = !!slot.querySelector('video.player') && slot.offsetParent !== null && r.top + full * 0.6 < top;
      if (away !== was) { slot.classList.toggle('away', away); slot.style.height = away ? `${full}px` : ''; }
      if (away) Object.assign(card.style, { top: `${top}px`, left: `${r.left}px`, width: `${r.width}px` });
      else if (was) Object.assign(card.style, { top: '', left: '', width: '' });
    });
  }
  addEventListener('scroll', pin, { passive: true });
  addEventListener('resize', pin);

  // every page stays in the document, so the strip meant is the first on the page showing
  const first = () => [...strips.values()].find((api) => api.root.isConnected && api.root.offsetParent !== null);
  window.macawStrip = {
    seek: (t, play) => { const api = first(); if (api) api.seekRecording(t, play); return !!api; },
    random: () => { const api = first(); if (api) api.random(); },
    shown: (id, data) => { for (const api of strips.values()) if (api.root.id === id) api.shown(data); },
  };

  let pending = 0;
  new MutationObserver(() => { clearTimeout(pending); pending = setTimeout(scan, 60); })
    .observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ['data-strip', 'open'] });
  document.addEventListener('toggle', () => { clearTimeout(pending); pending = setTimeout(scan, 60); }, true);
  document.addEventListener('keydown', (e) => {
    if (e.code !== 'Space' || e.target.closest('input, textarea, select, [contenteditable]')) return;
    const api = first();
    const video = api && document.getElementById(JSON.parse(api.root.dataset.strip).video || '');
    if (!video) return;
    e.preventDefault();
    if (video.paused) video.play(); else video.pause();
  });
})();
