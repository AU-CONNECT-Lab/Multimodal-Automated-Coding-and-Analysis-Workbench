// The Overview page: the workflow drawing, and pointing at and folding the grid's squares.
(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const LEFT = 100, PAD = 16, GAP = 8, MIN_COL = 124;
  let data = null, drawnFor = null;

  function el(name, attrs, parent) {
    const node = document.createElementNS(NS, name);
    for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, value);
    if (parent) parent.appendChild(node);
    return node;
  }

  function open(href) {
    window.history.pushState({}, '', href);
    window.dispatchEvent(new PopStateEvent('popstate'));
  }

  function draw(next) {
    if (next !== undefined) data = next;
    const box = document.getElementById('overview-flow');
    if (!box) return;
    box.replaceChildren();
    drawnFor = box;
    if (!data || !data.nodes || !data.nodes.length) return;
    const maxDepth = Math.max(...data.nodes.map((n) => n.depth));
    const colW = Math.max(MIN_COL, Math.floor((box.clientWidth - LEFT - 12) / (maxDepth + 1)));
    const width = LEFT + colW * (maxDepth + 1) + 12, nodeW = colW - 18;
    const svg = el('svg', { width, class: 'flow' }, box);
    // labels are measured as drawn, so a long name breaks onto a second line instead of running out of its box
    const probe = el('g', { class: 'node' }, svg);
    const measure = (text, cls) => { const t = el('text', cls ? { class: cls } : {}, probe); t.textContent = text; const w = t.getComputedTextLength(); t.remove(); return w; };
    const wrap = (text, room) => {
      if (measure(text) <= room) return [text];
      const words = text.split(' ');
      let first = words.shift();
      while (words.length && measure(first + ' ' + words[0]) <= room) first += ' ' + words.shift();
      return words.length ? [first, words.join(' ')] : [first];
    };
    const at = {};
    let y = 0;
    const bands = [];
    for (const lane of data.lanes) {
      const nodes = data.nodes.filter((n) => n.lane === lane);
      const stack = {};
      for (const n of nodes) {
        const room = nodeW - 30 - (n.issue ? measure(String(n.issue[0]), 'issue') + 10 : 0);
        const lines = wrap(n.label, room), h = lines.length > 1 ? 40 : 28;
        const top = stack[n.depth] ?? y + PAD;
        at[n.id] = { x: LEFT + n.depth * colW, y: top, h, lines };
        stack[n.depth] = top + h + GAP;
      }
      const bottom = Math.max(...Object.values(stack)) - GAP + PAD;
      // a short column sits in the middle of its lane
      for (const n of nodes) at[n.id].y += (bottom - y - 2 * PAD - (stack[n.depth] - GAP - y - PAD)) / 2;
      bands.push([lane, y, bottom - y]);
      y = bottom;
    }
    probe.remove();
    svg.setAttribute('height', y);
    svg.setAttribute('viewBox', `0 0 ${width} ${y}`);
    bands.forEach(([lane, top, height], k) => {
      if (k) el('line', { class: 'lane-rule', x1: 16, x2: width - 16, y1: top, y2: top }, svg);
      el('text', { class: 'lane-label', x: 20, y: top + height / 2 + 4 }, svg).textContent = lane.toUpperCase();
    });
    const edges = el('g', {}, svg);
    for (const [from, to] of data.edges) {
      const a = at[from], b = at[to];
      if (!a || !b) continue;
      const x0 = a.x + nodeW, y0 = a.y + a.h / 2, x1 = b.x, y1 = b.y + b.h / 2, bend = Math.max(30, (x1 - x0) / 2);
      el('path', { class: 'edge', 'data-from': from, 'data-to': to, d: `M${x0},${y0} C${x0 + bend},${y0} ${x1 - bend},${y1} ${x1},${y1}` }, edges);
    }
    for (const n of data.nodes) {
      const p = at[n.id];
      const g = el('g', { class: `node${n.check ? ' check' : ''}`, 'data-n': n.id }, svg);
      el('title', {}, g).textContent = n.tip;
      el('rect', { class: 'box', x: p.x, y: p.y, width: nodeW, height: p.h, rx: n.check ? Math.min(14, p.h / 2) : 6 }, g);
      el('circle', { class: `mark ${n.state}`, cx: p.x + 13, cy: p.y + (p.h > 30 ? 12.5 : p.h / 2), r: n.check ? 4.2 : 5 }, g);
      const lineY = p.h > 30 ? [p.y + 16.5, p.y + 31] : [p.y + 18.5];
      p.lines.forEach((line, k) => { el('text', { x: p.x + 24, y: lineY[k] }, g).textContent = line; });
      if (n.issue) el('text', { class: `issue ${n.issue[1]}`, x: p.x + nodeW - 9, y: lineY[0], 'text-anchor': 'end' }, g).textContent = n.issue[0];
      g.addEventListener('click', () => open(n.href));
    }
    const above = {};
    const ancestors = (id) => above[id] || (above[id] = new Set(data.edges.filter(([, to]) => to === id).flatMap(([from]) => [from, ...ancestors(from)])));
    svg.addEventListener('mouseover', (event) => {
      const g = event.target.closest('.node');
      if (!g) return;
      const lit = new Set([g.dataset.n, ...ancestors(g.dataset.n)]);
      svg.classList.add('focus');
      svg.querySelectorAll('.node').forEach((x) => x.classList.toggle('hot', lit.has(x.dataset.n)));
      svg.querySelectorAll('.edge').forEach((e) => e.classList.toggle('hot', lit.has(e.dataset.from) && lit.has(e.dataset.to)));
    });
    svg.addEventListener('mouseleave', () => { svg.classList.remove('focus'); svg.querySelectorAll('.hot').forEach((x) => x.classList.remove('hot')); });
  }

  // the drawing follows the card's width
  let lastWidth = 0;
  new ResizeObserver(() => {
    const box = document.getElementById('overview-flow');
    if (!box || !data) return;
    if (box !== drawnFor || Math.abs(box.clientWidth - lastWidth) > 4) { lastWidth = box.clientWidth; draw(); }
  }).observe(document.documentElement);

  // pointing at a square shows it beside the grid; the grid lights its column
  let hint = null;
  function clearGrid(table) {
    table.querySelectorAll('.hl').forEach((x) => x.classList.remove('hl'));
    const inspect = table.closest('.standing-card')?.querySelector('.inspect');
    if (inspect && hint !== null) inspect.innerHTML = hint;
  }
  let active = null;
  document.addEventListener('mouseover', (event) => {
    const table = event.target instanceof Element ? event.target.closest('table.standing') : null;
    if (active && active !== table) { clearGrid(active); active = null; }
    if (!table) return;
    active = table;
    const cell = event.target.closest('[data-col]');
    table.querySelectorAll('.hl').forEach((x) => x.classList.remove('hl'));
    if (!cell) return;
    table.querySelectorAll(`[data-col="${CSS.escape(cell.dataset.col)}"]`).forEach((x) => x.classList.add('hl'));
    const inspect = table.closest('.standing-card')?.querySelector('.inspect');
    if (!inspect || !cell.dataset.rec) return;
    if (hint === null || !inspect.dataset.pointing) hint = inspect.innerHTML;
    inspect.dataset.pointing = '1';
    const summary = cell.closest('tr').querySelector('td.summary')?.textContent || '';
    const rows = [[cell.querySelector('.check') ? 'Check' : 'Step', cell.dataset.label, cell.dataset.col],
      ['State', cell.dataset.state], ['Click', `opens ${cell.dataset.page}`], ['Recording', summary]];
    inspect.replaceChildren();
    const title = document.createElement('h3'); title.textContent = cell.dataset.rec; inspect.appendChild(title);
    const list = document.createElement('dl');
    for (const [name, value, code] of rows) {
      const dt = document.createElement('dt'); dt.textContent = name;
      const dd = document.createElement('dd'); dd.textContent = value;
      if (code) { const c = document.createElement('code'); c.textContent = code; dd.append(document.createElement('br'), c); }
      list.append(dt, dd);
    }
    inspect.appendChild(list);
  });
  document.addEventListener('mouseout', (event) => {
    const table = event.target instanceof Element ? event.target.closest('table.standing') : null;
    if (table && !(event.relatedTarget instanceof Element && table.contains(event.relatedTarget))) {
      clearGrid(table);
      const inspect = table.closest('.standing-card')?.querySelector('.inspect');
      if (inspect) delete inspect.dataset.pointing;
      active = null;
    }
  });

  // the recordings with everything done fold into one line
  document.addEventListener('click', (event) => {
    const fold = event.target instanceof Element ? event.target.closest('tr.fold-row th') : null;
    if (!fold) return;
    const done = fold.closest('table').querySelector('tbody.done');
    if (!done) return;
    const closed = done.classList.toggle('closed');
    fold.textContent = (closed ? '▸' : '▾') + fold.textContent.slice(1);
  });

  window.macawOverview = { draw };
  if (window.macawOverviewPending !== undefined) draw(window.macawOverviewPending);
})();
