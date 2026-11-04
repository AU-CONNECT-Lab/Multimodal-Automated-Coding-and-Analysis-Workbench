// The Setup page's room: the setup's people, devices and things in one scene, drawn in a turnable 3D view and three
// flat views. People choose and drag them here; the page's callbacks hold the model, check it and save it.
import * as THREE from 'three';
import { OrbitControls } from './three/OrbitControls.js';

const KINDS = ['people', 'cameras', 'microphones', 'things'];
const CM_PER = { cm: 1, m: 100, in: 2.54 };
const DECIMALS = { cm: 0, m: 2, in: 1 };
const LEVEL = { people: 'eyes', cameras: 'height', microphones: 'height', things: 'height' };
// heights drawn while none is given, and the floor grid's step, in centimetres
const STAND_IN_CM = { people: 120, cameras: 150, microphones: 100, things: 70 };
const GRID_CM = { cm: 50, m: 50, in: 60.96 };
const SCALE_BARS = { cm: [10, 20, 50, 100, 200, 500], m: [0.1, 0.2, 0.5, 1, 2, 5], in: [6, 12, 24, 36, 60, 120] };
// the drawing for a paper's colours, so the 3D view and the flat drawings agree; things and stands follow the style
const COLOURS = { cameras: 0x4a72d4, lens: 0x26324d, stand: 0x868e96, microphones: 0x2f9e44, things: 0xc9ced6, pick: 0xf08c00 };
// sizes of the flat drawings' marks, in centimetres, as in the drawing for a paper
const PAPER_CM = { person: 23, head: 11, device: 9, arrow: 16, body: 8 };
const SVGNS = 'http://www.w3.org/2000/svg';
// what an entry of the list carries when dragged into a view
const DROP = 'application/x-macaw-room';
const FLATS = {
  plan: { title: 'Plan', across: 'across →', up: 'along ↑' },
  top: { title: 'From above', across: 'across →', up: 'along ↑' },
  front: { title: 'From the front, along = 0', across: 'across →', up: 'up ↑' },
  side: { title: 'From the right', across: 'along →', up: 'up ↑' },
};
// handles drawn on layers, so each view shows the ones it can be dragged in
const TURN_LAYER = 1, LIFT_LAYER = 2, TILT_LAYER = 3, DEPTH_LAYER = 4;
const FAR = 1e6;

const S = {
  model: null, modelText: '', selected: null, frame: null, mode: 'free', fitKey: '',
  scene: new THREE.Scene(), fixtures: new THREE.Group(), items: new THREE.Group(), handles: new THREE.Group(),
  objects: new Map(), walls: [], floor: null, views: {}, drag: null, active: false, waiting: null, flatScale: null,
  queued: false, theme: {}, lensTimer: null, style: 'paper',
};
S.scene.add(S.fixtures, S.items, S.handles);

const units = () => (S.model && S.model.units) || 'cm';
const u = (cm) => cm / CM_PER[units()];
const round = (value) => Number(value.toFixed(DECIMALS[units()]));
const rad = (deg) => (deg * Math.PI) / 180;
const deg = (r) => (r * 180) / Math.PI;
const placed = (entry) => entry && entry.x != null && entry.y != null;
const levelOf = (kind, entry) => (entry[LEVEL[kind]] != null ? entry[LEVEL[kind]] : u(STAND_IN_CM[kind]));
const keyOf = (kind, name) => `${kind}/${name}`;
const tell = (id, data) => window.dash_clientside && window.dash_clientside.set_props && window.dash_clientside.set_props(id, { data });

function readTheme() {
  const css = getComputedStyle(document.body);
  const get = (name, fallback) => (css.getPropertyValue(name) || '').trim() || fallback;
  S.theme = {
    ink: get('--ink', '#1f1f21'), faint: get('--faint', '#8b8b90'), border: get('--border', '#d8d8da'),
    floor: get('--room-floor', '#ffffff'), grid: get('--room-grid', '#d8d8da'), wall: get('--room-wall', '#8b8b90'),
    wallAlpha: parseFloat(get('--room-wall-alpha', '0.16')), sky: get('--room-sky', '#ffffff'), ground: get('--room-ground', '#8a8a8a'),
  };
  COLOURS.things = new THREE.Color(get('--room-thing', '#c9ced6')).getHex();
  COLOURS.stand = new THREE.Color(get('--room-stand', '#868e96')).getHex();
}

// ---------- building the scene ----------

function dispose(object) {
  object.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) (Array.isArray(child.material) ? child.material : [child.material]).forEach((m) => m.dispose());
  });
}

function clear(group) {
  for (const child of [...group.children]) {
    group.remove(child);
    dispose(child);
  }
}

function solid(colour, ghost, extra = {}) {
  return new THREE.MeshStandardMaterial({
    color: colour, roughness: 0.62, metalness: 0.04, transparent: ghost || extra.opacity != null,
    opacity: ghost ? 0.42 : 1, ...extra,
  });
}

function rod(from, to, radius, material) {
  const a = new THREE.Vector3(...from), b = new THREE.Vector3(...to);
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, a.distanceTo(b), 10), material);
  mesh.position.copy(a).add(b).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize());
  return mesh;
}

function pickable(mesh, kind, name, part = 'body') {
  mesh.userData = { kind, name, part };
  mesh.castShadow = true;
  return mesh;
}

function extent() {
  const model = S.model;
  const walls = model.walls || {};
  let xs = [], ys = [];
  const tops = [u(250)];
  if (walls.width != null && walls.length != null) {
    xs.push(0, walls.width);
    ys.push(0, walls.length);
  }
  for (const kind of KINDS) {
    for (const entry of Object.values(model[kind] || {})) {
      if (!placed(entry)) continue;
      xs.push(entry.x);
      ys.push(entry.y);
      tops.push(levelOf(kind, entry) + u(30));
    }
  }
  if (!xs.length) {
    xs = [0, u(400)];
    ys = [0, u(300)];
  }
  const pad = walls.width != null ? u(20) : u(60);
  const grow = (low, high, least) => {
    const lack = least - (high - low);
    return lack > 0 ? [low - lack / 2, high + lack / 2] : [low, high];
  };
  const [x0, x1] = grow(Math.min(...xs) - pad, Math.max(...xs) + pad, u(300));
  const [y0, y1] = grow(Math.min(...ys) - pad, Math.max(...ys) + pad, u(300));
  return { x0, x1, y0, y1, top: Math.max(...tops) };
}

function buildFixtures() {
  clear(S.fixtures);
  S.walls = [];
  const model = S.model;
  const walls = model.walls || {};
  const box = extent();
  const hasWalls = walls.width != null && walls.length != null;
  const [fx0, fx1, fy0, fy1] = hasWalls ? [0, walls.width, 0, walls.length] : [box.x0, box.x1, box.y0, box.y1];

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(fx1 - fx0, fy1 - fy0),
    new THREE.MeshStandardMaterial({ color: S.theme.floor, roughness: 0.95 }),
  );
  floor.position.set((fx0 + fx1) / 2, (fy0 + fy1) / 2, -u(0.2));
  floor.receiveShadow = true;
  floor.userData = { part: 'floor' };
  S.fixtures.add(floor);
  S.floor = floor;

  const step = u(GRID_CM[units()]);
  const lines = [];
  for (let x = Math.ceil(fx0 / step) * step; x <= fx1 + 1e-9; x += step) lines.push(x, fy0, 0, x, fy1, 0);
  for (let y = Math.ceil(fy0 / step) * step; y <= fy1 + 1e-9; y += step) lines.push(fx0, y, 0, fx1, y, 0);
  const grid = new THREE.BufferGeometry();
  grid.setAttribute('position', new THREE.Float32BufferAttribute(lines, 3));
  S.fixtures.add(new THREE.LineSegments(grid, new THREE.LineBasicMaterial({ color: S.theme.grid, transparent: true, opacity: 0.9 })));

  if (hasWalls) {
    const high = Math.max(u(250), box.top);
    const thick = u(4);
    const sides = [
      { at: [walls.width / 2, -thick / 2], size: [walls.width + 2 * thick, thick], normal: [0, -1], point: [0, 0] },
      { at: [walls.width / 2, walls.length + thick / 2], size: [walls.width + 2 * thick, thick], normal: [0, 1], point: [0, walls.length] },
      { at: [-thick / 2, walls.length / 2], size: [thick, walls.length], normal: [-1, 0], point: [0, 0] },
      { at: [walls.width + thick / 2, walls.length / 2], size: [thick, walls.length], normal: [1, 0], point: [walls.width, 0] },
    ];
    for (const side of sides) {
      const group = new THREE.Group();
      const geometry = new THREE.BoxGeometry(side.size[0], side.size[1], high);
      const wall = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: S.theme.wall, transparent: true, opacity: S.theme.wallAlpha, depthWrite: false }));
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), new THREE.LineBasicMaterial({ color: S.theme.wall }));
      group.add(wall, edges);
      group.position.set(side.at[0], side.at[1], high / 2);
      S.fixtures.add(group);
      S.walls.push({ group, normal: new THREE.Vector3(side.normal[0], side.normal[1], 0), point: new THREE.Vector3(side.point[0], side.point[1], 0) });
    }
  }

  S.fixtures.add(new THREE.HemisphereLight(S.theme.sky, S.theme.ground, 1.9));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  const middle = new THREE.Vector3((fx0 + fx1) / 2, (fy0 + fy1) / 2, 0);
  const span = Math.max(fx1 - fx0, fy1 - fy0, box.top);
  sun.position.copy(middle).add(new THREE.Vector3(-0.5, -0.8, 1.6).multiplyScalar(span));
  sun.target.position.copy(middle);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -span, right: span, top: span, bottom: -span, near: 0.01 * span, far: 4 * span });
  sun.shadow.bias = -0.0005;
  S.fixtures.add(sun, sun.target);
}

function buildPerson(name, entry) {
  const group = new THREE.Group();
  const colour = new THREE.Color(S.model.colours[name] || '#1d4e89');
  const ghost = entry.eyes == null;
  const eyes = levelOf('people', entry);
  const r = Math.min(Math.max(eyes * 0.07, u(6)), u(11));
  const neck = eyes - 1.3 * r;
  const outline = [[0.001, 0], [0.95, 0], [0.92, 0.04], [0.8, 0.3], [1.3, 0.52], [1.18, 0.64], [1.55, 0.8], [1.7, 0.92], [1.3, 0.99], [0.55, 1.0], [0.001, 1.0]];
  const lathe = new THREE.LatheGeometry(outline.map(([across, up]) => new THREE.Vector2(across * r, up * neck)), 40);
  lathe.rotateX(Math.PI / 2);
  lathe.scale(1.15, 0.72, 1);
  const body = pickable(new THREE.Mesh(lathe, solid(colour, ghost)), 'people', name);
  const head = pickable(new THREE.Mesh(new THREE.SphereGeometry(r, 32, 20), solid(colour.clone().lerp(new THREE.Color(0xffffff), 0.3), ghost)), 'people', name);
  head.position.z = eyes;
  group.add(body, head);
  if (entry.facing != null) {
    const dark = solid(0x1f1f21, ghost);
    for (const side of [-1, 1]) {
      const eye = new THREE.Mesh(new THREE.SphereGeometry(r * 0.13, 12, 8), dark);
      eye.position.set(side * 0.38 * r, 0.9 * r, eyes + 0.08 * r);
      group.add(eye);
    }
    const nose = new THREE.Mesh(new THREE.ConeGeometry(r * 0.2, r * 0.45, 12), solid(colour.clone().lerp(new THREE.Color(0xffffff), 0.3), ghost));
    nose.position.set(0, r * 1.08, eyes - 0.2 * r);
    group.add(nose);
    const arrow = new THREE.Shape([new THREE.Vector2(-0.7 * r, 2.1 * r), new THREE.Vector2(0.7 * r, 2.1 * r), new THREE.Vector2(0, 3.1 * r)]);
    const mark = new THREE.Mesh(new THREE.ShapeGeometry(arrow), new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity: ghost ? 0.4 : 0.85 }));
    mark.position.z = u(0.3);
    group.add(mark);
  }
  group.position.set(entry.x, entry.y, 0);
  group.rotation.z = -rad(entry.facing || 0);
  return { group, anchor: new THREE.Vector3(entry.x, entry.y, eyes + 1.35 * r), reach: r * 3.2, top: eyes + r };
}

function deviceColour(kind, entry) {
  // a device the setup says records one person alone takes their colour; a shared one keeps its own
  const people = kind === 'cameras' ? new Set([...(entry.body || []), ...(entry.face || []), ...(entry.gaze || [])]) : new Set(entry.hears || []);
  const [person] = people;
  if (people.size === 1 && S.model.colours[person]) return new THREE.Color(S.model.colours[person]);
  return new THREE.Color(kind === 'cameras' ? COLOURS.cameras : COLOURS.microphones);
}

function frustum(lens, aspect, known, colour) {
  const far = u(260);
  const across = far * Math.tan(rad(lens) / 2);
  const up = across / aspect;
  const corners = [[-across, far, up], [across, far, up], [across, far, -up], [-across, far, -up]];
  const group = new THREE.Group();
  const points = [];
  for (const corner of corners) points.push(0, 0, 0, ...corner);
  corners.forEach((corner, index) => points.push(...corner, ...corners[(index + 1) % 4]));
  const edges = new THREE.BufferGeometry();
  edges.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
  const line = known
    ? new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity: 0.7 })
    : new THREE.LineDashedMaterial({ color: colour, dashSize: u(6), gapSize: u(5), transparent: true, opacity: 0.6 });
  const lines = new THREE.LineSegments(edges, line);
  if (!known) lines.computeLineDistances();
  const faces = new THREE.BufferGeometry();
  const tips = corners.flat();
  faces.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, ...tips], 3));
  faces.setIndex([0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1]);
  const cone = new THREE.Mesh(faces, new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false }));
  cone.layers.set(DEPTH_LAYER);
  group.add(lines, cone);
  return group;
}

function aspectOf(name) {
  return S.frame && S.frame.camera === name && S.frame.width ? S.frame.width / S.frame.height : 16 / 9;
}

function buildCamera(name, entry) {
  const group = new THREE.Group();
  const ghost = entry.height == null;
  const height = levelOf('cameras', entry);
  group.position.set(entry.x, entry.y, height);
  const aim = new THREE.Group();
  aim.rotation.set(-rad(entry.tilt || 0), 0, -rad(entry.facing || 0), 'ZXY');
  group.add(aim);
  const colour = deviceColour('cameras', entry);
  const shell = solid(colour, ghost);
  const box = pickable(new THREE.Mesh(new THREE.BoxGeometry(u(11), u(9), u(8)), shell), 'cameras', name);
  box.position.y = -u(4.5);
  const lens = pickable(new THREE.Mesh(new THREE.CylinderGeometry(u(2.8), u(3.3), u(4), 20), solid(COLOURS.lens, ghost)), 'cameras', name);
  lens.position.y = u(2);
  aim.add(box, lens);
  const stand = solid(COLOURS.stand, ghost);
  const foot = Math.min(u(45), height * 0.4);
  group.add(pickable(rod([0, 0, -u(4)], [0, 0, -height + foot], u(1.1), stand), 'cameras', name));
  for (let index = 0; index < 3; index += 1) {
    const angle = (index * 2 * Math.PI) / 3 + Math.PI / 6;
    group.add(rod([0, 0, -height + foot], [Math.cos(angle) * u(22), Math.sin(angle) * u(22), -height], u(0.8), stand));
  }
  if (entry.facing != null) aim.add(frustum(entry.lens || 60, aspectOf(name), entry.lens != null, colour));
  return { group, aim, anchor: new THREE.Vector3(entry.x, entry.y, height + u(12)), reach: u(24), top: height + u(5) };
}

function buildMicrophone(name, entry) {
  const group = new THREE.Group();
  const ghost = entry.height == null;
  const height = levelOf('microphones', entry);
  group.position.set(entry.x, entry.y, height);
  const head = new THREE.CapsuleGeometry(u(2.2), u(5), 6, 14);
  head.rotateX(Math.PI / 2);
  group.add(pickable(new THREE.Mesh(head, solid(deviceColour('microphones', entry), ghost)), 'microphones', name));
  const stand = solid(COLOURS.stand, ghost);
  group.add(pickable(rod([0, 0, -u(4)], [0, 0, -height], u(0.8), stand), 'microphones', name));
  const base = new THREE.Mesh(new THREE.CylinderGeometry(u(9), u(10), u(1.5), 24), stand);
  base.rotation.x = Math.PI / 2;
  base.position.z = -height + u(0.75);
  group.add(pickable(base, 'microphones', name));
  return { group, anchor: new THREE.Vector3(entry.x, entry.y, height + u(8)), reach: u(14), top: height + u(5) };
}

function buildThing(name, entry) {
  const ghost = entry.height == null || entry.width == null || entry.depth == null;
  const height = levelOf('things', entry);
  const width = entry.width != null ? entry.width : u(40);
  const depth = entry.depth != null ? entry.depth : u(40);
  const group = new THREE.Group();
  const geometry = new THREE.BoxGeometry(width, depth, height);
  const box = pickable(new THREE.Mesh(geometry, solid(COLOURS.things, false, { opacity: ghost ? 0.45 : 0.82 })), 'things', name);
  box.receiveShadow = true;
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), new THREE.LineBasicMaterial({ color: S.theme.faint }));
  group.add(box, edges);
  group.position.set(entry.x, entry.y, height / 2);
  return { group, anchor: new THREE.Vector3(entry.x, entry.y, height + u(6)), reach: Math.hypot(width, depth) / 2 + u(10), top: height };
}

const BUILD = { people: buildPerson, cameras: buildCamera, microphones: buildMicrophone, things: buildThing };

function buildItem(kind, name) {
  const key = keyOf(kind, name);
  const old = S.objects.get(key);
  if (old) {
    S.items.remove(old.group);
    dispose(old.group);
    S.objects.delete(key);
  }
  const entry = S.model[kind][name];
  if (!placed(entry)) return;
  const built = BUILD[kind](name, entry);
  built.kind = kind;
  built.name = name;
  S.items.add(built.group);
  S.objects.set(key, built);
}

function rebuild() {
  for (const built of S.objects.values()) {
    S.items.remove(built.group);
    dispose(built.group);
  }
  S.objects.clear();
  clear(S.fixtures);
  if (!S.model) {
    clear(S.handles);
    return;
  }
  readTheme();
  buildFixtures();
  for (const kind of KINDS) for (const name of Object.keys(S.model[kind] || {})) buildItem(kind, name);
  showSelection();
}

// ---------- the chosen one ----------

function chosen() {
  if (!S.selected || !S.model) return null;
  const { kind, name } = S.selected;
  if (!KINDS.includes(kind) || !S.model[kind] || !S.model[kind][name]) return null;
  return { kind, name, entry: S.model[kind][name], built: S.objects.get(keyOf(kind, name)) };
}

function handleMaterial(opacity = 1) {
  return new THREE.MeshBasicMaterial({ color: COLOURS.pick, transparent: true, opacity, depthTest: opacity > 0, depthWrite: false });
}

function onLayer(object, layer) {
  object.traverse((child) => child.layers.set(layer));
  return object;
}

function showSelection() {
  clear(S.handles);
  for (const built of S.objects.values()) {
    const picked = S.selected && built.kind === S.selected.kind && built.name === S.selected.name;
    built.group.traverse((child) => {
      if (child.isMesh && child.material && child.material.emissive) {
        child.material.emissive.set(picked ? COLOURS.pick : 0x000000);
        child.material.emissiveIntensity = picked ? 0.28 : 0;
      }
    });
  }
  const now = chosen();
  if (!now || !now.built) return;
  const { kind, name, entry, built } = now;
  const at = new THREE.Vector3(entry.x, entry.y, 0);
  const halo = new THREE.Mesh(new THREE.RingGeometry(built.reach, built.reach + u(1.6), 64), handleMaterial(0.9));
  halo.position.copy(at).setZ(u(0.4));
  S.handles.add(halo);
  if (kind === 'people' || kind === 'cameras') {
    const level = kind === 'cameras' ? levelOf(kind, entry) : u(0.6);
    const radius = kind === 'cameras' ? u(26) : built.reach + u(9);
    const turn = new THREE.Group();
    turn.position.set(entry.x, entry.y, level);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(radius, u(1.1), 10, 72), handleMaterial(entry.facing == null ? 0.45 : 0.95));
    const grip = new THREE.Mesh(new THREE.TorusGeometry(radius, u(5), 6, 48), handleMaterial(0));
    const knob = new THREE.Mesh(new THREE.SphereGeometry(u(6), 20, 12), handleMaterial(entry.facing == null ? 0.5 : 1));
    const facing = rad(entry.facing || 0);
    knob.position.set(Math.sin(facing) * radius, Math.cos(facing) * radius, 0);
    for (const mesh of [ring, grip, knob]) mesh.userData = { kind, name, part: 'turn' };
    turn.add(ring, grip, knob);
    S.handles.add(onLayer(turn, TURN_LAYER));
  }
  if (kind === 'cameras') {
    const knob = new THREE.Mesh(new THREE.SphereGeometry(u(5.5), 20, 12), handleMaterial(1));
    knob.userData = { kind, name, part: 'tilt' };
    built.aim.updateWorldMatrix(true, false);
    knob.position.copy(new THREE.Vector3(0, u(42), 0).applyMatrix4(built.aim.matrixWorld));
    const stick = rod([0, 0, 0], [0, u(42), 0], u(0.5), handleMaterial(0.8));
    stick.applyMatrix4(built.aim.matrixWorld);
    const tilt = new THREE.Group();
    tilt.add(knob, stick);
    S.handles.add(onLayer(tilt, TILT_LAYER));
  }
  const lift = new THREE.Mesh(new THREE.ConeGeometry(u(5), u(11), 18), handleMaterial(1));
  lift.rotation.x = Math.PI / 2;
  lift.position.set(entry.x, entry.y, built.top + u(kind === 'people' ? 16 : 14));
  lift.userData = { kind, name, part: 'lift' };
  const post = rod([entry.x, entry.y, built.top + u(2)], [entry.x, entry.y, built.top + u(11)], u(0.5), handleMaterial(0.8));
  const liftGroup = new THREE.Group();
  liftGroup.add(lift, post);
  S.handles.add(onLayer(liftGroup, LIFT_LAYER));
}

function choose(selected) {
  S.selected = selected;
  showSelection();
  tell('room-selected', selected);
  invalidate();
}

// ---------- views ----------

function makeView(name, el) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  renderer.shadowMap.enabled = name === 'main';
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  el.innerHTML = '';
  const backdrop = document.createElement('img');
  backdrop.className = 'room-backdrop';
  backdrop.alt = '';
  el.appendChild(backdrop);
  el.appendChild(renderer.domElement);
  renderer.domElement.tabIndex = 0;
  const labels = document.createElement('div');
  labels.className = 'room-labels';
  el.appendChild(labels);
  const title = document.createElement('div');
  title.className = 'room-view-title';
  title.textContent = name === 'main' ? '' : FLATS[name].title;
  const hint = document.createElement('div');
  hint.className = 'room-hint';
  hint.textContent = "Set the room's size in the list first.";
  el.appendChild(hint);
  el.appendChild(title);
  const fit = document.createElement('button');
  fit.className = 'button small room-fit';
  fit.type = 'button';
  fit.textContent = 'Fit';
  el.appendChild(fit);
  const kind = name === 'plan' ? 'top' : name;
  const view = { name, kind, el, renderer, labels, title, fit, backdrop, hint, tags: new Map(), width: 0, height: 0 };
  if (name === 'main') {
    view.camera = new THREE.PerspectiveCamera(42, 1, 1, FAR);
    view.camera.up.set(0, 0, 1);
    view.look = new THREE.PerspectiveCamera(50, 1, 1, FAR);
    for (const layer of [TURN_LAYER, LIFT_LAYER, TILT_LAYER, DEPTH_LAYER]) view.camera.layers.enable(layer);
    view.controls = new OrbitControls(view.camera, renderer.domElement);
    view.controls.maxPolarAngle = Math.PI * 0.495;
    view.controls.zoomToCursor = true;
  } else {
    view.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 1, FAR);
    if (kind === 'top') {
      view.camera.up.set(0, 1, 0);
      view.camera.layers.enable(TURN_LAYER);
    } else {
      view.camera.up.set(0, 0, 1);
      view.camera.layers.enable(TILT_LAYER);
    }
    view.svg = document.createElementNS(SVGNS, 'svg');
    view.svg.setAttribute('class', 'room-paper');
    el.insertBefore(view.svg, labels);
    view.controls = new OrbitControls(view.camera, el);
    view.controls.enableRotate = false;
    view.controls.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    view.controls.zoomToCursor = true;
    view.controls.screenSpacePanning = true;
    const legend = document.createElement('div');
    legend.className = 'room-axes';
    legend.textContent = `${FLATS[name].across}   ${FLATS[name].up}`;
    el.appendChild(legend);
    view.bar = document.createElement('div');
    view.bar.className = 'room-scale';
    el.appendChild(view.bar);
  }
  view.controls.addEventListener('change', invalidate);
  fit.addEventListener('click', () => {
    if (name === 'main') fitMain(view);
    else if (name === 'plan') fitPlan(view);
    else fitFlats(true);
    invalidate();
  });
  view.surface = name === 'main' ? renderer.domElement : el;
  const surface = view.surface;
  surface.addEventListener('pointerdown', (event) => pointerDown(view, event), { capture: true });
  surface.addEventListener('pointermove', (event) => pointerMove(view, event));
  surface.addEventListener('pointerup', (event) => pointerUp(view, event));
  surface.addEventListener('pointercancel', (event) => pointerUp(view, event));
  surface.addEventListener('pointerleave', () => {
    if (S.ghost && S.ghost.view === name) {
      S.ghost = null;
      invalidate();
    }
  });
  // an entry of the list dropped here is put where it lands, like a pin
  el.addEventListener('dragover', (event) => {
    if (event.dataTransfer && event.dataTransfer.types.includes(DROP)) event.preventDefault();
  });
  el.addEventListener('drop', (event) => {
    const carried = event.dataTransfer && event.dataTransfer.getData(DROP);
    if (!carried || !S.model) return;
    event.preventDefault();
    const { kind: what, name: which } = JSON.parse(carried);
    const point = floorAt(view, event);
    if (!point || !S.model[what] || !S.model[what][which]) return;
    if (!S.selected || S.selected.kind !== what || S.selected.name !== which) choose({ kind: what, name: which });
    putAt(what, which, point);
  });
  if (name === 'main') surface.addEventListener('wheel', (event) => wheel(view, event), { passive: false, capture: true });
  view.observer = new ResizeObserver(() => resize(view));
  view.observer.observe(el);
  return view;
}

function resize(view) {
  const width = view.el.clientWidth, height = view.el.clientHeight;
  if (!width || !height) return;
  const first = !view.width;
  view.width = width;
  view.height = height;
  view.renderer.setSize(width, height);
  if (view.name === 'main') {
    view.camera.aspect = width / height;
    view.camera.updateProjectionMatrix();
    if (first && S.model) fitMain(view);
  } else if (view.name === 'plan') {
    if (view.scale) keepScale(view);
    else if (S.model) fitPlan(view);
  } else if (S.flatScale && !first) {
    keepScale(view);
  } else if (S.model) {
    fitFlats(false);
  }
  invalidate();
}

function fitMain(view) {
  if (!S.model) return;
  const box = extent();
  const middle = new THREE.Vector3((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, box.top * 0.22);
  const span = Math.max(box.x1 - box.x0, box.y1 - box.y0);
  view.controls.target.copy(middle);
  // a view narrower than usual stands further back, so the room's width still fits
  const back = span * 1.45 * Math.max(1, 1.5 / view.camera.aspect);
  view.camera.position.copy(middle).add(new THREE.Vector3(-0.42, -1.0, 0.78).normalize().multiplyScalar(back));
  view.camera.near = span / 500;
  view.camera.far = span * 100;
  view.camera.updateProjectionMatrix();
  view.controls.update();
}

function flatFrame(name, box) {
  const high = box.top + u(20);
  if (name === 'top') return { across: box.x1 - box.x0, up: box.y1 - box.y0, middle: [(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, 0] };
  if (name === 'front') return { across: box.x1 - box.x0, up: high + u(20), middle: [(box.x0 + box.x1) / 2, box.y0, high / 2 - u(5)] };
  return { across: box.y1 - box.y0, up: high + u(20), middle: [box.x1, (box.y0 + box.y1) / 2, high / 2 - u(5)] };
}

function fitFlats(force) {
  if (!S.model) return;
  const box = extent();
  const flats = ['top', 'front', 'side'].map((name) => S.views[name]).filter((view) => view && view.width);
  if (!flats.length) return;
  // one scale for all three, so a length is as long in each
  S.flatScale = Math.min(...flats.map((view) => {
    const frame = flatFrame(view.name, box);
    return Math.min(view.width / frame.across, view.height / frame.up) * 0.9;
  }));
  for (const view of flats) {
    const frame = flatFrame(view.name, box);
    const [x, y, z] = frame.middle;
    const distance = Math.max(box.x1 - box.x0, box.y1 - box.y0, box.top) * 20;
    if (view.name === 'top') view.camera.position.set(x, y, distance);
    if (view.name === 'front') view.camera.position.set(x, y - distance, z);
    if (view.name === 'side') view.camera.position.set(x + distance, y, z);
    view.controls.target.set(x, y, z);
    view.camera.near = distance / 100;
    view.camera.far = distance * 3;
    view.camera.zoom = 1;
    view.camera.lookAt(x, y, z);
    keepScale(view);
    view.controls.update();
  }
  if (force) invalidate();
}

function fitPlan(view) {
  if (!S.model || !view || !view.width) return;
  const box = extent();
  const frame = flatFrame('top', box);
  const [x, y, z] = frame.middle;
  const distance = Math.max(box.x1 - box.x0, box.y1 - box.y0, box.top) * 20;
  view.scale = Math.min(view.width / frame.across, view.height / frame.up) * 0.9;
  view.camera.position.set(x, y, distance);
  view.controls.target.set(x, y, z);
  view.camera.near = distance / 100;
  view.camera.far = distance * 3;
  view.camera.zoom = 1;
  view.camera.lookAt(x, y, z);
  keepScale(view);
  view.controls.update();
}

function keepScale(view) {
  const scale = view.scale || S.flatScale || 1;
  Object.assign(view.camera, { left: -view.width / (2 * scale), right: view.width / (2 * scale), top: view.height / (2 * scale), bottom: -view.height / (2 * scale) });
  view.camera.updateProjectionMatrix();
}

function lookFrom(view) {
  // the main view as the chosen camera of the setup sees the room, over a frame of its file when there is one
  const name = S.frame && S.frame.camera;
  const entry = name && S.model && S.model.cameras[name];
  if (S.mode !== 'camera' || !entry || !placed(entry)) return null;
  const aspect = aspectOf(name);
  const lens = entry.lens || 60;
  const look = view.look;
  const height = levelOf('cameras', entry);
  const facing = rad(entry.facing || 0), tilt = rad(entry.tilt || 0);
  const heading = new THREE.Vector3(Math.sin(facing), Math.cos(facing), 0);
  const ahead = heading.clone().multiplyScalar(Math.cos(tilt)).add(new THREE.Vector3(0, 0, -Math.sin(tilt)));
  const right = new THREE.Vector3(heading.y, -heading.x, 0);
  look.position.set(entry.x, entry.y, height);
  look.up.copy(new THREE.Vector3().crossVectors(right, ahead));
  look.lookAt(look.position.clone().add(ahead));
  look.aspect = aspect;
  look.fov = deg(2 * Math.atan(Math.tan(rad(lens) / 2) / aspect));
  look.near = u(5);
  look.far = u(100000);
  look.updateProjectionMatrix();
  // the picture's shape inside the view, so the drawing lies on the frame
  let width = view.width, height2 = view.width / aspect;
  if (height2 > view.height) {
    height2 = view.height;
    width = view.height * aspect;
  }
  return { camera: look, name, width: Math.round(width), height: Math.round(height2), left: Math.round((view.width - width) / 2), top: Math.round((view.height - height2) / 2) };
}

function applyMode() {
  const planning = S.mode === 'plan';
  if (S.views.plan) S.views.plan.el.style.display = planning ? '' : 'none';
  const view = S.views.main;
  if (!view) return;
  view.el.style.display = planning ? 'none' : '';
  const looking = lookFrom(view);
  view.looking = looking;
  view.controls.enabled = !looking;
  view.el.classList.toggle('looking', !!looking);
  view.fit.style.display = looking ? 'none' : '';
  const uri = looking && S.frame && S.frame.uri;
  view.backdrop.style.display = uri ? 'block' : 'none';
  if (uri && view.backdrop.getAttribute('src') !== uri) view.backdrop.src = uri;
  view.title.textContent = looking ? `What ${looking.name} sees` : '';
}

function invalidate() {
  if (S.queued) return;
  S.queued = true;
  requestAnimationFrame(() => {
    S.queued = false;
    for (const view of Object.values(S.views)) draw(view);
  });
}

function drawn(view) {
  return !!view.svg && (view.name === 'plan' || S.style === 'paper');
}

function draw(view) {
  if (!view.width || !view.height || !view.el.isConnected) return;
  if (view.hint) {
    const walls = (S.model && S.model.walls) || {};
    view.hint.style.display = S.model && (walls.width == null || walls.length == null) && !view.looking && view.kind !== 'front' && view.kind !== 'side' ? '' : 'none';
  }
  if (view.svg) {
    const paper = drawn(view);
    view.svg.style.display = paper ? 'block' : 'none';
    view.renderer.domElement.style.display = paper ? 'none' : 'block';
    view.labels.style.display = paper ? 'none' : '';
    if (view.bar) view.bar.style.display = paper ? 'none' : '';
    view.el.style.background = paper ? 'var(--panel)' : '';
    if (paper) {
      drawPaper(view);
      return;
    }
    view.svg.replaceChildren();
  }
  const looking = view.name === 'main' ? view.looking : null;
  const camera = looking ? looking.camera : view.camera;
  const canvas = view.renderer.domElement;
  if (looking) {
    view.renderer.setSize(looking.width, looking.height);
    Object.assign(canvas.style, { position: 'absolute', left: `${looking.left}px`, top: `${looking.top}px` });
    Object.assign(view.backdrop.style, { left: `${looking.left}px`, top: `${looking.top}px`, width: `${looking.width}px`, height: `${looking.height}px` });
  } else if (canvas.style.position === 'absolute') {
    view.renderer.setSize(view.width, view.height);
    Object.assign(canvas.style, { position: '', left: '', top: '' });
  }
  // walls between the viewer and the room are left out, so they hide nothing
  const eye = camera.position;
  for (const wall of S.walls) wall.group.visible = eye.clone().sub(wall.point).dot(wall.normal) <= 0;
  if (S.floor) S.floor.visible = !looking;
  const own = looking ? S.objects.get(keyOf('cameras', looking.name)) : null;
  if (own) own.group.visible = false;
  const ghost = S.ghost && S.ghost.view === view.name ? ghostMark() : null;
  if (ghost) {
    ghost.position.set(S.ghost.point.x, S.ghost.point.y, u(0.5));
    ghost.visible = true;
  }
  view.renderer.render(S.scene, camera);
  if (ghost) ghost.visible = false;
  if (own) own.group.visible = true;
  placeLabels(view, camera, looking);
  if (view.bar) showScale(view);
}

function ghostMark() {
  // where the chosen entry would go, under the pointer
  if (!S.ghostMesh) {
    S.ghostMesh = new THREE.Mesh(new THREE.RingGeometry(0.8, 1, 48), handleMaterial(0.85));
    S.ghostMesh.visible = false;
    S.scene.add(S.ghostMesh);
  }
  S.ghostMesh.scale.setScalar(u(20));
  return S.ghostMesh;
}

function placeLabels(view, camera, looking) {
  const seen = new Set();
  const width = looking ? looking.width : view.width, height = looking ? looking.height : view.height;
  const left = looking ? looking.left : 0, top = looking ? looking.top : 0;
  for (const [key, built] of S.objects) {
    if (looking && built.kind === 'cameras' && built.name === looking.name) continue;
    const at = built.anchor.clone().project(camera);
    if (at.z < -1 || at.z > 1 || Math.abs(at.x) > 1.05 || Math.abs(at.y) > 1.05) continue;
    let tag = view.tags.get(key);
    if (!tag) {
      tag = document.createElement('span');
      tag.className = 'room-label';
      view.labels.appendChild(tag);
      view.tags.set(key, tag);
    }
    tag.textContent = built.name;
    tag.classList.toggle('chosen', !!S.selected && S.selected.kind === built.kind && S.selected.name === built.name);
    tag.style.left = `${left + ((at.x + 1) / 2) * width}px`;
    tag.style.top = `${top + ((1 - at.y) / 2) * height}px`;
    seen.add(key);
  }
  for (const [key, tag] of view.tags) {
    if (!seen.has(key)) {
      tag.remove();
      view.tags.delete(key);
    }
  }
}

function showScale(view) {
  const perUnit = (view.width / (view.camera.right - view.camera.left)) * view.camera.zoom;
  const lengths = SCALE_BARS[units()];
  const length = lengths.filter((value) => value * perUnit <= 130).pop() || lengths[0];
  view.bar.innerHTML = `<span class="room-scale-bar" style="width:${Math.round(length * perUnit)}px"></span>${length} ${units()}`;
}

// ---------- the flat drawings, as in the drawing for a paper ----------

function node(tag, attrs, parent, text) {
  const made = document.createElementNS(SVGNS, tag);
  for (const [key, value] of Object.entries(attrs || {})) if (value != null) made.setAttribute(key, value);
  if (text != null) made.textContent = text;
  if (parent) parent.appendChild(made);
  return made;
}

function arrowMark(parent, from, direction, length, className) {
  const tip = [from[0] + direction[0] * length, from[1] + direction[1] * length];
  const head = Math.min(7, length * 0.45);
  const back = [tip[0] - direction[0] * head, tip[1] - direction[1] * head];
  const side = [-direction[1] * head * 0.5, direction[0] * head * 0.5];
  node('line', { x1: from[0], y1: from[1], x2: back[0], y2: back[1], class: className }, parent);
  node('polygon', { points: [tip, [back[0] + side[0], back[1] + side[1]], [back[0] - side[0], back[1] - side[1]]].map((p) => p.join(',')).join(' '), class: `${className} head` }, parent);
  return tip;
}

function drawPaper(view) {
  const svg = view.svg;
  svg.replaceChildren();
  if (!S.model) return;
  svg.setAttribute('width', view.width);
  svg.setAttribute('height', view.height);
  const camera = view.camera;
  camera.updateMatrixWorld();
  const project = (x, y, z = 0) => {
    const at = new THREE.Vector3(x, y, z).project(camera);
    return [((at.x + 1) / 2) * view.width, ((1 - at.y) / 2) * view.height, at.z];
  };
  const per = (view.width / (camera.right - camera.left)) * camera.zoom;
  const size = (cm, least) => Math.max(u(cm) * per, least);
  const model = S.model;
  const walls = model.walls || {};
  const hasWalls = walls.width != null && walls.length != null;
  const box = extent();
  const picked = (kind, name) => S.selected && S.selected.kind === kind && S.selected.name === name;
  const grab = (kind, name, part = 'body') => ({ 'data-kind': kind, 'data-name': name, 'data-part': part });
  const marked = (base, kind, name) => (picked(kind, name) ? `${base} chosen` : base);
  const layer = {};
  for (const name of ['back', 'things', 'lines', 'people', 'devices', 'words', 'knobs']) layer[name] = node('g', {}, svg);
  const reads = {};
  for (const [name, entry] of Object.entries(model.cameras)) reads[name] = [...new Set([...(entry.body || []), ...(entry.face || []), ...(entry.gaze || [])])];

  if (view.kind === 'top') {
    if (hasWalls) {
      const corners = [[0, 0], [walls.width, 0], [walls.width, walls.length], [0, walls.length]].map(([x, y]) => project(x, y).slice(0, 2).join(','));
      // what a camera sees ends at the walls
      const clip = node('clipPath', { id: `room-walls-${view.name}` }, node('defs', {}, svg));
      node('polygon', { points: corners.join(' ') }, clip);
      layer.back.setAttribute('clip-path', `url(#room-walls-${view.name})`);
      node('polygon', { points: corners.join(' '), class: 'wall' }, layer.things);
    }
    const fallback = 0.45 * Math.hypot(box.x1 - box.x0, box.y1 - box.y0);
    for (const [name, entry] of Object.entries(model.cameras)) {
      if (!placed(entry) || entry.facing == null) continue;
      const lens = entry.lens || 60;
      const near = (reads[name] || []).map((person) => model.people[person]).filter(placed).map((person) => Math.hypot(person.x - entry.x, person.y - entry.y));
      const reach = near.length ? 1.3 * Math.max(...near) : fallback;
      const points = [project(entry.x, entry.y)];
      for (let step = 0; step <= 24; step += 1) {
        const angle = rad(entry.facing - lens / 2 + (lens * step) / 24);
        points.push(project(entry.x + reach * Math.sin(angle), entry.y + reach * Math.cos(angle)));
      }
      node('polygon', { points: points.map((p) => p.slice(0, 2).join(',')).join(' '), class: entry.lens == null ? 'fan unknown' : 'fan' }, layer.back);
    }
    for (const [name, entry] of Object.entries(model.things)) {
      if (!placed(entry)) continue;
      const width = entry.width != null ? entry.width : u(40), depth = entry.depth != null ? entry.depth : u(40);
      const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([sx, sy]) => project(entry.x + (sx * width) / 2, entry.y + (sy * depth) / 2));
      node('polygon', { points: corners.map((p) => p.slice(0, 2).join(',')).join(' '), class: marked('thing', 'things', name), ...grab('things', name) }, layer.things);
      // in a corner, where people sitting at its sides leave the name readable
      node('text', { x: corners[0][0] + 4, y: corners[0][1] - 4, class: 'thing-name' }, layer.words, name);
    }
    const radius = size(PAPER_CM.person, 9);
    for (const [name, entry] of Object.entries(model.people)) {
      if (!placed(entry)) continue;
      const [x, y] = project(entry.x, entry.y);
      node('circle', { cx: x, cy: y, r: radius, class: marked('person', 'people', name), ...grab('people', name) }, layer.people);
      const facing = rad(entry.facing || 0);
      const direction = [Math.sin(facing), -Math.cos(facing)];
      const fits = (2 * radius * 0.85) / (0.6 * Math.max(name.length, 1));
      if (fits >= 9) node('text', { x, y, class: 'name inside', 'font-size': Math.min(13, fits) }, layer.words, name);
      else {
        // behind the person, so the facing arrow stays clear
        const away = entry.facing != null ? [-direction[0], -direction[1]] : [1, 0];
        const anchor = away[0] > 0.38 ? 'start' : away[0] < -0.38 ? 'end' : 'middle';
        const baseline = away[1] > 0.38 ? 'hanging' : away[1] < -0.38 ? 'auto' : 'middle';
        node('text', { x: x + away[0] * (radius + 4), y: y + away[1] * (radius + 4), class: 'name beside', style: `text-anchor:${anchor};dominant-baseline:${baseline}` }, layer.words, name);
      }
      const start = [x + direction[0] * radius * 1.15, y + direction[1] * radius * 1.15];
      const tip = entry.facing != null ? arrowMark(layer.people, start, direction, size(PAPER_CM.arrow, 12), 'arrow') : [start[0] + direction[0] * 14, start[1] + direction[1] * 14];
      if (picked('people', name)) node('circle', { cx: tip[0], cy: tip[1], r: 6.5, class: entry.facing == null ? 'knob unset' : 'knob', ...grab('people', name, 'turn') }, layer.knobs);
    }
    const square = size(PAPER_CM.device, 10);
    for (const [name, entry] of Object.entries(model.cameras)) {
      if (!placed(entry)) continue;
      const [x, y] = project(entry.x, entry.y);
      node('rect', { x: x - square / 2, y: y - square / 2, width: square, height: square, class: marked('camera', 'cameras', name), ...grab('cameras', name) }, layer.devices);
      const facing = rad(entry.facing || 0);
      const direction = [Math.sin(facing), -Math.cos(facing)];
      const start = [x + direction[0] * square * 0.7, y + direction[1] * square * 0.7];
      const tip = entry.facing != null ? arrowMark(layer.devices, start, direction, size(PAPER_CM.arrow, 12), 'camera-arrow') : [start[0] + direction[0] * 14, start[1] + direction[1] * 14];
      const behind = entry.facing != null ? [-direction[0], -direction[1]] : [0, 1];
      node('text', { x: x + behind[0] * (square + 9), y: y + behind[1] * (square + 9), class: 'device-name' }, layer.words, name);
      if (picked('cameras', name)) node('circle', { cx: tip[0], cy: tip[1], r: 6.5, class: entry.facing == null ? 'knob unset' : 'knob', ...grab('cameras', name, 'turn') }, layer.knobs);
    }
    const dot = size(PAPER_CM.device / 2, 5);
    for (const [name, entry] of Object.entries(model.microphones)) {
      if (!placed(entry)) continue;
      const [x, y] = project(entry.x, entry.y);
      node('circle', { cx: x, cy: y, r: dot, class: marked('microphone', 'microphones', name), ...grab('microphones', name) }, layer.devices);
      node('text', { x: x + dot + 4, y, class: 'name beside' }, layer.words, name);
    }
  } else {
    const front = view.kind === 'front';
    const along = (entry) => (front ? entry.x : entry.y);
    const at = (a, z) => (front ? project(a, box.y0, z) : project(box.x1, a, z));
    const [low, high] = front ? [box.x0, box.x1] : [box.y0, box.y1];
    const top = Math.max(u(250), box.top);
    const floor = [at(low, 0), at(high, 0)];
    node('line', { x1: floor[0][0], y1: floor[0][1], x2: floor[1][0], y2: floor[1][1], class: 'floor' }, layer.back);
    if (hasWalls) {
      for (const wall of [0, front ? walls.width : walls.length]) {
        const [a, b] = [at(wall, 0), at(wall, top)];
        node('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], class: 'wall' }, layer.back);
      }
    }
    // farther first, so what is nearer is drawn over it
    const depth = (entry) => project(entry.x, entry.y)[2];
    const things = Object.entries(model.things).filter(([, entry]) => placed(entry)).sort((a, b) => depth(b[1]) - depth(a[1]));
    for (const [name, entry] of things) {
      const height = levelOf('things', entry);
      const width = front ? (entry.width != null ? entry.width : u(40)) : (entry.depth != null ? entry.depth : u(40));
      const [a, b] = [at(along(entry) - width / 2, height), at(along(entry) + width / 2, 0)];
      node('rect', { x: a[0], y: a[1], width: b[0] - a[0], height: b[1] - a[1], class: marked(entry.height == null ? 'thing unset' : 'thing', 'things', name), ...grab('things', name) }, layer.things);
      node('text', { x: a[0] + 4, y: a[1] + 13, class: 'thing-name' }, layer.words, name);
    }
    const head = size(PAPER_CM.head, 6);
    const heads = {};
    const people = Object.entries(model.people).filter(([, entry]) => placed(entry)).sort((a, b) => depth(b[1]) - depth(a[1]));
    for (const [name, entry] of people) {
      const eyes = levelOf('people', entry);
      const [x, y] = at(along(entry), eyes);
      const foot = at(along(entry), 0);
      heads[name] = [x, y];
      const group = node('g', { class: entry.eyes == null ? 'unset' : null }, layer.people);
      node('line', { x1: foot[0], y1: foot[1], x2: x, y2: y + head, class: 'body', 'stroke-width': size(PAPER_CM.body, 3), ...grab('people', name) }, group);
      node('circle', { cx: x, cy: y, r: head, class: marked('person', 'people', name), ...grab('people', name) }, group);
      node('text', { x, y: y - head - 6, class: 'name above' }, layer.words, name);
    }
    const square = size(PAPER_CM.device, 10);
    for (const [name, entry] of Object.entries(model.cameras)) {
      if (!placed(entry)) continue;
      const [x, y] = at(along(entry), levelOf('cameras', entry));
      for (const person of reads[name] || []) {
        if (heads[person]) node('line', { x1: x, y1: y, x2: heads[person][0], y2: heads[person][1], class: 'sight' }, layer.lines);
      }
      if (entry.facing != null) {
        // where it points, as this drawing shows it: its tilt seen from the side
        const facing = rad(entry.facing), tilt = rad(entry.tilt || 0);
        const ahead = new THREE.Vector3(Math.sin(facing) * Math.cos(tilt), Math.cos(facing) * Math.cos(tilt), -Math.sin(tilt));
        const end = front ? project(entry.x + ahead.x, box.y0, levelOf('cameras', entry) + ahead.z) : project(box.x1, entry.y + ahead.y, levelOf('cameras', entry) + ahead.z);
        const flat = [end[0] - x, end[1] - y];
        const length = Math.hypot(...flat);
        if (length > 1e-6 * per && length / per > 0.25) arrowMark(layer.devices, [x + (flat[0] / length) * square * 0.7, y + (flat[1] / length) * square * 0.7], [flat[0] / length, flat[1] / length], size(PAPER_CM.arrow, 12) * Math.min(1, length / per), 'camera-arrow');
      }
      node('rect', { x: x - square / 2, y: y - square / 2, width: square, height: square, class: marked(entry.height == null ? 'camera unset' : 'camera', 'cameras', name), ...grab('cameras', name) }, layer.devices);
      node('text', { x, y: y - square / 2 - 6, class: 'device-name above' }, layer.words, name);
    }
    const dot = size(PAPER_CM.device / 2, 5);
    for (const [name, entry] of Object.entries(model.microphones)) {
      if (!placed(entry)) continue;
      const [x, y] = at(along(entry), levelOf('microphones', entry));
      node('circle', { cx: x, cy: y, r: dot, class: marked(entry.height == null ? 'microphone unset' : 'microphone', 'microphones', name), ...grab('microphones', name) }, layer.devices);
      node('text', { x: x + dot + 4, y, class: 'name beside' }, layer.words, name);
    }
  }
  if (S.ghost && S.ghost.view === view.name) {
    const [x, y] = project(S.ghost.point.x, S.ghost.point.y);
    node('circle', { cx: x, cy: y, r: size(PAPER_CM.person, 9), class: 'ghost' }, svg);
  }
  const lengths = SCALE_BARS[units()];
  const length = lengths.filter((value) => value * per <= 120).pop() || lengths[0];
  const left = 16, bottom = view.height - 16, across = length * per;
  node('path', { d: `M${left},${bottom - 4}V${bottom}H${left + across}V${bottom - 4}`, class: 'scale' }, svg);
  node('text', { x: left + across / 2, y: bottom - 7, class: 'scale-text' }, svg, `${length} ${units()}`);
}

function paperHit(view, event) {
  const target = event.target instanceof Element ? event.target.closest('[data-kind]') : null;
  if (!target || !S.model) return null;
  const { kind, name, part } = target.dataset;
  const entry = S.model[kind] && S.model[kind][name];
  if (!entry) return null;
  aimAt(view, event, view.camera);
  const normal = view.kind === 'top' ? new THREE.Vector3(0, 0, 1) : view.kind === 'front' ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  const point = onPlane(normal, new THREE.Vector3(entry.x, entry.y, 0));
  return point ? { object: { userData: { kind, name, part } }, point } : null;
}

// ---------- pointing and dragging ----------

const ray = new THREE.Raycaster();

function aimAt(view, event, camera) {
  const rect = (view.name === 'main' ? view.renderer.domElement : view.el).getBoundingClientRect();
  const point = new THREE.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
  ray.setFromCamera(point, camera);
  ray.layers.mask = camera.layers.mask;
  return ray;
}

function hitAt(view, event) {
  if (drawn(view)) return paperHit(view, event);
  const camera = view.looking ? view.looking.camera : view.camera;
  aimAt(view, event, camera);
  const handles = ray.intersectObjects(S.handles.children, true).filter((hit) => hit.object.userData.part);
  if (handles.length) return handles[0];
  const items = ray.intersectObjects(S.items.children, true).filter((hit) => hit.object.userData.kind);
  if (view.looking) return items.find((hit) => !(hit.object.userData.kind === 'cameras' && hit.object.userData.name === view.looking.name)) || null;
  return items[0] || null;
}

function onPlane(normal, point) {
  const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(normal, point);
  const found = new THREE.Vector3();
  return ray.ray.intersectPlane(plane, found) ? found : null;
}

function placing() {
  const now = chosen();
  return now && !placed(now.entry) ? now : null;
}

function floorAt(view, event) {
  // the floor under the pointer, in a view that shows the floor
  if (view.kind === 'front' || view.kind === 'side') return null;
  aimAt(view, event, view.looking ? view.looking.camera : view.camera);
  return onPlane(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0));
}

function putAt(kind, name, point) {
  const fields = { x: round(point.x), y: round(point.y) };
  Object.assign(S.model[kind][name], fields);
  S.ghost = null;
  buildItem(kind, name);
  showSelection();
  invalidate();
  tell('room-moved', { kind, name, fields, at: Date.now() });
}

function pointerDown(view, event) {
  S.active = true;
  if (event.button !== 0 || !S.model) return;
  const start = { x: event.clientX, y: event.clientY };
  const pending = placing();
  if (pending) {
    const point = floorAt(view, event);
    if (point) {
      // a click on the floor pins the chosen entry there
      putAt(pending.kind, pending.name, point);
      view.controls.enabled = false;
      S.drag = { view, part: 'none', start, moved: true };
      event.preventDefault();
      return;
    }
  }
  const hit = hitAt(view, event);
  if (!hit) {
    if (view.looking) {
      const entry = S.model.cameras[view.looking.name];
      S.drag = { view, part: 'aim', kind: 'cameras', name: view.looking.name, start, before: { ...entry }, moved: false };
      view.surface.setPointerCapture(event.pointerId);
    } else {
      S.drag = { view, part: 'none', start, moved: false };
    }
    return;
  }
  const { kind, name, part } = hit.object.userData;
  if (!S.selected || S.selected.kind !== kind || S.selected.name !== name) choose({ kind, name });
  const entry = S.model[kind][name];
  S.drag = { view, kind, name, part, start, before: { ...entry }, point: hit.point.clone(), moved: false };
  if (part === 'lift') {
    aimAt(view, event, view.looking ? view.looking.camera : view.camera);
    S.drag.from = upright(entry.x, entry.y);
  }
  view.controls.enabled = false;
  view.surface.setPointerCapture(event.pointerId);
  event.preventDefault();
}

function upright(x, y) {
  // the height on the upright line through (x, y) nearest the pointer's ray
  const origin = ray.ray.origin, direction = ray.ray.direction;
  const across = new THREE.Vector3(x - origin.x, y - origin.y, 0);
  const flat = Math.hypot(direction.x, direction.y);
  if (flat < 1e-6) return null;
  const along = (across.x * direction.x + across.y * direction.y) / (flat * flat);
  return origin.z + along * direction.z;
}

function pointerMove(view, event) {
  const drag = S.drag;
  if (!drag || drag.view !== view) {
    if (drag || !S.model) return;
    if (placing()) {
      const point = floorAt(view, event);
      view.surface.style.cursor = point ? 'crosshair' : 'not-allowed';
      S.ghost = point ? { view: view.name, point } : null;
      invalidate();
      return;
    }
    if (S.ghost) {
      S.ghost = null;
      invalidate();
    }
    if (!drawn(view)) {
      const hit = hitAt(view, event);
      const part = hit && hit.object.userData.part;
      view.surface.style.cursor = !hit ? (view.looking ? 'move' : '') : part === 'body' ? 'grab' : 'pointer';
    }
    return;
  }
  if (Math.hypot(event.clientX - drag.start.x, event.clientY - drag.start.y) > 3) drag.moved = true;
  if (!drag.moved || drag.part === 'none') return;
  const camera = view.looking ? view.looking.camera : view.camera;
  aimAt(view, event, camera);
  const before = drag.before;
  const level = LEVEL[drag.kind];
  let fields = {};
  if (drag.part === 'aim') {
    const perPixel = (before.lens || 60) / (view.looking ? view.looking.width : view.width);
    fields = {
      facing: (((before.facing || 0) - (event.clientX - drag.start.x) * perPixel) % 360 + 360) % 360,
      tilt: Math.max(-89, Math.min(89, (before.tilt || 0) - (event.clientY - drag.start.y) * perPixel)),
    };
    fields.facing = Number(fields.facing.toFixed(1));
    fields.tilt = Number(fields.tilt.toFixed(1));
  } else if (drag.part === 'body') {
    if (view.kind === 'front' || view.kind === 'side') {
      const normal = view.kind === 'front' ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
      const found = onPlane(normal, drag.point);
      if (!found) return;
      const axis = view.kind === 'front' ? 'x' : 'y';
      fields[axis] = round(before[axis] + found[axis] - drag.point[axis]);
      fields[level] = round(Math.max(0, levelOf(drag.kind, before) + found.z - drag.point.z));
    } else {
      const found = onPlane(new THREE.Vector3(0, 0, 1), drag.point);
      if (!found) return;
      fields = { x: round(before.x + found.x - drag.point.x), y: round(before.y + found.y - drag.point.y) };
    }
  } else if (drag.part === 'turn') {
    const height = drag.kind === 'cameras' ? levelOf('cameras', before) : u(0.6);
    const found = onPlane(new THREE.Vector3(0, 0, 1), new THREE.Vector3(before.x, before.y, height));
    if (!found) return;
    let facing = (deg(Math.atan2(found.x - before.x, found.y - before.y)) + 360) % 360;
    facing = event.shiftKey ? Number(facing.toFixed(1)) : (Math.round(facing / 5) * 5) % 360;
    fields = { facing };
  } else if (drag.part === 'tilt') {
    const facing = rad(before.facing || 0);
    const heading = new THREE.Vector3(Math.sin(facing), Math.cos(facing), 0);
    const centre = new THREE.Vector3(before.x, before.y, levelOf('cameras', before));
    const found = onPlane(new THREE.Vector3(heading.y, -heading.x, 0), centre);
    if (!found) return;
    const gap = found.clone().sub(centre);
    const tilt = deg(Math.atan2(-gap.z, gap.dot(heading)));
    fields = { tilt: Math.max(-89, Math.min(89, event.shiftKey ? Number(tilt.toFixed(1)) : Math.round(tilt))) };
  } else if (drag.part === 'lift') {
    const now = upright(before.x, before.y);
    if (now == null || drag.from == null) return;
    fields[level] = round(Math.max(0, levelOf(drag.kind, before) + now - drag.from));
  }
  drag.fields = fields;
  Object.assign(S.model[drag.kind][drag.name], fields);
  buildItem(drag.kind, drag.name);
  showSelection();
  if (drag.part === 'aim' || (view.looking && drag.kind === 'cameras' && drag.name === view.looking.name)) applyMode();
  invalidate();
}

function pointerUp(view, event) {
  const drag = S.drag;
  if (!drag || drag.view !== view) return;
  S.drag = null;
  view.controls.enabled = !view.looking;
  if (view.surface.hasPointerCapture(event.pointerId)) view.surface.releasePointerCapture(event.pointerId);
  if (drag.part === 'none') {
    if (!drag.moved && S.selected) choose(null);
  } else if (drag.moved && drag.fields) {
    tell('room-moved', { kind: drag.kind, name: drag.name, fields: drag.fields, at: Date.now() });
  }
  if (S.waiting) {
    const waiting = S.waiting;
    S.waiting = null;
    update(waiting);
  }
}

function wheel(view, event) {
  if (!view.looking || !S.model) return;
  event.preventDefault();
  event.stopPropagation();
  const name = view.looking.name;
  const entry = S.model.cameras[name];
  const lens = Math.max(10, Math.min(170, (entry.lens || 60) * Math.exp(event.deltaY * 0.0008)));
  entry.lens = Number(lens.toFixed(1));
  buildItem('cameras', name);
  showSelection();
  applyMode();
  invalidate();
  clearTimeout(S.lensTimer);
  S.lensTimer = setTimeout(() => tell('room-moved', { kind: 'cameras', name, fields: { lens: entry.lens }, at: Date.now() }), 350);
}

function keyDown(event) {
  const target = event.target;
  if (target && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName))) return;
  if (event.key === 'Escape' && placing()) {
    S.ghost = null;
    choose(null);
    return;
  }
  if (!S.active || !S.selected || !S.model) return;
  const now = chosen();
  if (!now) return;
  if (event.key === 'Escape') {
    choose(null);
  } else if ((event.key === 'Delete' || event.key === 'Backspace') && placed(now.entry)) {
    event.preventDefault();
    tell('room-moved', { op: 'unplace', kind: now.kind, name: now.name, at: Date.now() });
  } else if ((event.key === 'r' || event.key === 'R') && (now.kind === 'people' || now.kind === 'cameras') && placed(now.entry)) {
    const facing = (((now.entry.facing || 0) + (event.shiftKey ? -15 : 15)) % 360 + 360) % 360;
    now.entry.facing = facing;
    buildItem(now.kind, now.name);
    showSelection();
    applyMode();
    invalidate();
    tell('room-moved', { kind: now.kind, name: now.name, fields: { facing }, at: Date.now() });
  }
}

document.addEventListener('keydown', keyDown);
document.addEventListener('dragstart', (event) => {
  const item = event.target instanceof Element ? event.target.closest('.room-item[data-kind]') : null;
  if (!item || !event.dataTransfer) return;
  event.dataTransfer.setData(DROP, JSON.stringify({ kind: item.dataset.kind, name: item.dataset.name }));
  event.dataTransfer.effectAllowed = 'move';
});
document.addEventListener('pointerdown', (event) => {
  if (!(event.target instanceof Element) || !event.target.closest('.room-canvas')) S.active = false;
}, { capture: true });

function watchTheme() {
  const again = () => {
    if (!S.model) return;
    rebuild();
    invalidate();
  };
  new MutationObserver(again).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'data-style'] });
  if (window.matchMedia) window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', again);
}

function mount() {
  for (const name of ['main', 'plan', 'top', 'front', 'side']) {
    const el = document.getElementById(`room-${name}`);
    const view = S.views[name];
    if (!el || (view && view.el === el && el.isConnected)) continue;
    if (view) {
      view.observer.disconnect();
      view.controls.dispose();
      view.renderer.dispose();
    }
    S.views[name] = makeView(name, el);
  }
}

function update(state) {
  if (S.drag) {
    S.waiting = state;
    return;
  }
  mount();
  S.mode = state.mode || 'free';
  S.style = state.style || 'paper';
  S.frame = state.frame || null;
  S.selected = state.selected || null;
  const text = JSON.stringify(state.model || null);
  if (text !== S.modelText) {
    S.modelText = text;
    S.model = state.model ? JSON.parse(text) : null;
    rebuild();
    const walls = S.model ? S.model.walls || {} : {};
    const fitKey = S.model ? `${S.model.setup}|${S.model.units}|${walls.width}|${walls.length}` : '';
    if (fitKey !== S.fitKey) {
      S.fitKey = fitKey;
      if (S.views.main && S.views.main.width) fitMain(S.views.main);
      if (S.views.plan) fitPlan(S.views.plan);
      fitFlats(false);
    }
  } else {
    // the frame's shape draws a camera's angle of view
    for (const name of Object.keys((S.model && S.model.cameras) || {})) buildItem('cameras', name);
    showSelection();
  }
  applyMode();
  invalidate();
}

window.macawRoom = { update };
watchTheme();
if (window.macawRoomPending) {
  const pending = window.macawRoomPending;
  delete window.macawRoomPending;
  update(pending);
}
