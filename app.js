// Figurine Labeler — browser-only segmentation labeling. Everything (incl. the models) runs locally in the browser.
// Workflow per photo: Propose (whole photo, or inside a box you drag) -> per proposal: Accept / Decline / Improve
// (outline, paint & erase), or Draw myself -> pick the class -> saved immediately.
// Labels: <folder>/labels/<photo>.png (object map, 8-bit gray: 0 = background, k = object k, same size as photo),
//         <folder>/labels/<photo>.json ({"objects": [{"id": k, "class": "..."}]}), <folder>/labels/classes.json.
import { AutoProcessor, RawImage, Sam2Model, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";
import { unzlibSync, zipSync, zlibSync } from "https://cdn.jsdelivr.net/npm/fflate@0.8.2/esm/browser.js";

// Models (ONNX versions of Apache-2.0 models, no login needed) are pinned to fixed revisions on Hugging Face,
// so the tool keeps behaving the same even if the model repositories change.
const SAM_ID = "onnx-community/sam2.1-hiera-tiny-ONNX", SAM_REV = "814a066640debee5a91e70aa401fb8e17e030503";
const DET_ID = "onnx-community/owlv2-base-patch16-ensemble-ONNX", DET_REV = "180d6ef7599bd69bb48db5c72bd49ad03ddfab80";
const DET_QUERY = "toy figurine", DET_THRESHOLD = 0.25;
const WORK_MAX = 2048; // photos are edited at most at this size (long side); labels are saved at full size
const BRUSH = 12; // brush radius in screen pixels (zoom in for finer work)
const PALETTE = ["#22c55e", "#ef4444", "#a855f7", "#f59e0b", "#3b82f6", "#ec4899", "#14b8a6", "#84cc16", "#f97316"];
const WORK_COLOR = [0, 220, 255];

const $ = (id) => document.getElementById(id);
const canvas = $("canvas"), ctx = canvas.getContext("2d");
const stem = (name) => name.replace(/\.[^.]+$/, "");
const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));

// ---------------------------------------------------------------- PNG (8-bit grayscale) codec
const CRC = new Uint32Array(256).map((_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c; });
const crc32 = (b) => { let c = ~0; for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 255] ^ (c >>> 8); return ~c >>> 0; };
function concat(parts) { const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0)); let o = 0; for (const p of parts) { out.set(p, o); o += p.length; } return out; }
function chunk(type, data) {
  const out = new Uint8Array(12 + data.length), dv = new DataView(out.buffer);
  dv.setUint32(0, data.length); out.set(new TextEncoder().encode(type), 4); out.set(data, 8);
  dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}
function encodePng(px, w, h) {
  const raw = new Uint8Array((w + 1) * h);
  for (let y = 0; y < h; y++) raw.set(px.subarray(y * w, (y + 1) * w), y * (w + 1) + 1);
  const ihdr = new Uint8Array(13), dv = new DataView(ihdr.buffer);
  dv.setUint32(0, w); dv.setUint32(4, h); ihdr.set([8, 0, 0, 0, 0], 8); // 8-bit, grayscale, no interlace
  return concat([new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", zlibSync(raw)), chunk("IEND", new Uint8Array())]);
}
function decodePng(b) { // grayscale 1- or 8-bit, non-interlaced (what this tool and PIL mode "L"/"1" write)
  const dv = new DataView(b.buffer, b.byteOffset); const idat = [];
  let w, h, depth, type;
  for (let i = 8; i < b.length;) {
    const len = dv.getUint32(i), t = String.fromCharCode(...b.subarray(i + 4, i + 8));
    if (t === "IHDR") { w = dv.getUint32(i + 8); h = dv.getUint32(i + 12); depth = b[i + 16]; type = b[i + 17]; if (b[i + 20]) throw Error("interlaced PNG"); }
    if (t === "IDAT") idat.push(b.subarray(i + 8, i + 8 + len));
    i += 12 + len;
  }
  if (type !== 0 || (depth !== 8 && depth !== 1)) throw Error(`label PNG must be 1- or 8-bit grayscale (got type ${type}, depth ${depth})`);
  const data = unzlibSync(concat(idat)), rb = Math.ceil((w * depth) / 8), rows = new Uint8Array(rb * h);
  for (let y = 0; y < h; y++) {
    const f = data[y * (rb + 1)], src = data.subarray(y * (rb + 1) + 1, (y + 1) * (rb + 1)), o = y * rb;
    for (let x = 0; x < rb; x++) {
      const a = x ? rows[o + x - 1] : 0, up = y ? rows[o - rb + x] : 0, c = x && y ? rows[o - rb + x - 1] : 0;
      const p = a + up - c, pa = Math.abs(p - a), pb = Math.abs(p - up), pc = Math.abs(p - c);
      rows[o + x] = src[x] + [0, a, up, (a + up) >> 1, pa <= pb && pa <= pc ? a : pb <= pc ? up : c][f];
    }
  }
  if (depth === 8) return { w, h, px: rows };
  const px = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) px[y * w + x] = (rows[y * rb + (x >> 3)] >> (7 - (x & 7))) & 1;
  return { w, h, px };
}

// ---------------------------------------------------------------- EXIF orientation (JPEG APP1 / PNG eXIf)
async function exifOrientation(file) {
  try {
    const b = new Uint8Array(await file.slice(0, 512 * 1024).arrayBuffer()), dv = new DataView(b.buffer);
    let tiff = -1;
    if (b[0] === 0xff && b[1] === 0xd8) {
      for (let i = 2; i + 10 < b.length && b[i] === 0xff;) {
        if (b[i + 1] === 0xe1 && String.fromCharCode(...b.subarray(i + 4, i + 8)) === "Exif") { tiff = i + 10; break; }
        if (b[i + 1] === 0xda) break;
        i += 2 + dv.getUint16(i + 2);
      }
    } else if (b[0] === 0x89) {
      for (let i = 8; i + 8 < b.length;) {
        const len = dv.getUint32(i), t = String.fromCharCode(...b.subarray(i + 4, i + 8));
        if (t === "eXIf") { tiff = i + 8; break; }
        i += 12 + len;
      }
    }
    if (tiff < 0) return 1;
    const le = b[tiff] === 0x49, ifd = tiff + dv.getUint32(tiff + 4, le);
    for (let k = 0, n = dv.getUint16(ifd, le); k < n; k++) {
      const e = ifd + 2 + 12 * k;
      if (dv.getUint16(e, le) === 0x0112) return dv.getUint16(e + 8, le);
    }
  } catch { /* unreadable metadata: treat as upright */ }
  return 1;
}

// ---------------------------------------------------------------- opening photos & storing labels
// Three ways in, one interface. source = {kind: "folder" | "photo", name, list, io}; io = getPhoto(p),
// replacePhoto(p, blob), read/write/remove(fname) inside labels/, labelled() -> Set of stems, autosave (bool).
let source = null, io = null, photos = [], classes = [], done = new Set(); // no predefined classes: users name their own
async function openFolderFS() { // Chrome/Edge: labels are written straight into <folder>/labels/
  const dir = await window.showDirectoryPicker({ mode: "readwrite" });
  const labels = await dir.getDirectoryHandle("labels", { create: true });
  const list = [];
  for await (const [name, h] of dir.entries()) if (h.kind === "file" && /\.(jpe?g|png)$/i.test(name)) list.push({ name, handle: h });
  return { kind: "folder", name: dir.name, list, io: {
    autosave: true,
    getPhoto: (p) => p.handle.getFile(),
    async replacePhoto(p, blob) { const w = await p.handle.createWritable(); await w.write(blob); await w.close(); },
    async read(f) { try { return new Uint8Array(await (await (await labels.getFileHandle(f)).getFile()).arrayBuffer()); } catch (e) { if (e.name === "NotFoundError") return null; throw e; } },
    async write(f, data) { const w = await (await labels.getFileHandle(f, { create: true })).createWritable(); await w.write(data); await w.close(); },
    async remove(f) { await labels.removeEntry(f).catch((e) => { if (e.name !== "NotFoundError") throw e; }); },
    async labelled() { const s = new Set(); for await (const [n] of labels.entries()) if (n.endsWith(".json") && n !== "classes.json") s.add(stem(n)); return s; },
  } };
}
function openFolderFallback(files) { // other browsers: read the folder, keep labels in memory, download them as a zip
  const labelFiles = new Map(), list = [], name = files[0]?.webkitRelativePath.split("/")[0] || "photos";
  for (const f of files) {
    const parts = f.webkitRelativePath.split("/");
    if (parts.length === 2 && /\.(jpe?g|png)$/i.test(f.name)) list.push({ name: f.name, file: f });
    if (parts.length === 3 && parts[1] === "labels") labelFiles.set(f.name, f);
  }
  return { kind: "folder", name, list, io: memoryIo(labelFiles, `${name}-labels.zip`) };
}
function openSinglePhoto(file) { // one photo: labels kept in memory, downloaded as a zip
  return { kind: "photo", name: file.name, list: [{ name: file.name, file }], io: memoryIo(new Map(), `${stem(file.name)}-labels.zip`) };
}
function memoryIo(labelFiles, zipName) { // labels live in memory until downloaded (zip with the same labels/ layout)
  const pending = new Map(), names = () => [...new Set([...labelFiles.keys(), ...pending.keys()])].filter((n) => pending.get(n) !== null);
  const changed = () => { mio.unsaved = true; renderList(); };
  const mio = {
    autosave: false, unsaved: false,
    getPhoto: async (p) => p.file,
    async replacePhoto(p, blob) { // continue with the upright copy, and give it to the user to keep with the labels
      p.file = new File([blob], p.name, { type: blob.type }); download(blob, p.name);
      alert(`Downloaded an upright copy of ${p.name}. Keep it instead of the original: the labels belong to the upright copy.`);
    },
    async read(f) { if (pending.has(f)) return pending.get(f); const x = labelFiles.get(f); return x ? new Uint8Array(await x.arrayBuffer()) : null; },
    async write(f, data) { pending.set(f, typeof data === "string" ? new TextEncoder().encode(data) : data); changed(); },
    async remove(f) { pending.set(f, null); changed(); },
    async labelled() { return new Set(names().filter((n) => n.endsWith(".json") && n !== "classes.json").map(stem)); },
    async zip() {
      if (!names().length) return alert("There are no labels to download yet.");
      const out = {}; for (const n of names()) out[`labels/${n}`] = await mio.read(n);
      download(new Blob([zipSync(out)]), zipName); mio.unsaved = false; renderList();
    },
  };
  return mio;
}
function download(blob, name) { const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000); }

async function afterOpen(opened) {
  source = opened; io = opened.io; cur = null; work = null; mode = "idle"; showToken++; refresh();
  const list = opened.list.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  // labels are named after the photo without extension: names must differ by more than extension or upper/lower case,
  // and "classes" is taken by labels/classes.json
  const seen = new Set(["classes"]), skipped = [];
  photos = list.filter((p) => { const k = stem(p.name).toLowerCase(); return seen.has(k) ? (skipped.push(p.name), false) : seen.add(k); });
  if (skipped.length) alert(`Skipped: ${skipped.join(", ")}. A photo can't be called "classes", and photo names must differ by more than the extension or upper/lower case. Rename them to label them.`);
  const c = await io.read("classes.json").catch(() => null), labelled = await io.labelled();
  if (source !== opened) return; // the user went back to the start screen meanwhile
  try { classes = c ? JSON.parse(new TextDecoder().decode(c)) : []; if (!Array.isArray(classes) || classes.some((x) => typeof x !== "string")) throw Error(); }
  catch { classes = []; alert("labels/classes.json can't be read; the class list will be rebuilt from the labels."); }
  done = labelled;
  if (!photos.length) { alert("No JPG or PNG photos found in this folder."); return goHome(); }
  $("welcome").style.display = "none";
  renderList(); show(0);
}
function goHome() { // back to the start screen
  source = io = cur = work = null; photos = []; done = new Set(); mode = "idle"; showToken++;
  $("welcome").style.display = ""; banner(""); status(""); renderList(); refresh();
}

// ---------------------------------------------------------------- models (downloaded once, then cached by the browser)
// Both models start loading silently when the page opens (see the end of this file). Their progress is only shown
// if the user clicks Propose before a model it needs is ready.
let device = "wasm", f16 = false, samP = null, detP = null, samReady = false, detReady = false;
let loadMsg = "", waiting = false;
const progress = (name) => (p) => { // transformers.js reports the total download progress of each model
  if (p.status !== "progress_total") return;
  loadMsg = p.progress < 100 ? `Downloading the ${name} (first time only): ${Math.round(p.progress)}%` : `Starting the ${name}…`;
  if (waiting) status(loadMsg);
};
function sam() {
  return (samP ??= (async () => {
    const [model, processor] = await Promise.all([
      Sam2Model.from_pretrained(SAM_ID, {
        revision: SAM_REV,
        progress_callback: progress("segmentation model (SAM 2.1)"),
        device: { vision_encoder: device, prompt_encoder_mask_decoder: "wasm" },
        dtype: { vision_encoder: device === "webgpu" && f16 ? "fp16" : "fp32", prompt_encoder_mask_decoder: "fp32" },
      }),
      AutoProcessor.from_pretrained(SAM_ID, { revision: SAM_REV }),
    ]);
    return { model, processor };
  })());
}
const detector = () => (detP ??= pipeline("zero-shot-object-detection", DET_ID, { revision: DET_REV, progress_callback: progress("detection model (OWLv2)"), device, dtype: device === "webgpu" ? (f16 ? "fp16" : "fp32") : "q8" }));
const fail = (e) => { console.error(e); status("Something went wrong: " + e.message); };
const status = (t) => ($("status").textContent = t);

async function segment(c, box) { // SAM 2.1 with a box prompt -> best mask (work resolution)
  const [{ model, processor }, { inputs, emb }] = await Promise.all([sam(), c.emb]);
  const ip = processor.image_processor, sizes = [inputs.original_sizes, inputs.reshaped_input_sizes];
  const out = await model({ ...emb, input_boxes: ip.reshape_input_points([[box]], ...sizes, true) });
  const [m] = await processor.post_process_masks(out.pred_masks, ...sizes), scores = Array.from(out.iou_scores.data);
  const k = scores.indexOf(Math.max(...scores)), hw = c.W * c.H;
  return m.data.slice(k * hw, (k + 1) * hw);
}

// ---------------------------------------------------------------- state
// cur: the open photo {i, name, W, H, s, img, bmp, raw, emb, map (object ids), objects [{id, class}], locked}
// mode: idle (choose what to label) | review (a proposal) | improve (outline/paint/erase; also used to draw from scratch)
//       | classify (pick class) | selected (an accepted object)
let cur = null, mode = "idle", busy = false;
let box = null;        // optional focus box drawn in idle mode
let work = null;       // mask being reviewed / improved (Uint8Array, work resolution)
let editing = null;    // id of the accepted object being changed (null = a new proposal)
let queue = [], shown = 0, checkLabelled = true, undoStack = [], brush = "paint";
let scratch = false;       // true while drawing a new object by hand ("Draw myself")
let beforeImprove = null;  // proposal mask as it was before Improve (restored by Cancel)
let saveFailed = false;    // a save did not complete: warn before leaving the page

let showToken = 0;
async function show(i) { // open photo i; the current photo can't be edited while the next one loads
  if (i < 0 || i >= photos.length || busy) return;
  if ((work || saveFailed) && !confirm("Leave this photo? Changes that are not saved will be lost.")) return;
  busy = true; refresh(); status("Opening the photo…");
  try { await loadPhoto(i); } catch (e) { fail(e); } finally { busy = false; refresh(); }
}
async function loadPhoto(i) {
  const token = ++showToken, p = photos[i], file = await io.getPhoto(p);
  const orientation = await exifOrientation(file), bmp = await createImageBitmap(file);
  const s = Math.min(1, WORK_MAX / Math.max(bmp.width, bmp.height)), W = Math.round(bmp.width * s), H = Math.round(bmp.height * s);
  const img = new OffscreenCanvas(W, H);
  img.getContext("2d").drawImage(bmp, 0, 0, W, H);
  // full: authoritative full-resolution object map; map: working copy; changed: ids whose pixels were edited
  const c = { i, name: p.name, W, H, s, img, bmp, map: new Uint8Array(W * H), full: new Uint8Array(bmp.width * bmp.height), changed: new Set(), objects: [], locked: orientation !== 1 };
  let warning = "";
  if (!c.locked) {
    c.raw = RawImage.fromCanvas(img);
    c.emb = sam().then(async ({ model, processor }) => { const inputs = await processor(c.raw); return { inputs, emb: await model.get_image_embeddings(inputs) }; });
    c.emb.catch(fail);
    warning = await loadLabels(c);
  }
  if (token !== showToken) return; // the user already moved on to another photo
  cur = c; mode = "idle"; box = work = editing = null; queue = []; hoverId = 0;
  fitView(); status("");
  if (orientation !== 1) {
    banner(`This photo is stored sideways with a “rotate me” note, which other software reads inconsistently. To label it, make it upright (${io.autosave ? "this replaces the photo file with an upright copy" : "you'll download an upright copy to use instead"}).`,
      ["Make it upright", () => makeUpright(c)]);
  } else banner(warning);
  renderList(); refresh();
}

async function loadLabels(c) { // -> warning text ("" if fine); unreadable or incomplete labels lock the photo
  let w, h, px, objects;
  try {
    const j = await io.read(stem(c.name) + ".json"), png = await io.read(stem(c.name) + ".png");
    if (!j && !png) return "";
    if (!j || !png) throw Error(`${stem(c.name)}.${j ? "png" : "json"} is missing in labels/`);
    ({ w, h, px } = decodePng(png));
    objects = JSON.parse(new TextDecoder().decode(j)).objects;
    if (!Array.isArray(objects) || objects.some((o) => !Number.isInteger(o.id) || o.id < 1 || o.id > 255 || typeof o.class !== "string")
      || new Set(objects.map((o) => o.id)).size !== objects.length) throw Error("unexpected format");
  } catch (e) {
    c.locked = true;
    return `The saved labels of this photo can't be read (${e.message}). Editing is off to protect them.`;
  }
  if (w !== c.bmp.width || h !== c.bmp.height) {
    c.locked = true;
    return `The saved labels (${w}×${h}) don't match this photo (${c.bmp.width}×${c.bmp.height}). Was the photo changed? Editing is off to protect the labels.`;
  }
  c.full = px;
  for (let y = 0; y < c.H; y++) { const sy = Math.min(h - 1, Math.floor((y + 0.5) / c.s)); for (let x = 0; x < c.W; x++) c.map[y * c.W + x] = px[sy * w + Math.min(w - 1, Math.floor((x + 0.5) / c.s))]; }
  const present = new Uint8Array(256), known = new Set(objects.map((o) => o.id));
  for (let k = 0; k < px.length; k++) present[px[k]] = 1;
  const orphans = present.reduce((ids, v, id) => (v && id && !known.has(id) ? [...ids, id] : ids), []);
  c.objects = [...objects, ...orphans.map((id) => ({ id, class: "unknown" }))];
  for (const o of c.objects) if (o.class !== "unknown" && !classes.includes(o.class)) classes.push(o.class);
  return orphans.length ? `${orphans.length} object(s) in this photo had no class (probably an interrupted save) and are marked “unknown”. Click them to check and fix.` : "";
}

function banner(text, button) { // message bar with an optional [label, onclick] button; plain text only (file names can't inject markup)
  const b = $("banner"); b.textContent = text; b.style.display = text ? "block" : "none";
  if (button) { const el = document.createElement("button"); el.textContent = button[0]; el.onclick = button[1]; b.append(" ", el); }
}

async function makeUpright(c) {
  const full = new OffscreenCanvas(c.bmp.width, c.bmp.height);
  full.getContext("2d").drawImage(c.bmp, 0, 0); // createImageBitmap already applied the rotation
  await io.replacePhoto(photos[c.i], await full.convertToBlob({ type: /\.png$/i.test(c.name) ? "image/png" : "image/jpeg", quality: 0.95 }));
  show(c.i);
}

// ---------------------------------------------------------------- actions
async function propose() {
  const c = cur, masks = [], fromBox = !!box;
  if (!(box ? samReady : samReady && detReady)) { // still downloading: say so, then continue by itself
    waiting = true; status(loadMsg || "Preparing the models…");
    try { await Promise.all(box ? [sam()] : [sam(), detector()]); }
    catch (e) { throw new Error(`could not load the models (${e.message}). Check the internet connection and reload the page`); }
    finally { waiting = false; }
  }
  if (box) { status("Finding the object in the box…"); masks.push(cleanMask(await segment(c, box), c.W, c.H)); }
  else {
    status("Looking for objects…");
    const found = (await (await detector())(c.raw, [DET_QUERY], { threshold: DET_THRESHOLD })).sort((a, b) => b.score - a.score);
    let boxes = [];
    for (const { box: b } of found) { const bb = [b.xmin, b.ymin, b.xmax, b.ymax]; if (boxes.length < 10 && !boxes.some((o) => boxIoU(o, bb) > 0.5)) boxes.push(bb); }
    // drop "group" boxes drawn around several figurines at once (they contain >= 2 other boxes): SAM can't segment those
    boxes = boxes.filter((a) => boxes.filter((b) => b !== a && boxInside(a, b) >= 0.8).length < 2);
    for (const bb of boxes) masks.push(cleanMask(await segment(c, bb), c.W, c.H));
  }
  if (c !== cur) return;
  box = null;
  if (!masks.length) return status(cur.objects.length ? "No new objects found." : "Nothing found. Drag a box around the object, then Propose.");
  queue = masks; shown = 0; checkLabelled = !fromBox; status("");
  nextProposal();
}
function nextProposal() { // show the next queued proposal
  editing = null; scratch = false; undoStack = []; work = null;
  // detector proposals that are mostly labelled already are skipped, checked now rather than at Propose time,
  // so that e.g. a part of an object accepted a moment ago isn't proposed again (a box you drew is always shown)
  while (queue.length && !work) { const m = queue.shift(); if (!checkLabelled || !mostlyLabelled(m)) work = m; }
  if (work) { shown++; return setMode("review"); }
  if (!shown) status("No new objects found.");
  setMode("idle");
}
function mostlyLabelled(m) {
  let area = 0, taken = 0;
  for (let k = 0; k < m.length; k++) if (m[k]) { area++; if (cur.map[k]) taken++; }
  return !area || taken >= 0.5 * area;
}
function startDraw() { // a new object drawn by hand, starting from an empty mask
  editing = null; scratch = true; work = new Uint8Array(cur.W * cur.H); undoStack = []; brush = "outline";
  if (box) { zoomToRect(...box); box = null; }
  setMode("improve");
}
function startImprove() {
  if (editing !== null) work = cur.map.map((v) => (v === editing ? 1 : 0));
  beforeImprove = work.slice(); undoStack = []; brush = "paint"; zoomTo(work); setMode("improve");
}
function cancelImprove() { // leave the drawing screen without keeping any of its changes
  undoStack = [];
  if (scratch) { scratch = false; work = null; setMode("idle"); }
  else if (editing !== null) { work = null; setMode("selected"); }
  else { work = beforeImprove; setMode("review"); }
}
async function finishImprove() {
  if (scratch) {
    if (work.some((v) => v)) return setMode("classify"); // drawn by hand: no need to ask "is this right?"
    scratch = false; work = null; status("Nothing was drawn."); return setMode("idle");
  }
  if (editing === null) return setMode("review");
  commit(editing); await save(); work = null;
  if (cur.objects.some((o) => o.id === editing)) setMode("selected"); else { editing = null; setMode("idle"); }
}
async function assign(cls) {
  const c = cur;
  if (editing !== null) c.objects.find((o) => o.id === editing).class = cls;
  else { const id = Math.max(0, ...c.objects.map((o) => o.id)) + 1; if (id > 255) return alert("Too many objects in one photo."); c.objects.push({ id, class: cls }); commit(id); }
  await save();
  if (editing !== null) setMode("selected"); else nextProposal();
}
async function newClass() {
  const name = (window.prompt("Name of the new class (for example: smurf):") ?? "").trim();
  if (!name) return;
  if (!classes.includes(name)) classes.push(name);
  await assign(name);
}
async function removeObject() {
  const c = cur;
  for (let k = 0; k < c.map.length; k++) if (c.map[k] === editing) c.map[k] = 0;
  c.changed.add(editing); await save(); editing = null; setMode("idle");
}
async function nothingHere() { cur.objects = []; cur.map.fill(0); cur.full.fill(0); await save(true); refresh(); }
function commit(id) { // write the work mask as object "id" into the working map
  const c = cur; c.changed.add(id);
  for (let k = 0; k < c.map.length; k++) { if (work[k]) c.map[k] = id; else if (c.map[k] === id) c.map[k] = 0; }
}

async function save(nothing = false) {
  const c = cur, name = stem(c.name), OW = c.bmp.width, OH = c.bmp.height;
  // only objects edited here are re-drawn into the full-resolution map; all other pixels stay exactly as loaded
  for (const id of c.changed) {
    for (let k = 0; k < c.full.length; k++) if (c.full[k] === id) c.full[k] = 0;
    const m = c.s === 1 ? c.map.map((v) => +(v === id)) : upsample(c.map, id, c.W, c.H, OW, OH);
    for (let k = 0; k < m.length; k++) if (m[k]) c.full[k] = id;
  }
  c.changed.clear();
  const present = new Uint8Array(256);
  for (let k = 0; k < c.full.length; k++) present[c.full[k]] = 1;
  c.objects = c.objects.filter((o) => present[o.id]); // object ids never change, so PNG and JSON can't get mixed up
  const empty = !c.objects.length && !nothing; // no objects left: remove the label files
  saveFailed = true; // stays set (and warns on leaving the page) if any write or delete below fails
  try {
    if (empty) { await io.remove(name + ".png"); await io.remove(name + ".json"); }
    else {
      await io.write(name + ".png", encodePng(c.full, OW, OH));
      await io.write(name + ".json", JSON.stringify({ objects: c.objects }, null, 1));
      await io.write("classes.json", JSON.stringify(classes, null, 1));
    }
  } catch (e) { throw Error(`the labels could not be saved (${e.message}). Please try again`); }
  saveFailed = false;
  if (empty) done.delete(name); else done.add(name);
  status(io.autosave ? "Saved ✓" : "Done ✓ (download the labels before closing the page)"); renderList();
}
function upsample(map, id, W, H, OW, OH) { // object "id" of the working map -> smooth full-resolution 0/1 mask
  const small = new OffscreenCanvas(W, H), big = new OffscreenCanvas(OW, OH);
  const sc = small.getContext("2d"), bc = big.getContext("2d", { willReadFrequently: true });
  const im = sc.createImageData(W, H);
  for (let k = 0; k < map.length; k++) if (map[k] === id) im.data[4 * k + 3] = 255;
  sc.putImageData(im, 0, 0); bc.imageSmoothingQuality = "high"; bc.drawImage(small, 0, 0, OW, OH);
  const a = bc.getImageData(0, 0, OW, OH).data, out = new Uint8Array(OW * OH);
  for (let k = 0; k < out.length; k++) out[k] = a[4 * k + 3] > 127;
  return out;
}
const overlap = (a, b) => Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
const boxArea = (a) => (a[2] - a[0]) * (a[3] - a[1]);
const boxIoU = (a, b) => overlap(a, b) / (boxArea(a) + boxArea(b) - overlap(a, b));
const boxInside = (a, b) => overlap(a, b) / boxArea(b); // fraction of box b that lies inside box a

// Clean-up of model proposals (never applied to the user's own edits): SAM's coarse output leaves specks and
// pinholes. Remove pieces smaller than 10% of the largest piece, and fill only pinholes (< 0.05% of the object):
// real gaps (e.g. between legs, or between an arm and a held object) measured 0.18-0.6% of the object in test
// photos and must stay open.
const MIN_PIECE = 0.1, MAX_HOLE = 0.0005;
function cleanMask(m, W, H) {
  let x0 = W, y0 = H, x1 = -1, y1 = -1; // work inside the mask's bounding box (+1 px margin)
  for (let k = 0; k < m.length; k++) if (m[k]) { const x = k % W, y = (k / W) | 0; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  if (x1 < 0) return m;
  x0 = Math.max(0, x0 - 1); y0 = Math.max(0, y0 - 1); x1 = Math.min(W - 1, x1 + 1); y1 = Math.min(H - 1, y1 + 1);
  const w = x1 - x0 + 1, h = y1 - y0 + 1, a = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) a[y * w + x] = m[(y + y0) * W + x + x0] ? 1 : 0;
  const fg = components(a, w, h, 1), largest = Math.max(...fg.sizes.slice(1));
  let area = 0;
  for (let k = 0; k < a.length; k++) if (a[k]) { if (fg.sizes[fg.lab[k]] < MIN_PIECE * largest) a[k] = 0; else area++; }
  const bg = components(a, w, h, 0); // background pieces not touching the box border are holes
  for (let k = 0; k < a.length; k++) if (!a[k] && !bg.edge[bg.lab[k]] && bg.sizes[bg.lab[k]] < MAX_HOLE * area) a[k] = 1;
  const out = m.slice();
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[(y + y0) * W + x + x0] = a[y * w + x];
  return out;
}
function components(a, w, h, v) { // 4-connected regions of pixels equal to v -> per-pixel label, size and border contact per label
  const lab = new Int32Array(a.length), sizes = [0], edge = [false], stack = new Int32Array(a.length);
  for (let s = 0; s < a.length; s++) {
    if (a[s] !== v || lab[s]) continue;
    const id = sizes.length; let n = 0, top = 0, touches = false;
    lab[s] = id; stack[top++] = s;
    while (top) {
      const k = stack[--top], x = k % w, y = (k / w) | 0; n++;
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) touches = true;
      for (const j of [x > 0 ? k - 1 : -1, x < w - 1 ? k + 1 : -1, y > 0 ? k - w : -1, y < h - 1 ? k + w : -1])
        if (j >= 0 && a[j] === v && !lab[j]) { lab[j] = id; stack[top++] = j; }
    }
    sizes.push(n); edge.push(touches);
  }
  return { lab, sizes, edge };
}

// ---------------------------------------------------------------- the button bar (the whole UI lives here)
function setMode(m) { mode = m; refresh(); }
async function run(fn) { if (busy) return; busy = true; refresh(); try { await fn(); } catch (e) { fail(e); } finally { busy = false; refresh(); } }
function refresh() {
  const c = cur, bar = $("actions");
  bar.innerHTML = "";
  $("title").textContent = c ? c.name : "";
  const add = (label, fn, { main = false, on = false } = {}) => {
    const b = document.createElement("button");
    b.textContent = label; b.className = (main ? "main" : "") + (on ? " on" : "");
    b.disabled = busy || !c || c.locked; b.onclick = () => run(fn);
    bar.append(b);
  };
  let hint = "";
  if (!c) hint = "";
  else if (mode === "idle") {
    add("✨ Propose", propose, { main: true }); add("✏️ Draw myself", startDraw);
    if (!c.objects.length && !done.has(stem(c.name))) add("Nothing to label here", nothingHere);
    hint = box ? "Propose and Draw myself will focus on the box. Drag again to change it, click outside it to remove it."
      : (c.objects.length ? "Click a coloured object to improve it, change its class or delete it. Or propose / draw more objects."
        : "Click Propose to find objects automatically, or draw one yourself.") + " Tip: first drag a box around an object to focus on it.";
  } else if (mode === "review") {
    add("✓ Accept", () => setMode("classify"), { main: true }); add("✗ Decline", nextProposal); add("✎ Improve", startImprove);
    hint = (shown > 1 || queue.length ? `Proposal ${shown}. ` : "") + "Is the highlighted area right?";
  } else if (mode === "improve") {
    add("✏️ Outline", () => (brush = "outline"), { on: brush === "outline" });
    add("🖌 Paint", () => (brush = "paint"), { on: brush === "paint" }); add("🧽 Erase", () => (brush = "erase"), { on: brush === "erase" });
    add("↶ Undo", undo); add("✗ Cancel", cancelImprove); add("✓ Done", finishImprove, { main: true });
    const how = { outline: "Hold the mouse button and draw around an area to add it", paint: "Paint over pixels that belong to the object", erase: "Erase pixels that don't belong to the object" }[brush];
    hint = `${how}. Pinch or ⌘/Ctrl + scroll to zoom, scroll to move around.`;
  } else if (mode === "classify") {
    classes.forEach((cls) => add(cls, () => assign(cls), { main: true }));
    add("+ New class…", newClass, { main: !classes.length }); add("Cancel", () => setMode(scratch ? "improve" : editing === null ? "review" : "selected"));
    hint = classes.length ? "What is it?" : "What is it? Create its class with “+ New class…” (you name it once, then it is one click).";
  } else if (mode === "selected") {
    add("✎ Improve", startImprove); add("🏷 Change class", () => setMode("classify")); add("🗑 Delete", removeObject); add("Back", () => { editing = null; setMode("idle"); });
    hint = `Selected: ${c.objects.find((o) => o.id === editing)?.class}.`;
  }
  $("hint").textContent = hint;
  rebuildLayers(); render();
}
function undo() { if (undoStack.length) { work = undoStack.pop(); rebuildLayers(); render(); } }

function renderList() { // the sidebar: what is open, where its labels go, and the photo list
  const s = source, el = $("list"); el.innerHTML = "";
  $("where").hidden = $("other").hidden = !s; $("zip").hidden = !s || io.autosave;
  $("zipState").textContent = s && !io.autosave && io.unsaved ? "● not downloaded yet" : "";
  $("progress").textContent = s?.kind === "folder" ? `${done.size} of ${photos.length} photos done` : "";
  if (!s) return;
  $("whereName").textContent = `${s.kind === "folder" ? "📁" : "🖼"} ${s.name}`;
  $("whereNote").textContent = io.autosave ? `Labels are saved automatically to ${s.name}/labels/`
    : s.kind === "folder" ? "This browser can't save into the folder: download the labels before closing. To continue later, replace the labels folder next to your photos with the one in the zip."
    : "Download the labels when you're done.";
  if (s.kind === "photo") return;
  photos.forEach((p, i) => {
    const d = document.createElement("div");
    d.textContent = `${done.has(stem(p.name)) ? "✓" : "○"}  ${p.name}`;
    if (cur?.i === i) d.className = "current";
    d.onclick = () => show(i);
    el.append(d);
  });
}

// ---------------------------------------------------------------- drawing
const view = { scale: 1, x: 0, y: 0 }, layers = { objects: null, work: null };
const classColor = (cls) => PALETTE[Math.max(0, classes.indexOf(cls)) % PALETTE.length];
function fitView() {
  const r = canvas.getBoundingClientRect();
  view.scale = Math.min(r.width / cur.W, r.height / cur.H) * 0.98;
  view.x = (r.width - cur.W * view.scale) / 2; view.y = (r.height - cur.H * view.scale) / 2;
}
function zoomTo(m) {
  const c = cur; let x0 = c.W, y0 = c.H, x1 = 0, y1 = 0;
  for (let k = 0; k < m.length; k++) if (m[k]) { const x = k % c.W, y = (k / c.W) | 0; x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  if (x1 >= x0) zoomToRect(x0, y0, x1, y1);
}
function zoomToRect(x0, y0, x1, y1) { // fit the rectangle (with some margin) into the view
  const r = canvas.getBoundingClientRect(), bw = (x1 - x0) * 1.8 + 20, bh = (y1 - y0) * 1.8 + 20;
  view.scale = Math.min(r.width / bw, r.height / bh);
  view.x = r.width / 2 - ((x0 + x1) / 2) * view.scale; view.y = r.height / 2 - ((y0 + y1) / 2) * view.scale;
}
function layer(test, color) { // translucent fill + opaque outline
  const c = cur, W = c.W, n = c.map.length, im = new ImageData(W, c.H), d = im.data;
  for (let k = 0; k < n; k++) {
    if (!test(k)) continue;
    const [r, g, b, a] = color(k);
    const edge = k % W === 0 || k % W === W - 1 || k < W || k >= n - W || !test(k - 1) || !test(k + 1) || !test(k - W) || !test(k + W);
    d[4 * k] = r; d[4 * k + 1] = g; d[4 * k + 2] = b; d[4 * k + 3] = edge ? 255 : a;
  }
  const out = new OffscreenCanvas(W, c.H); out.getContext("2d").putImageData(im, 0, 0); return out;
}
function rebuildLayers() {
  const c = cur; if (!c) return;
  const hidden = mode === "improve" ? editing : null; // the object being improved is drawn as the work mask instead
  const col = Object.fromEntries(c.objects.map((o) => [o.id, [...rgb(classColor(o.class)), o.id === editing ? 190 : mode === "idle" && o.id === hoverId ? 170 : 100]]));
  layers.objects = layer((k) => c.map[k] > 0 && c.map[k] !== hidden, (k) => col[c.map[k]] ?? [255, 255, 255, 100]);
  layers.work = work ? layer((k) => work[k] > 0, () => [...WORK_COLOR, 120]) : null;
}
function render() {
  const r = canvas.getBoundingClientRect(), dpr = devicePixelRatio;
  if (canvas.width !== Math.round(r.width * dpr) || canvas.height !== Math.round(r.height * dpr)) { canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr); }
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!cur) return;
  ctx.setTransform(dpr * view.scale, 0, 0, dpr * view.scale, dpr * view.x, dpr * view.y);
  ctx.imageSmoothingEnabled = view.scale < 2; ctx.drawImage(cur.img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  if (layers.objects) ctx.drawImage(layers.objects, 0, 0);
  if (layers.work) ctx.drawImage(layers.work, 0, 0);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const b = drag?.box ?? (mode === "idle" ? box : null);
  if (b) { ctx.setLineDash([6, 4]); ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 2; ctx.strokeRect(b[0] * view.scale + view.x, b[1] * view.scale + view.y, (b[2] - b[0]) * view.scale, (b[3] - b[1]) * view.scale); ctx.setLineDash([]); }
  if (drag?.lasso) { // the outline being drawn, closed back to its start
    ctx.strokeStyle = "#facc15"; ctx.lineWidth = 2; ctx.beginPath();
    drag.lasso.forEach(([x, y], k) => ctx[k ? "lineTo" : "moveTo"](x * view.scale + view.x, y * view.scale + view.y));
    ctx.closePath(); ctx.stroke();
  }
  const outlining = mode === "improve" && brush === "outline";
  canvas.style.cursor = outlining ? "crosshair" : mode === "idle" && hoverId ? "pointer" : "default";
  if (mode === "improve" && !outlining && mouse) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(mouse[0], mouse[1], BRUSH, 0, 2 * Math.PI); ctx.stroke(); }
}

// ---------------------------------------------------------------- mouse & keys
let drag = null, mouse = null, spaceDown = false, hoverId = 0; // hoverId: saved object under the mouse (highlighted)
const idAt = ([x, y]) => { x = Math.floor(x); y = Math.floor(y); return cur && x >= 0 && y >= 0 && x < cur.W && y < cur.H ? cur.map[y * cur.W + x] : 0; };
const toImage = (e) => { const r = canvas.getBoundingClientRect(); return [(e.clientX - r.left - view.x) / view.scale, (e.clientY - r.top - view.y) / view.scale]; };
function stamp(x0, y0, x1, y1) { // paint/erase a brush stroke segment into the work mask
  const c = cur, r = BRUSH / view.scale, v = brush === "paint" ? 1 : 0, steps = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / (r / 2)));
  for (let s = 0; s <= steps; s++) {
    const x = x0 + ((x1 - x0) * s) / steps, y = y0 + ((y1 - y0) * s) / steps;
    for (let yy = Math.max(0, Math.floor(y - r)); yy <= Math.min(c.H - 1, y + r); yy++)
      for (let xx = Math.max(0, Math.floor(x - r)); xx <= Math.min(c.W - 1, x + r); xx++)
        if ((xx - x) ** 2 + (yy - y) ** 2 <= r * r) work[yy * c.W + xx] = v;
  }
  layers.work = layer((k) => work[k] > 0, () => [...WORK_COLOR, 120]); render();
}
function fillOutline(pts) { // add the area enclosed by a hand-drawn outline to the work mask
  if (pts.length < 3) return;
  const c = cur, xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.max(0, Math.floor(Math.min(...xs))), y0 = Math.max(0, Math.floor(Math.min(...ys)));
  const x1 = Math.min(c.W - 1, Math.ceil(Math.max(...xs))), y1 = Math.min(c.H - 1, Math.ceil(Math.max(...ys)));
  if (x1 < x0 || y1 < y0) return;
  const w = x1 - x0 + 1, h = y1 - y0 + 1, g = new OffscreenCanvas(w, h).getContext("2d", { willReadFrequently: true });
  g.beginPath(); pts.forEach(([x, y], k) => g[k ? "lineTo" : "moveTo"](x - x0, y - y0)); g.closePath(); g.fill();
  const a = g.getImageData(0, 0, w, h).data;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (a[4 * (y * w + x) + 3] > 127) work[(y + y0) * c.W + x + x0] = 1;
}
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("mousedown", (e) => {
  if (!cur || busy) return;
  const [x, y] = toImage(e);
  if (spaceDown || e.button !== 0) { drag = { pan: true, x0: e.clientX - view.x, y0: e.clientY - view.y }; return; }
  if (cur.locked) return;
  if (mode === "improve") {
    undoStack.push(work.slice());
    if (brush === "outline") drag = { lasso: [[x, y]] }; else { drag = { paint: true, x, y }; stamp(x, y, x, y); }
  }
  else if (mode === "idle") drag = { x0: x, y0: y, sx: e.clientX, sy: e.clientY };
});
window.addEventListener("mousemove", (e) => {
  const r = canvas.getBoundingClientRect(); mouse = [e.clientX - r.left, e.clientY - r.top];
  if (drag?.pan) { view.x = e.clientX - drag.x0; view.y = e.clientY - drag.y0; }
  else if (drag?.paint) { const [x, y] = toImage(e); stamp(drag.x, drag.y, x, y); drag.x = x; drag.y = y; return; }
  else if (drag?.lasso) { const [x, y] = toImage(e), [px, py] = drag.lasso.at(-1); if (Math.hypot(x - px, y - py) * view.scale > 2) drag.lasso.push([x, y]); }
  else if (drag && Math.hypot(e.clientX - drag.sx, e.clientY - drag.sy) > 6) { const [x, y] = toImage(e); drag.box = [Math.min(x, drag.x0), Math.min(y, drag.y0), Math.max(x, drag.x0), Math.max(y, drag.y0)]; }
  const over = !drag && mode === "idle" && e.target === canvas ? idAt(toImage(e)) : 0; // hover: show what a click would select
  if (over !== hoverId) { hoverId = over; rebuildLayers(); }
  if (cur) render();
});
window.addEventListener("mouseup", () => {
  const d = drag; drag = null;
  if (!d || d.pan || !cur) return;
  if (d.paint) return refresh(); // enables Undo
  if (d.lasso) { fillOutline(d.lasso); return refresh(); }
  if (d.box) { box = d.box; return refresh(); }
  const id = idAt([d.x0, d.y0]); // a click: select an object, or remove the box
  if (id) { editing = id; box = null; setMode("selected"); } else { box = null; refresh(); }
});
canvas.addEventListener("wheel", (e) => {
  e.preventDefault(); if (!cur) return;
  if (e.ctrlKey || e.metaKey) { // pinch (trackpad) or ⌘/Ctrl + scroll: zoom around the cursor
    const r = canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top, f = Math.exp(-e.deltaY * 0.01);
    view.x = mx - (mx - view.x) * f; view.y = my - (my - view.y) * f; view.scale *= f;
  } else { view.x -= e.deltaX; view.y -= e.deltaY; }
  render();
}, { passive: false });
window.addEventListener("keydown", (e) => {
  if (e.key === " ") { spaceDown = true; if (e.target === document.body) e.preventDefault(); }
  if (!cur || busy) return;
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z" && mode === "improve") { e.preventDefault(); undo(); }
  if (mode === "idle" && e.key === "ArrowLeft") show(cur.i - 1);
  if (mode === "idle" && e.key === "ArrowRight") show(cur.i + 1);
});
window.addEventListener("keyup", (e) => { if (e.key === " ") spaceDown = false; });
window.addEventListener("resize", () => cur && render());

let opening = false; // one open at a time (reading a big folder takes a moment)
async function openWith(get) {
  if (opening) return;
  opening = true;
  try { await afterOpen(await get()); } catch (e) { if (e.name !== "AbortError") fail(e); } finally { opening = false; }
}
$("openFolder").onclick = () => ("showDirectoryPicker" in window ? openWith(openFolderFS) : $("folderInput").click());
$("openPhoto").onclick = () => $("photoInput").click();
$("folderInput").onchange = (e) => { const f = Array.from(e.target.files); e.target.value = ""; if (f.length) openWith(() => openFolderFallback(f)); };
$("photoInput").onchange = (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) openWith(() => openSinglePhoto(f)); };
if (!("showDirectoryPicker" in window)) $("folderNote").textContent = "This browser can't save into folders: you'll download the labels as a zip file. (Chrome or Edge save them automatically.)";
$("zip").onclick = () => io.zip();
$("other").onclick = () => {
  if (busy || ((work || saveFailed || io?.unsaved) && !confirm("Leave? Labels that are not saved or downloaded will be lost."))) return;
  goHome();
};
window.addEventListener("beforeunload", (e) => { if (work || busy || saveFailed || io?.unsaved) e.preventDefault(); }); // busy: e.g. a save in progress
$("fit").onclick = () => { if (cur) { fitView(); render(); } };

if (navigator.gpu) { const a = await navigator.gpu.requestAdapter().catch(() => null); if (a) { device = "webgpu"; f16 = a.features.has("shader-f16"); } }
sam().then(() => { samReady = true; return detector(); }).then( // segmentation first (enough for box proposals)
  () => { detReady = true; },
  (e) => { console.error(e); status(`Could not load the models (${e.message}). Check the internet connection and reload the page.`); });
