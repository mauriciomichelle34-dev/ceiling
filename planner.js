'use strict';
const $ = id => document.getElementById(id);
const photos = [];
let current = null, ready = false, busy = false, uploading = false, selected = null;
const geometryFields = ['center_x_px', 'center_y_px', 'dish_diameter_px', 'dish_diameter_mm'];
let backendOrigin = null;
let maxImageBytes = 100 * 1024 * 1024;
const exportIds = ['csvDownload', 'layoutDownload', 'jsonDownload'];
const cultureKey = 'mycelium-batch-culture-v1';
const cultureInputs = ['speciesName', 'cultureStartedAt'];
const defaultCulture = {speciesName: '', cultureStartedAt: ''};
let cultureStorageAvailable = true;

function cultureStartText(raw) {
  if (!raw) return {text: '未填写', valid: true};
  const parts = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})$/.exec(raw);
  const date = new Date(raw);
  if (!parts || !Number.isFinite(date.getTime()) || [date.getFullYear(), date.getMonth()+1, date.getDate(), date.getHours(), date.getMinutes()].some((v, i) => v !== Number(parts[i+1]))) {
    return {text: '请检查日期', valid: false};
  }
  return {text: `${parts[1]}-${parts[2]}-${parts[3]} ${parts[4]}:${parts[5]}`, valid: true};
}
function renderBatchCulture() {
  $('cultureSpeciesValue').textContent = $('speciesName').value.trim() || '菌丝种类未填写';
  const start = cultureStartText($('cultureStartedAt').value);
  $('cultureStartValue').textContent = start.text;
  $('cultureStartValue').classList.toggle('invalid', !start.valid);
  $('cultureStartedAt').setAttribute('aria-invalid', String(!start.valid));
  $('cultureHelper').textContent = !start.valid
    ? '请检查培养开始时间。'
    : cultureStorageAvailable
      ? '本批所有照片共用，资料已保存在当前浏览器。'
      : '当前浏览器无法保存资料，刷新后需要重新填写。';
}
function saveBatchCulture() {
  const values = Object.fromEntries(cultureInputs.map(id => [id, $(id).value]));
  try { localStorage.setItem(cultureKey, JSON.stringify(values)); cultureStorageAvailable = true; }
  catch { cultureStorageAvailable = false; }
  renderBatchCulture();
}
function initializeBatchCulture() {
  let stored = {};
  try {
    const data = JSON.parse(localStorage.getItem(cultureKey));
    if (data && typeof data === 'object' && !Array.isArray(data)) stored = data;
  } catch { /* Ignore an invalid saved record and use blank values. */ }
  for (const id of cultureInputs) {
    const value = typeof stored[id] === 'string' ? stored[id] : defaultCulture[id];
    $(id).value = id === 'speciesName' ? value.slice(0, 80) : value;
    $(id).addEventListener('input', saveBatchCulture);
    $(id).addEventListener('change', saveBatchCulture);
  }
  $('resetCulture').addEventListener('click', () => {
    for (const id of cultureInputs) $(id).value = defaultCulture[id];
    saveBatchCulture();
  });
  saveBatchCulture();
}
let areaKey = '', areaTimer = null, areaSequence = 0;
const RECOGNITION_MS = 800;
const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
let areaAnimation = null, areaAnimationFrame = null, displayedAreaKey = '';

function cancelAreaAnimation() {
  if (areaAnimationFrame !== null) cancelAnimationFrame(areaAnimationFrame);
  areaAnimationFrame = null; areaAnimation = null; displayedAreaKey = '';
}
function finishAreaAnimation(animation) {
  if (animation.sequence !== areaSequence || animation.photo !== current || animation.key !== areaKey) return;
  if (areaAnimationFrame !== null) cancelAnimationFrame(areaAnimationFrame);
  areaAnimationFrame = null; areaAnimation = null; displayedAreaKey = animation.key;
  showArea(animation.data); draw();
}
function presentArea(photo, key, sequence, c, data) {
  const animation = {photo, key, sequence, c, data, started: null, progress: 0};
  if (motionPreference.matches || document.hidden || !data.foreground_pixels || !photo.region) {
    finishAreaAnimation(animation); return;
  }
  areaAnimation = animation;
  $('areaPanel').dataset.phase = 'recognizing';
  $('areaInfo').textContent = '正在描绘识别出的菌丝区域…';
  const frame = timestamp => {
    if (areaAnimation !== animation || sequence !== areaSequence || photo !== current || key !== areaKey) return;
    if (animation.started === null) animation.started = timestamp;
    animation.progress = Math.min(1, (timestamp-animation.started)/RECOGNITION_MS);
    if (animation.progress >= 1) { finishAreaAnimation(animation); return; }
    draw(); areaAnimationFrame = requestAnimationFrame(frame);
  };
  areaAnimationFrame = requestAnimationFrame(frame);
}

function showArea(data) {
  $('areaPanel').dataset.phase = 'ready';
  $('areaPanel').setAttribute('aria-busy', 'false');
  $('myceliumArea').textContent = data.mycelium_area_mm2.toLocaleString('zh-CN', {minimumFractionDigits: 2, maximumFractionDigits: 2});
  $('dishCoverage').textContent = data.dish_coverage_percent.toFixed(2);
  $('areaInfo').textContent = `培养皿总面积 ${data.dish_area_mm2.toLocaleString('zh-CN', {maximumFractionDigits: 2})} mm² · 仅计入自动内边界与标定圆内的菌丝。`;
}
function updateArea() {
  let c;
  try {
    if (uploading) throw Error('正在识别新照片…');
    c = calibration();
  } catch (error) {
    clearTimeout(areaTimer); areaKey = ''; areaSequence++;
    cancelAreaAnimation();
    $('areaPanel').dataset.phase = 'waiting';
    $('areaPanel').setAttribute('aria-busy', String(uploading));
    $('myceliumArea').textContent = '—'; $('dishCoverage').textContent = '—';
    $('areaInfo').textContent = error.message;
    return;
  }
  const photo = current, key = JSON.stringify([photo.meta.id, c]);
  if (key === areaKey) return;
  clearTimeout(areaTimer); areaKey = key;
  const sequence = ++areaSequence;
  cancelAreaAnimation();
  $('areaPanel').dataset.phase = 'pending';
  $('areaPanel').setAttribute('aria-busy', 'true');
  $('myceliumArea').textContent = '—'; $('dishCoverage').textContent = '—';
  $('areaInfo').textContent = '正在测算菌丝体面积…';
  if (photo.area?.key === key) { presentArea(photo, key, sequence, c, photo.area.data); return; }
  areaTimer = setTimeout(async () => {
    try {
      const data = await request('/api/area', {prediction_id: photo.meta.id, calibration: c});
      if (sequence !== areaSequence || photo !== current) return;
      photo.area = {key, data}; presentArea(photo, key, sequence, c, data);
    } catch (error) {
      if (sequence !== areaSequence || photo !== current) return;
      areaKey = ''; cancelAreaAnimation(); draw();
      $('areaPanel').dataset.phase = 'error'; $('areaPanel').setAttribute('aria-busy', 'false');
      $('areaInfo').textContent = `面积测算失败：${error.message} 修改标定参数可重试。`;
    }
  }, 250);
}

function notice(message, error = false) {
  $('notice').textContent = message;
  $('notice').classList.toggle('error', error);
}
function numeric(id, label, positive = false) {
  const raw = $(id).value.trim(), value = Number(raw);
  if (raw === '' || !Number.isFinite(value) || value < 0 || (positive && value === 0)) {
    throw Error(`请填写有效的${label}。`);
  }
  return value;
}
function coordinateOrigin(meta = current?.meta) {
  if (!meta) throw Error('请先上传照片。');
  if (!backendOrigin) throw Error('后台固定原点尚未加载，请刷新网页。');
  if (meta.width !== backendOrigin.image_width || meta.height !== backendOrigin.image_height) {
    throw Error(`照片尺寸与后台原点不匹配，请使用 ${backendOrigin.image_width} × ${backendOrigin.image_height} px 的原相机画幅。`);
  }
  return {...backendOrigin};
}
function calibration() {
  if (!current) throw Error('拍照或上传后自动标定。');
  const c = current.meta.calibration;
  if (!c || c.dish_diameter_mm !== 90) throw Error('请重新上传照片，使用后台 90 mm 自动标定。');
  return {...c, coordinate_origin:coordinateOrigin()};
}
function capture() {
  if (!current) return;
  current.fields = Object.fromEntries(['diameter', 'gap'].map(id => [id, $(id).value]));
}
function geometryDirty() {
  if (!current?.plan) return false;
  try {
    const c = calibration(true), p = current.plan;
    return geometryFields.some(key => Math.abs(c[key] - p.calibration[key]) > 1e-8) ||
      JSON.stringify(c.coordinate_origin) !== JSON.stringify(p.calibration.coordinate_origin) ||
      numeric('diameter', '取样直径', true) !== p.diameter_mm || numeric('gap', '间隙') !== p.gap_mm;
  } catch { return true; }
}
function depthDirty() {
  if (!current?.plan) return false;
  try {
    return current.plan.points.some(p => {
      const raw = current.overrides[String(p.id)];
      return (raw !== undefined) !== p.depth_overridden || (raw !== undefined && Number(raw) !== p.z_mm);
    });
  } catch { return true; }
}
function planUrl(file) {
  return `/results/${current.meta.id}/plans/${current.plan.plan_id}/${file}`;
}
function link(id, href, enabled = true) {
  const el = $(id);
  el.classList.toggle('disabled', !enabled);
  el.setAttribute('aria-disabled', String(!enabled));
  if (enabled && href) el.href = href; else el.removeAttribute('href');
}
function controls() {
  const automaticLock = typeof automaticLocksPLC === 'function' && automaticLocksPLC();
  const has = Boolean(current), plan = current?.plan, geometric = geometryDirty(), depths = depthDirty();
  let validCalibration = false;
  try { calibration(); validCalibration = true; } catch {}
  let validOrigin = false;
  try { coordinateOrigin(); validOrigin = true; } catch {}
  $('files').disabled = busy || !ready || (typeof plcCameraControls === 'function' && plcCameraControls());
  $('photoSelect').disabled = busy;
  $('generate').disabled = !has || !ready || busy || !validCalibration || !validOrigin || automaticLock;
  $('generate').textContent = busy ? '正在处理…' : plan ? '重新生成取样排布' : '生成取样排布';
  $('saveDepths').disabled = !plan || !depths || geometric || busy || automaticLock;
  for (const id of ['diameter', 'gap']) $(id).disabled = busy || automaticLock;
  for (const el of $('pointRows').querySelectorAll('input,button')) el.disabled = busy || automaticLock;
  if (plan) {
    $('pointCount').textContent = plan.count;
    $('coverage').textContent = (plan.coverage_fraction * 100).toFixed(1);
    $('planStatus').textContent = busy ? '处理中' : geometric ? '需重新生成' : depths ? '深度待保存' : plan.count ? '规划已保存' : '没有可用取样点';
    $('planDetails').textContent = geometric ? '标定或排布参数已改变，请重新生成。' : depths ? '保存深度修改后即可导出。' : `外 → 内 · ${plan.sampling_order?.layer_count ?? '—'} 层 · ${plan.diameter_mm} mm 直径 · ${plan.gap_mm} mm 间隙`;
  } else {
    $('pointCount').textContent = '—'; $('coverage').textContent = '—';
    $('planStatus').textContent = busy ? '处理中' : '尚未生成';
    $('planDetails').textContent = 'X 向左为正 · Y 向上为正 · 原点为固定螺丝中心';
  }
  const downloadable = Boolean(plan && !geometric && !depths && !busy);
  const files = ['points.csv', 'layout.svg', 'plan.json'];
  exportIds.forEach((id, i) => link(id, plan ? planUrl(files[i]) : null, downloadable));
  link('maskDownload', current?.meta.mask, has && !busy);
  updateScale();
  updateArea();
  if (typeof updateAutomaticControls === 'function') updateAutomaticControls();
  if (typeof updatePLCControls === 'function') updatePLCControls();
  if (typeof updateCameraControls === 'function') updateCameraControls();
}
function updateScale() {
  try {
    const c = calibration();
    $('scaleInfo').textContent = `已自动标定 · ${c.mm_per_pixel.toFixed(5)} mm/px`;
  } catch (error) { $('scaleInfo').textContent = error.message; }
}
async function request(url, body, binary = false, name = '') {
  const headers = binary ? {'Content-Type': 'application/octet-stream', 'X-File-Name': encodeURIComponent(name)} : {'Content-Type': 'application/json'};
  const response = await fetch(url, {method: 'POST', headers, body: binary ? body : JSON.stringify(body)});
  let data;
  try { data = await response.json(); } catch { throw Error('服务响应异常，请检查启动窗口。'); }
  if (!response.ok) throw Error(data.error || '处理失败，请重试。');
  return data;
}
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image(); image.onload = () => resolve(image); image.onerror = () => reject(Error('图片预览加载失败，请重新上传。')); image.src = url;
  });
}
async function addRecognizedPhoto(meta) {
  const [original, region, regionOutline] = await Promise.all([
    loadImage(meta.original),loadImage(meta.region),loadImage(meta.region_outline)]);
  const photo = {meta, original, region, regionOutline, glowCanvas:document.createElement('canvas'),
    plan:null, overrides:{},
    fields:{diameter:$('diameter').value, gap:$('gap').value}};
  photos.push(photo);
  const option=document.createElement('option'); option.value=String(photos.length-1); option.textContent=meta.name;
  $('photoSelect').append(option); showPhoto(photo);
  await createPlanForCurrentPhoto();
}
async function upload(files) {
  if (typeof automaticEnabled !== 'undefined' && automaticEnabled) return uploadAutomatic(files);
  if (busy || !ready || !files.length) return;
  capture(); busy = true; uploading = true; controls(); draw();
  const errors = [];
  for (let i = 0; i < files.length; i++) {
    const file = files[i]; notice(`正在识别 ${i + 1} / ${files.length}：${file.name}`);
    try {
      if (file.size > maxImageBytes) throw Error('单张图片不能超过 100 MB。');
      if (!/\.(jpe?g|png|bmp)$/i.test(file.name)) throw Error('请选择 JPG、PNG 或 BMP 图片。');
      const meta = await request('/api/predict', file, true, file.name);
      await addRecognizedPhoto(meta);
    } catch (error) { errors.push(`${file.name}：${error.message}`); }
  }
  busy = false; uploading = false; $('files').value = ''; controls(); draw();
  if (errors.length) notice(errors.join('；'), true);
  else notice(current.plan?.count ? `已自动识别并生成 ${current.plan.count} 个取样点。测试时可手动写坐标；正式运行请启动自动流程。` : '识别已完成，当前参数没有可用取样点，请调整直径或间隙后重新生成。', !current.plan?.count);
}
function showPhoto(photo) {
  current = photo; selected = null;
  for (const [id, value] of Object.entries(photo.fields)) $(id).value = value;
  $('photoSelect').value = String(photos.indexOf(photo));
  $('photoSelect').hidden = photos.length < 2; $('photoSelectLabel').hidden = photos.length < 2;
  $('filename').textContent = photo.meta.name;
  $('imageInfo').textContent = `${photo.meta.width} × ${photo.meta.height} px · 已识别菌丝区域`;
  $('scopeInfo').textContent = '完整原图保持不变。白色虚线自动圈出培养皿内缘，菌丝结果只显示在边界内。';
  if (photo.meta.infrared?.status === 'excluded') {
    $('scopeInfo').textContent += ' 已自动排除红外亮点及紧贴光晕，附近菌丝仍按模型预测保留。';
  } else if (photo.meta.infrared) {
    $('scopeInfo').textContent += ' 本图未检测到符合条件的红外亮点，请核对原图。';
  }
  $('empty').hidden = true; $('canvas').hidden = false;
  renderTable(); controls(); draw();
}
function renderTable() {
  const rows = $('pointRows'); rows.replaceChildren();
  if (!current?.plan || !current.plan.count) {
    const row = document.createElement('tr'), cell = document.createElement('td'); cell.colSpan = 6; cell.className = 'table-empty';
    cell.textContent = current?.plan ? '当前区域放不下所设大小的圆。可检查标注、标定或调整取样直径。' : '生成排布后显示坐标。';
    row.append(cell); rows.append(row); return;
  }
  const fragment = document.createDocumentFragment();
  for (const p of current.plan.points) {
    const row = document.createElement('tr'); row.dataset.id = String(p.id); row.classList.toggle('selected', selected === p.id);
    for (const value of [String(p.id).padStart(3, '0'), p.layer ? `第 ${p.layer} 层` : '—', p.x_mm.toFixed(3), p.y_mm.toFixed(3)]) {
      const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
    }
    const depthCell = document.createElement('td'), container = document.createElement('div'); container.className = 'depth-cell';
    const input = document.createElement('input'); input.type = 'number'; input.min = '0'; input.max = '1000'; input.step = 'any'; input.dataset.point = String(p.id);
    input.setAttribute('aria-label', `点 ${p.id} 的取样深度`); input.placeholder = String(previewDepth()); input.value = current.overrides[String(p.id)] ?? '';
    const reset = document.createElement('button'); reset.className = 'reset'; reset.textContent = '恢复默认'; reset.type = 'button';
    const modeCell = document.createElement('td'), mode = document.createElement('span'); mode.className = 'mode'; modeCell.append(mode);
    const updateMode = () => { const override = current.overrides[String(p.id)] !== undefined; mode.textContent = override ? '预览自定义' : '默认预览值'; mode.classList.toggle('override', override); };
    input.addEventListener('input', () => {
      if (input.value.trim() === '') delete current.overrides[String(p.id)]; else current.overrides[String(p.id)] = input.value;
      updateMode(); controls();
    });
    reset.addEventListener('click', () => { delete current.overrides[String(p.id)]; input.value = ''; updateMode(); controls(); });
    container.append(input, reset); depthCell.append(container); row.append(depthCell, modeCell); updateMode();
    row.addEventListener('click', () => selectPoint(p.id, false)); fragment.append(row);
  }
  rows.append(fragment);
}
function selectPoint(id, scroll = true) {
  selected = id;
  for (const row of $('pointRows').querySelectorAll('tr[data-id]')) row.classList.toggle('selected', Number(row.dataset.id) === id);
  if (scroll) $('pointRows').querySelector(`tr[data-id="${id}"]`)?.scrollIntoView({behavior: 'smooth', block: 'nearest'});
  const p = current.plan.points.find(item => item.id === id);
  $('pointHint').textContent = `选中点 ${id} · 第 ${p.layer ?? '—'} 层：X ${p.x_mm.toFixed(3)} / Y ${p.y_mm.toFixed(3)} mm。Z 仅为旧版预览字段，不下发 PLC。`;
  draw();
}
function drawRegion(ctx, m, unit) {
  if (!$('showMask').checked || !current.region) return;
  const animation = areaAnimation?.photo === current ? areaAnimation : null;
  if (!animation && displayedAreaKey !== areaKey) return;
  if (!animation && !displayedAreaKey) return;
  const c = animation ? animation.c : current.area.data.calibration;
  ctx.save();
  ctx.beginPath(); ctx.arc(c.center_x_px, c.center_y_px, c.dish_diameter_px/2, 0, Math.PI*2); ctx.clip();
  const boundary = current.meta.dish_boundary;
  if (boundary) {
    ctx.beginPath(); ctx.arc(boundary.center_x_px, boundary.center_y_px, boundary.radius_px, 0, Math.PI*2); ctx.clip();
  }
  const progress = animation?.progress ?? 1;
  const bounds = current.meta.region_bounds || [0, 0, m.width, m.height];
  const left = Math.max(bounds[0], c.center_x_px-c.dish_diameter_px/2);
  const right = Math.min(bounds[0]+bounds[2], c.center_x_px+c.dish_diameter_px/2);
  const band = Math.max(8*unit, (right-left)*.17);
  const eased = progress*progress*(3-2*progress);
  const sweep = left-band+(right-left+2*band)*eased;
  ctx.save(); ctx.beginPath(); ctx.rect(0, 0, animation ? sweep : m.width, m.height); ctx.clip();
  ctx.globalAlpha = .45; ctx.drawImage(current.region, 0, 0, m.width, m.height); ctx.restore();
  if (animation) {
    const fade = progress > .82 ? (1-progress)/.18 : 1;
    ctx.save(); ctx.globalAlpha = (.25+.7*Math.sin(Math.PI*progress))*fade;
    ctx.shadowColor = '#e1fff0'; ctx.shadowBlur = 10*unit;
    ctx.drawImage(current.regionOutline, 0, 0, m.width, m.height); ctx.restore();
    const glow = current.glowCanvas;
    if (glow.width !== current.region.width || glow.height !== current.region.height) {
      glow.width = current.region.width; glow.height = current.region.height;
    }
    const g = glow.getContext('2d'), scale = glow.width/m.width;
    g.clearRect(0, 0, glow.width, glow.height); g.drawImage(current.region, 0, 0);
    g.globalCompositeOperation = 'source-in';
    const gradient = g.createLinearGradient((sweep-band)*scale, 0, (sweep+band)*scale, 0);
    gradient.addColorStop(0, 'rgba(235,255,246,0)'); gradient.addColorStop(.5, `rgba(245,255,250,${.7*fade})`); gradient.addColorStop(1, 'rgba(235,255,246,0)');
    g.fillStyle = gradient; g.fillRect(0, 0, glow.width, glow.height); g.globalCompositeOperation = 'source-over';
    ctx.drawImage(glow, 0, 0, m.width, m.height);
  }
  ctx.restore();
}
function draw() {
  if (!current) return;
  const canvas = $('canvas'), m = current.meta, ratio = Math.min(1, 1800 / m.width);
  const width = Math.round(m.width * ratio), height = Math.round(m.height * ratio);
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  const fit = Math.max(180, $('viewport').clientWidth - 34), z = Number($('zoom').value);
  canvas.style.width = `${fit * z}px`; canvas.style.height = `${fit * z * m.height / m.width}px`;
  const ctx = canvas.getContext('2d'); ctx.drawImage(current.original, 0, 0, canvas.width, canvas.height);
  ctx.save(); ctx.scale(canvas.width / m.width, canvas.height / m.height);
  const unit = m.width / canvas.width, c = m.calibration, cx = c?.center_x_px, cy = c?.center_y_px, d = c?.dish_diameter_px;
  drawRegion(ctx, m, unit);
  const boundary = m.dish_boundary;
  if (boundary) {
    ctx.strokeStyle = '#edf5f0'; ctx.lineWidth = 1.3*unit; ctx.setLineDash([4*unit, 5*unit]);
    ctx.beginPath(); ctx.arc(boundary.center_x_px, boundary.center_y_px, boundary.radius_px, 0, Math.PI*2); ctx.stroke(); ctx.setLineDash([]);
  }
  if (Number.isFinite(cx) && Number.isFinite(cy) && d > 0) {
    ctx.strokeStyle = '#63d2d8'; ctx.lineWidth = 1.5 * unit; ctx.setLineDash([8 * unit, 5 * unit]);
    ctx.beginPath(); ctx.arc(cx, cy, d / 2, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
  }
  let origin; try { origin = coordinateOrigin(); } catch {}
  if (origin) {
    const ox = origin.x_px, oy = origin.y_px, a = 65*unit;
    ctx.strokeStyle = '#ff9dce'; ctx.fillStyle = '#ffb6dc'; ctx.lineWidth = 2*unit;
    ctx.beginPath(); ctx.moveTo(ox+8*unit,oy); ctx.lineTo(ox-a,oy); ctx.moveTo(ox,oy+8*unit); ctx.lineTo(ox,oy-a);
    ctx.moveTo(ox-a+7*unit,oy-4*unit); ctx.lineTo(ox-a,oy); ctx.lineTo(ox-a+7*unit,oy+4*unit);
    ctx.moveTo(ox-4*unit,oy-a+7*unit); ctx.lineTo(ox,oy-a); ctx.lineTo(ox+4*unit,oy-a+7*unit); ctx.stroke();
    ctx.beginPath(); ctx.arc(ox,oy,4*unit,0,Math.PI*2); ctx.stroke();
    ctx.font = `${12*unit}px sans-serif`; ctx.textAlign='left'; ctx.textBaseline='alphabetic';
    ctx.fillText('X+',ox-a-22*unit,oy+4*unit); ctx.fillText('Y+',ox+6*unit,oy-a); ctx.fillText('0 螺丝',ox+7*unit,oy+17*unit);
  }
  const plan = current.plan;
  if (plan) {
    ctx.globalAlpha = geometryDirty() ? .45 : 1;
    for (const p of plan.points) {
      const active = p.id === selected;
      ctx.beginPath(); ctx.arc(p.image_x_px, p.image_y_px, plan.radius_px, 0, Math.PI * 2);
      ctx.fillStyle = active ? 'rgba(255,232,156,.38)' : 'rgba(233,188,76,.12)'; ctx.fill();
      ctx.strokeStyle = active ? '#fff3bd' : '#edc668'; ctx.lineWidth = (active ? 2.5 : 1.2) * unit; ctx.stroke();
      if ($('showNumbers').checked) {
        const font = Math.min(plan.radius_px * .72, 13 * unit);
        ctx.font = `600 ${font}px sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.lineWidth = 2 * unit;
        ctx.strokeStyle = '#1c2e22'; ctx.strokeText(String(p.id), p.image_x_px, p.image_y_px); ctx.fillStyle = '#fff5db'; ctx.fillText(String(p.id), p.image_x_px, p.image_y_px);
      }
    }
  }
  ctx.restore();
}
function previewDepth() { return current?.plan?.default_depth_mm ?? 0; }
async function createPlanForCurrentPhoto() {
  const photo = current;
  const c = calibration(), diameter = numeric('diameter', '取样直径', true), gap = numeric('gap', '间隙');
  capture(); notice('正在自动生成取样排布…');
  const plan = await request('/api/plans/create', {prediction_id:photo.meta.id, calibration:c,
    diameter_mm:diameter, gap_mm:gap});
  photo.plan = plan; photo.overrides = {};
  if (photo === current) {
    selected = null;
    $('pointHint').textContent = '按编号从外层向内层取样。Z 仅为旧版预览字段，不下发 PLC。';
    renderTable(); controls(); draw();
  }
  notice(plan.count ? `已自动生成 ${plan.count} 个取样点，共 ${plan.sampling_order.layer_count} 层。` : '没有满足大小、间隙和边界约束的取样圆，请调整参数后重新生成。', !plan.count);
  return plan;
}
async function generate() {
  if (busy) return;
  busy = true; controls();
  try { await createPlanForCurrentPhoto(); }
  catch (error) { notice(error.message, true); }
  finally { busy = false; controls(); draw(); }
}
async function saveDepths() {
  if (busy || !current?.plan || geometryDirty()) return;
  try {
    const depth = previewDepth(), overrides = {};
    if (depth > 1000) throw Error('预览深度不能超过 1000 mm。');
    for (const [id, raw] of Object.entries(current.overrides)) {
      const value = Number(raw);
      if (!Number.isFinite(value) || value < 0 || value > 1000) throw Error(`点 ${id} 的深度应为 0–1000 mm。`);
      overrides[id] = value;
    }
    capture(); busy = true; controls();
    current.plan = await request('/api/plans/update', {prediction_id: current.meta.id, plan_id: current.plan.plan_id, default_depth_mm: depth, overrides});
    current.overrides = Object.fromEntries(current.plan.points.filter(p => p.depth_overridden).map(p => [String(p.id), String(p.z_mm)]));
    renderTable(); notice('预览深度已保存；该字段不下发 PLC。');
  } catch (error) { notice(error.message, true); }
  finally { busy = false; controls(); draw(); }
}
$('files').addEventListener('change', e => upload([...e.target.files]));
for (const kind of ['dragenter', 'dragover']) $('drop').addEventListener(kind, e => { e.preventDefault(); $('drop').classList.add('over'); });
for (const kind of ['dragleave', 'drop']) $('drop').addEventListener(kind, e => { e.preventDefault(); $('drop').classList.remove('over'); });
$('drop').addEventListener('drop', e => upload([...e.dataTransfer.files]));
$('photoSelect').addEventListener('change', () => { capture(); showPhoto(photos[Number($('photoSelect').value)]); notice('已切换照片；请检查当前标定和参数。'); });
for (const id of ['diameter', 'gap']) $(id).addEventListener('input', () => {
  capture(); controls(); draw();
});
$('canvas').addEventListener('click', event => {
  if (busy || !current) return;
  const rect = $('canvas').getBoundingClientRect(), x = (event.clientX - rect.left) / rect.width * current.meta.width, y = (event.clientY - rect.top) / rect.height * current.meta.height;
  if (current.plan) {
    const p = current.plan.points.find(p => Math.hypot(x - p.image_x_px, y - p.image_y_px) <= current.plan.radius_px);
    if (p) selectPoint(p.id);
  }
});
for (const id of ['showMask', 'showNumbers', 'zoom']) $(id).addEventListener('change', draw);
window.addEventListener('resize', draw);
document.addEventListener('visibilitychange', () => {
  if (document.hidden && areaAnimation) finishAreaAnimation(areaAnimation);
});
motionPreference.addEventListener('change', () => {
  if (motionPreference.matches && areaAnimation) finishAreaAnimation(areaAnimation);
});
$('generate').addEventListener('click', generate);
$('saveDepths').addEventListener('click', saveDepths);
async function health() {
  try {
    const response = await fetch('/api/health'); if (!response.ok) throw Error(); const data = await response.json();
    if (!data.automatic_dish_calibration || data.fixed_dish_diameter_mm !== 90 || !data.original_image_inference || !data.dish_interior_only || !data.region_animation || !data.area_measurement || !data.sampling_planner || data.sampling_order !== 'outer_boundary_layers') throw Error(); backendOrigin = data.fixed_coordinate_origin; maxImageBytes = data.max_upload_bytes || maxImageBytes; ready = Boolean(data.ready && backendOrigin); $('health').textContent = `模型已连接 · ${data.device}`;
    $('statusDot').classList.toggle('ready', ready);
    if (data.shared) $('privacy').textContent = '无密码临时共享：照片经 Cloudflare 转到服务提供者电脑处理并保存。';
  } catch { ready = false; $('health').textContent = '服务未连接'; $('statusDot').classList.remove('ready'); notice('请重新运行 Start_Local_App.cmd，再刷新网页；如旧服务还在运行，请先关闭旧启动窗口。', true); }
  controls();
}
initializeBatchCulture();
health();
