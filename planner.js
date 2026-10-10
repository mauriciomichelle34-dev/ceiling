'use strict';
const $ = id => document.getElementById(id);
const photos = [];
let current = null, ready = false, busy = false, picking = null, selected = null;
const profileKey = 'mycelium-fixed-calibration-v1';
const geometryFields = ['center_x_px', 'center_y_px', 'dish_diameter_px', 'dish_diameter_mm'];
const calibrationInputs = ['centerX', 'centerY', 'dishPx', 'dishMm'];
const exportIds = ['csvDownload', 'layoutDownload', 'jsonDownload', 'deviceDownload'];
let areaKey = '', areaTimer = null, areaSequence = 0;

function showArea(data) {
  $('myceliumArea').textContent = data.mycelium_area_mm2.toLocaleString('zh-CN', {minimumFractionDigits: 2, maximumFractionDigits: 2});
  $('dishCoverage').textContent = data.dish_coverage_percent.toFixed(2);
  $('areaInfo').textContent = `培养皿总面积 ${data.dish_area_mm2.toLocaleString('zh-CN', {maximumFractionDigits: 2})} mm² · 按标定圆内识别出的菌丝区域估算。`;
}
function updateArea() {
  let c;
  try { c = calibration(); } catch (error) {
    clearTimeout(areaTimer); areaKey = ''; areaSequence++;
    $('myceliumArea').textContent = '—'; $('dishCoverage').textContent = '—';
    $('areaInfo').textContent = error.message;
    return;
  }
  const photo = current, key = JSON.stringify([photo.meta.id, c]);
  if (key === areaKey) return;
  clearTimeout(areaTimer); areaKey = key;
  const sequence = ++areaSequence;
  if (photo.area?.key === key) { showArea(photo.area.data); return; }
  $('myceliumArea').textContent = '—'; $('dishCoverage').textContent = '—';
  $('areaInfo').textContent = '正在测算菌丝体面积…';
  areaTimer = setTimeout(async () => {
    try {
      const data = await request('/api/area', {prediction_id: photo.meta.id, calibration: c});
      if (sequence !== areaSequence || photo !== current) return;
      photo.area = {key, data}; showArea(data);
    } catch (error) {
      if (sequence !== areaSequence || photo !== current) return;
      areaKey = ''; $('areaInfo').textContent = `面积测算失败：${error.message} 修改标定参数可重试。`;
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
function calibration() {
  if (!current) throw Error('请先上传照片。');
  const c = {
    center_x_px: numeric('centerX', '圆心横坐标'), center_y_px: numeric('centerY', '圆心纵坐标'),
    dish_diameter_px: numeric('dishPx', '图中圆直径', true), dish_diameter_mm: numeric('dishMm', '培养皿实际直径', true)
  };
  const r = c.dish_diameter_px / 2, m = current.meta;
  if (c.center_x_px - r < -.5 || c.center_y_px - r < -.5 || c.center_x_px + r > m.width - .5 || c.center_y_px + r > m.height - .5) {
    throw Error('标定圆超出图片，请调整圆心和直径。');
  }
  return c;
}
function capture() {
  if (!current) return;
  current.fields = Object.fromEntries([...calibrationInputs, 'diameter', 'gap', 'defaultZ'].map(id => [id, $(id).value]));
}
function geometryDirty() {
  if (!current?.plan) return false;
  try {
    const c = calibration(), p = current.plan;
    return geometryFields.some(key => Math.abs(c[key] - p.calibration[key]) > 1e-8) ||
      numeric('diameter', '取样直径', true) !== p.diameter_mm || numeric('gap', '间隙') !== p.gap_mm;
  } catch { return true; }
}
function depthDirty() {
  if (!current?.plan) return false;
  try {
    if (numeric('defaultZ', '统一深度') !== current.plan.default_depth_mm) return true;
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
  const has = Boolean(current), plan = current?.plan, geometric = geometryDirty(), depths = depthDirty();
  let validCalibration = false;
  try { calibration(); validCalibration = true; } catch {}
  $('files').disabled = busy || !ready;
  $('photoSelect').disabled = busy;
  for (const id of ['pickCenter', 'pickEdge']) $(id).disabled = !has || busy;
  $('saveCalibration').disabled = !validCalibration || busy;
  $('generate').disabled = !has || !ready || busy || !validCalibration;
  $('generate').textContent = busy ? '正在处理…' : plan ? '重新生成取样排布' : '生成取样排布';
  $('saveDepths').disabled = !plan || !depths || geometric || busy;
  $('applyDepth').disabled = !plan || geometric || busy;
  $('previewActions').disabled = !plan || geometric || depths || busy;
  for (const id of [...calibrationInputs, 'diameter', 'gap', 'defaultZ']) $(id).disabled = busy;
  for (const el of $('pointRows').querySelectorAll('input,button')) el.disabled = busy;
  if (plan) {
    $('pointCount').textContent = plan.count;
    $('coverage').textContent = (plan.coverage_fraction * 100).toFixed(1);
    $('planStatus').textContent = busy ? '处理中' : geometric ? '需重新生成' : depths ? '深度待保存' : plan.count ? '规划已保存' : '没有可用取样点';
    $('planDetails').textContent = geometric ? '标定或排布参数已改变，请重新生成。' : depths ? '保存深度修改后即可导出。' : `外 → 内 · ${plan.sampling_order?.layer_count ?? '—'} 层 · ${plan.diameter_mm} mm 直径 · ${plan.gap_mm} mm 间隙`;
  } else {
    $('pointCount').textContent = '—'; $('coverage').textContent = '—';
    $('planStatus').textContent = busy ? '处理中' : '尚未生成';
    $('planDetails').textContent = 'X 向右 · Y 向上 · 原点在培养皿中心';
  }
  const downloadable = Boolean(plan && !geometric && !depths && !busy);
  const files = ['points.csv', 'layout.svg', 'plan.json', 'preview.json'];
  exportIds.forEach((id, i) => link(id, plan ? planUrl(files[i]) : null, downloadable));
  link('maskDownload', current?.meta.mask, has && !busy);
  if (geometric || depths) $('actions').hidden = true;
  updateScale();
  updateArea();
}
function updateScale() {
  try {
    const c = calibration();
    $('scaleInfo').textContent = `1 像素 = ${(c.dish_diameter_mm / c.dish_diameter_px).toFixed(5)} mm · 确认青色标定圆后保存。`;
    const saved = current?.profileUsed;
    $('calibrationBadge').textContent = saved ? '已复用标定' : '已填写';
    $('calibrationBadge').classList.toggle('saved', Boolean(saved));
  } catch (error) {
    $('scaleInfo').textContent = current ? error.message : '请先上传照片并填写实际直径。';
    $('calibrationBadge').textContent = '待确认';
    $('calibrationBadge').classList.remove('saved');
  }
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
function getProfile(meta) {
  try {
    const p = JSON.parse(localStorage.getItem(profileKey));
    return p && p.image_width === meta.width && p.image_height === meta.height ? p : null;
  } catch { return null; }
}
async function upload(files) {
  if (busy || !ready || !files.length) return;
  capture(); busy = true; picking = null; controls();
  const errors = [];
  for (let i = 0; i < files.length; i++) {
    const file = files[i]; notice(`正在识别 ${i + 1} / ${files.length}：${file.name}`);
    try {
      if (file.size > 40 * 1024 * 1024) throw Error('单张图片不能超过 40 MB。');
      if (!/\.(jpe?g|png)$/i.test(file.name)) throw Error('请选择 JPG 或 PNG 图片。');
      const meta = await request('/api/predict', file, true, file.name);
      const [original, overlay] = await Promise.all([loadImage(meta.original), loadImage(meta.overlay)]);
      const profile = getProfile(meta), c = profile || meta.dish_suggestion;
      const photo = {
        meta, original, overlay, plan: null, overrides: {}, profileUsed: Boolean(profile),
        fields: {centerX: String(c.center_x_px), centerY: String(c.center_y_px), dishPx: String(c.dish_diameter_px),
          dishMm: profile ? String(c.dish_diameter_mm) : '', diameter: $('diameter').value, gap: $('gap').value, defaultZ: $('defaultZ').value}
      };
      photos.push(photo);
      const option = document.createElement('option'); option.value = String(photos.length - 1); option.textContent = meta.name; $('photoSelect').append(option);
      showPhoto(photo);
    } catch (error) { errors.push(`${file.name}：${error.message}`); }
  }
  busy = false; $('files').value = ''; controls();
  if (errors.length) notice(errors.join('；'), true);
  else notice(current.profileUsed ? '已复用固定拍摄标定。检查青色圆与培养皿匹配后，生成排布。' : '菌丝识别完成。请填写实际直径，并检查青色圆是否与培养皿匹配。');
}
function showPhoto(photo) {
  current = photo; picking = null; selected = null;
  for (const [id, value] of Object.entries(photo.fields)) $(id).value = value;
  $('photoSelect').value = String(photos.indexOf(photo));
  $('photoSelect').hidden = photos.length < 2; $('photoSelectLabel').hidden = photos.length < 2;
  $('filename').textContent = photo.meta.name;
  $('imageInfo').textContent = `${photo.meta.width} × ${photo.meta.height} px · 已识别菌丝区域`;
  $('empty').hidden = true; $('canvas').hidden = false; $('actions').hidden = true;
  $('pickHint').textContent = '青色圆是标定参考，请确认与培养皿匹配。';
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
    input.setAttribute('aria-label', `点 ${p.id} 的取样深度`); input.placeholder = $('defaultZ').value; input.value = current.overrides[String(p.id)] ?? '';
    const reset = document.createElement('button'); reset.className = 'reset'; reset.textContent = '恢复统一'; reset.type = 'button';
    const modeCell = document.createElement('td'), mode = document.createElement('span'); mode.className = 'mode'; modeCell.append(mode);
    const updateMode = () => { const override = current.overrides[String(p.id)] !== undefined; mode.textContent = override ? '单独设置' : '使用统一值'; mode.classList.toggle('override', override); };
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
  $('pointHint').textContent = `选中点 ${id} · 第 ${p.layer ?? '—'} 层：X ${p.x_mm.toFixed(3)} / Y ${p.y_mm.toFixed(3)} mm。Z 留空时使用统一深度。`;
  draw();
}
function draw() {
  if (!current) return;
  const canvas = $('canvas'), m = current.meta, ratio = Math.min(1, 1800 / m.width);
  canvas.width = Math.round(m.width * ratio); canvas.height = Math.round(m.height * ratio);
  const fit = Math.max(180, $('viewport').clientWidth - 34), z = Number($('zoom').value);
  canvas.style.width = `${fit * z}px`; canvas.style.height = `${fit * z * m.height / m.width}px`;
  canvas.classList.toggle('picking', Boolean(picking));
  const ctx = canvas.getContext('2d'); ctx.drawImage($('showMask').checked ? current.overlay : current.original, 0, 0, canvas.width, canvas.height);
  ctx.save(); ctx.scale(canvas.width / m.width, canvas.height / m.height);
  const unit = m.width / canvas.width, cx = Number($('centerX').value), cy = Number($('centerY').value), d = Number($('dishPx').value);
  if (Number.isFinite(cx) && Number.isFinite(cy) && d > 0) {
    ctx.strokeStyle = '#63d2d8'; ctx.lineWidth = 1.5 * unit; ctx.setLineDash([8 * unit, 5 * unit]);
    ctx.beginPath(); ctx.arc(cx, cy, d / 2, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
    const a = d * .095; ctx.beginPath(); ctx.moveTo(cx - a * .2, cy); ctx.lineTo(cx + a, cy); ctx.moveTo(cx, cy + a * .2); ctx.lineTo(cx, cy - a); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx + a - 6 * unit, cy - 4 * unit); ctx.lineTo(cx + a, cy); ctx.lineTo(cx + a - 6 * unit, cy + 4 * unit); ctx.moveTo(cx - 4 * unit, cy - a + 6 * unit); ctx.lineTo(cx, cy - a); ctx.lineTo(cx + 4 * unit, cy - a + 6 * unit); ctx.stroke();
    ctx.fillStyle = '#94e1e3'; ctx.font = `${12 * unit}px sans-serif`; ctx.fillText('X', cx + a + 5 * unit, cy); ctx.fillText('Y', cx + 5 * unit, cy - a); ctx.fillText('0', cx + 5 * unit, cy + 14 * unit);
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
async function generate() {
  if (busy) return;
  try {
    const c = calibration(), diameter = numeric('diameter', '取样直径', true), gap = numeric('gap', '间隙'), depth = numeric('defaultZ', '统一深度');
    capture(); busy = true; picking = null; controls(); notice('正在尝试不同方向与偏移，计算完整落在菌丝区域内的取样圆…');
    const plan = await request('/api/plans/create', {prediction_id: current.meta.id, calibration: c, diameter_mm: diameter, gap_mm: gap, default_depth_mm: depth});
    current.plan = plan; current.overrides = {}; selected = null; $('actions').hidden = true;
    $('pointHint').textContent = '按编号从外层向内层取样；Z 留空时使用统一深度。'; renderTable();
    notice(plan.count ? `已保存 ${plan.count} 个取样点，编号从外圈逐层向内，共 ${plan.sampling_order.layer_count} 层。` : '没有满足大小、间隙和边界约束的取样圆，请检查参数和菌丝区域。', !plan.count);
  } catch (error) { notice(error.message, true); }
  finally { busy = false; controls(); draw(); }
}
async function saveDepths() {
  if (busy || !current?.plan || geometryDirty()) return;
  try {
    const depth = numeric('defaultZ', '统一深度'), overrides = {};
    if (depth > 1000) throw Error('统一深度不能超过 1000 mm。');
    for (const [id, raw] of Object.entries(current.overrides)) {
      const value = Number(raw);
      if (!Number.isFinite(value) || value < 0 || value > 1000) throw Error(`点 ${id} 的深度应为 0–1000 mm。`);
      overrides[id] = value;
    }
    capture(); busy = true; controls();
    current.plan = await request('/api/plans/update', {prediction_id: current.meta.id, plan_id: current.plan.plan_id, default_depth_mm: depth, overrides});
    current.overrides = Object.fromEntries(current.plan.points.filter(p => p.depth_overridden).map(p => [String(p.id), String(p.z_mm)]));
    renderTable(); notice('深度已保存；单独设置的点保持自己的深度，其余点使用统一值。');
  } catch (error) { notice(error.message, true); }
  finally { busy = false; controls(); draw(); }
}
$('files').addEventListener('change', e => upload([...e.target.files]));
for (const kind of ['dragenter', 'dragover']) $('drop').addEventListener(kind, e => { e.preventDefault(); $('drop').classList.add('over'); });
for (const kind of ['dragleave', 'drop']) $('drop').addEventListener(kind, e => { e.preventDefault(); $('drop').classList.remove('over'); });
$('drop').addEventListener('drop', e => upload([...e.dataTransfer.files]));
$('photoSelect').addEventListener('change', () => { capture(); showPhoto(photos[Number($('photoSelect').value)]); notice('已切换照片；请检查当前标定和参数。'); });
for (const id of [...calibrationInputs, 'diameter', 'gap', 'defaultZ']) $(id).addEventListener('input', () => {
  if (calibrationInputs.includes(id) && current) current.profileUsed = false;
  capture(); controls(); draw();
  if (id === 'defaultZ') for (const input of $('pointRows').querySelectorAll('input')) input.placeholder = $('defaultZ').value;
});
$('saveCalibration').addEventListener('click', () => {
  try {
    const c = calibration();
    localStorage.setItem(profileKey, JSON.stringify({...c, image_width: current.meta.width, image_height: current.meta.height}));
    current.profileUsed = true; controls(); notice('固定拍摄标定已保存在这个浏览器中。同尺寸且培养皿位置、大小一致的照片会复用。');
  } catch (error) { notice(error.message, true); }
});
function beginPick(mode) {
  picking = picking === mode ? null : mode;
  $('pickHint').textContent = picking === 'center' ? '请在右侧图片上点击培养皿圆心。' : picking === 'edge' ? '请点击培养皿的圆边，直径按到圆心的距离计算。' : '青色圆是标定参考，请确认与培养皿匹配。'; draw();
}
$('pickCenter').addEventListener('click', () => beginPick('center'));
$('pickEdge').addEventListener('click', () => beginPick('edge'));
$('canvas').addEventListener('click', event => {
  if (busy || !current) return;
  const rect = $('canvas').getBoundingClientRect(), x = (event.clientX - rect.left) / rect.width * current.meta.width, y = (event.clientY - rect.top) / rect.height * current.meta.height;
  if (picking) {
    if (picking === 'center') { $('centerX').value = x.toFixed(3); $('centerY').value = y.toFixed(3); }
    else $('dishPx').value = (2 * Math.hypot(x - Number($('centerX').value), y - Number($('centerY').value))).toFixed(3);
    picking = null; current.profileUsed = false; capture(); controls(); draw(); $('pickHint').textContent = '已调整标定圆，请核对后保存。'; return;
  }
  if (current.plan) {
    const p = current.plan.points.find(p => Math.hypot(x - p.image_x_px, y - p.image_y_px) <= current.plan.radius_px);
    if (p) selectPoint(p.id);
  }
});
for (const id of ['showMask', 'showNumbers', 'zoom']) $(id).addEventListener('change', draw);
window.addEventListener('resize', draw);
$('generate').addEventListener('click', generate);
$('saveDepths').addEventListener('click', saveDepths);
$('applyDepth').addEventListener('click', saveDepths);
$('previewActions').addEventListener('click', async () => {
  try {
    const response = await fetch(planUrl('preview.json')); if (!response.ok) throw Error('指令预览读取失败。');
    const data = await response.json(); $('actions').textContent = JSON.stringify(data, null, 2); $('actions').hidden = false;
    notice('这里只显示取样位置与深度指令，尚未连接设备。');
  } catch (error) { notice(error.message, true); }
});
async function health() {
  try {
    const response = await fetch('/api/health'); if (!response.ok) throw Error(); const data = await response.json();
    if (!data.area_measurement || !data.sampling_planner || data.sampling_order !== 'outer_boundary_layers') throw Error(); ready = Boolean(data.ready); $('health').textContent = `模型已连接 · ${data.device}`;
    $('statusDot').classList.toggle('ready', ready);
    if (data.shared) $('privacy').textContent = '无密码临时共享：照片经 Cloudflare 转到服务提供者电脑处理并保存。';
  } catch { ready = false; $('health').textContent = '服务未连接'; $('statusDot').classList.remove('ready'); notice('请重新运行 Start_Local_App.cmd，再刷新网页；如旧服务还在运行，请先关闭旧启动窗口。', true); }
  controls();
}
health();
