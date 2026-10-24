'use strict';
let automaticEnabled = false;
let automaticSnapshot = null;
let automaticTimer = null;
let automaticReading = false;
const automaticLoaded = new Set();
const automaticPhases = {queued:'等待处理', recognizing:'识别中', planning:'划圈中',
  waiting_idle:'等待交接空闲', sending:'写入 PLC', writing_coordinates:'覆盖坐标', waiting_taken:'等待 PLC 取数',
  releasing:'收回 PC 信号', waiting_reset:'等待 PLC 复位', completed:'交接完成',
  failed:'已停止 · 需处理', cancelled:'已停止', interrupted:'上次中断 · 未重发'};

function automaticLocksPLC() {
  return automaticEnabled || Boolean(automaticSnapshot?.active_job_id);
}

function updateAutomaticControls() {
  let valid = false;
  try {
    calibration(); numeric('diameter', '取样圆直径', true);
    numeric('gap', '间隙'); numeric('defaultZ', '统一深度'); valid = true;
  } catch {}
  const active = Boolean(automaticSnapshot?.active_job_id);
  $('automaticSaveStart').disabled = busy || automaticEnabled || active || !valid;
  $('automaticStartSaved').disabled = busy || automaticEnabled || active || !automaticSnapshot?.configured;
  $('automaticStop').disabled = !automaticEnabled && !active;
  $('automaticMode').disabled = busy || automaticEnabled || active;
}

async function automaticRequest(action, body = {}) {
  const response = await fetch(`/api/automatic/${action}`, {method:'POST',
    headers:{'Content-Type':'application/json', 'X-PLC-Request':'1'}, body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw Error(data.error || '自动处理请求失败。');
  return data;
}

function renderAutomatic(data) {
  automaticSnapshot = data; automaticEnabled = data.enabled;
  const mode = data.settings?.transfer_mode || 'execute';
  $('automaticBadge').textContent = data.enabled ? mode === 'coordinates_only' ? '坐标测试运行' : '联动运行' : data.active_job_id ? '正在结束当前任务' : '未启动';
  if (data.enabled || data.active_job_id) $('automaticMode').value = mode;
  updateAutomaticFlow();
  $('automaticBadge').classList.toggle('saved', data.enabled);
  const settings = data.settings;
  $('automaticSettings').textContent = settings
    ? `${mode === 'coordinates_only' ? '坐标覆盖测试' : '联动执行'}；固定参数：${settings.image_width} × ${settings.image_height} px；培养皿 ${settings.calibration.dish_diameter_mm} mm；取样圆 ${settings.diameter_mm} mm；间隙 ${settings.gap_mm} mm。`
    : '首次使用：先上传参考照片，填写培养皿实际直径并确认标定，再保存当前参数。';
  const active = data.jobs.find(job => job.job_id === data.active_job_id);
  const latest = active || data.jobs[0];
  if (latest?.plc_status) showPLCStatus(latest.plc_status);
  $('automaticStatus').textContent = active ? active.message
    : data.enabled ? `已启动，等待新照片。待处理 ${data.queued} 张。`
    : latest?.state === 'failed' ? `自动处理已停止：${latest.message}`
    : latest?.state === 'interrupted' ? latest.message : '自动处理未启动。';
  const rows = $('automaticJobs'); rows.replaceChildren();
  for (const job of data.jobs) {
    const row = document.createElement('tr');
    const phase = job.state === 'completed' && job.transfer_result?.mode === 'coordinates_only'
      ? '覆盖完成' : automaticPhases[job.state] || job.state;
    for (const value of [job.filename, phase, job.count ?? '—', job.message]) {
      const cell = document.createElement('td'); cell.textContent = String(value); row.append(cell);
    }
    rows.append(row);
  }
  if (!data.jobs.length) {
    const row = document.createElement('tr'), cell = document.createElement('td');
    cell.colSpan = 4; cell.className = 'table-empty'; cell.textContent = '还没有自动处理记录。';
    row.append(cell); rows.append(row);
  }
  controls();
}

async function displayAutomaticResult(job) {
  if (!job.meta || automaticLoaded.has(job.job_id)) return;
  const settings = job.applied_settings || automaticSnapshot.settings, c = settings.calibration;
  const [original, region, regionOutline] = await Promise.all([
    loadImage(job.meta.original), loadImage(job.meta.region), loadImage(job.meta.region_outline)]);
  const photo = {meta:job.meta, original, region, regionOutline,
    glowCanvas:document.createElement('canvas'), plan:job.plan || null, overrides:{}, profileUsed:true,
    fields:{centerX:String(c.center_x_px), centerY:String(c.center_y_px), dishPx:String(c.dish_diameter_px),
      dishMm:String(c.dish_diameter_mm), diameter:String(settings.diameter_mm), gap:String(settings.gap_mm),
      defaultZ:String(settings.default_depth_mm)}};
  photos.push(photo);
  const option = document.createElement('option'); option.value = String(photos.length-1);
  option.textContent = job.filename; $('photoSelect').append(option);
  automaticLoaded.add(job.job_id);
  showPhoto(photo);
}

async function refreshAutomatic() {
  if (automaticReading) return;
  automaticReading = true;
  clearTimeout(automaticTimer);
  try {
    const data = await automaticRequest('status'); renderAutomatic(data);
    const candidate = data.jobs.find(job => job.has_result
      && (job.has_plan || ['failed','interrupted','cancelled'].includes(job.state)));
    // Display the newest result without needing a manual generation/send click.
    if (candidate && !automaticLoaded.has(candidate.job_id) && !busy) {
      const detail = await automaticRequest('job', {job_id:candidate.job_id});
      await displayAutomaticResult(detail);
    }
  } catch (error) {
    $('automaticStatus').textContent = `自动处理状态暂时无法读取：${error.message}`;
  } finally {
    automaticReading = false;
    if (automaticEnabled || automaticSnapshot?.active_job_id) automaticTimer = setTimeout(refreshAutomatic, 1000);
  }
}

async function startAutomatic(useSaved) {
  if (busy) return;
  try {
    if ($('automaticMode').value === 'coordinates_only') {
      const capabilitiesResponse = await fetch('/api/health');
      const capabilities = await capabilitiesResponse.json();
      if (!capabilities.coordinate_test_available) throw Error('当前后台尚未支持坐标测试，请先重启新版程序。');
    }
    let body = useSaved ? {settings:{...automaticSnapshot.settings, transfer_mode:$('automaticMode').value}} : {};
    if (!useSaved) body.settings = {image_width:current.meta.width, image_height:current.meta.height,
      calibration:calibration(), diameter_mm:numeric('diameter', '取样圆直径', true),
      gap_mm:numeric('gap', '间隙'), default_depth_mm:numeric('defaultZ', '统一深度'), plc_timeout_s:60,
      transfer_mode:$('automaticMode').value};
    renderAutomatic(await automaticRequest('start', body));
    notice($('automaticMode').value === 'coordinates_only'
      ? '坐标覆盖测试已启动。新照片自动覆盖数量和坐标，4.0 保持 False，不发送联动通知。'
      : '联动执行已启动。新照片自动识别、划圈、写入 PLC 并发送完成信号。');
    refreshAutomatic();
  } catch (error) { notice(error.message, true); }
}

async function uploadAutomatic(files) {
  if (busy || !ready || !files.length) return;
  busy = true; uploading = true; controls();
  try {
    for (const file of files) {
      if (file.size > 40*1024*1024 || !/\.(jpe?g|png)$/i.test(file.name)) throw Error('请选择 40 MB 以内的 JPG 或 PNG。');
      const response = await fetch('/api/automatic/images', {method:'POST', headers:{
        'Content-Type':'application/octet-stream', 'X-PLC-Request':'1',
        'X-File-Name':encodeURIComponent(file.name), 'X-Capture-ID':crypto.randomUUID()}, body:file});
      const data = await response.json();
      if (!response.ok) throw Error(data.error || '图片接收失败。');
      notice(`${file.name} 已进入自动处理队列。`);
    }
  } catch (error) { notice(error.message, true); }
  finally {
    busy = false; uploading = false; $('files').value = ''; controls();
    refreshAutomatic();
  }
}

function updateAutomaticFlow() {
  $('automaticFlow').textContent = $('automaticMode').value === 'coordinates_only'
    ? '接收照片 → 菌丝识别 → 划圈取点 → 覆盖 DB100 数量与坐标 → 等待下一张'
    : '接收照片 → 菌丝识别 → 划圈取点 → 写入 DB100 → 等待 PLC 取数 → 接收下一张';
}
$('automaticMode').addEventListener('change', updateAutomaticFlow);
$('automaticSaveStart').addEventListener('click', () => startAutomatic(false));
$('automaticStartSaved').addEventListener('click', () => startAutomatic(true));
$('automaticStop').addEventListener('click', async () => {
  try {
    renderAutomatic(await automaticRequest('stop'));
    notice('已停止接收和下发新图片。已开始的底层传输会结束；已发给 PLC 的数据不会撤回。');
    refreshAutomatic();
  } catch (error) { notice(error.message, true); }
});
refreshAutomatic();
