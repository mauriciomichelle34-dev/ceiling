'use strict';
let automaticEnabled = false;
let automaticSnapshot = null;
let automaticTimer = null;
let automaticReading = false;
const automaticLoaded = new Set();
const automaticPhases = {queued:'等待处理', recognizing:'识别中', planning:'划圈中',
  waiting_idle:'等待交接空闲', sending:'写入 PLC', writing_coordinates:'覆盖坐标', waiting_taken:'等待 PLC 取数',
  releasing:'收回 PC 信号', waiting_reset:'等待 PLC 复位', completed:'交接完成',
  waiting_cycle_finish:'等待设备完成及复位',
  failed:'已停止 · 需处理', cancelled:'已停止', interrupted:'上次中断 · 未重发'};

function automaticLocksPLC() {
  return automaticEnabled || Boolean(automaticSnapshot?.active_job_id || automaticSnapshot?.camera_busy);
}

function plcCameraControls() {
  return automaticLocksPLC() && automaticSnapshot?.settings?.input_source === 'plc_camera';
}

function automaticReference() {
  if (current) return {meta:current.meta, calibration:calibration(true), reused:false};
  const frame = typeof cameraSnapshot === 'undefined' ? null : cameraSnapshot?.last_capture;
  const meta = !frame?.processing_error && frame?.processing?.kind === 'analysis' ? frame.processing.meta : null;
  if (!meta?.calibration || meta.calibration.dish_diameter_mm !== 90) {
    throw Error('请先拍摄或上传一张识别成功的参考照片，再启动自动处理。');
  }
  return {meta, calibration:{...meta.calibration, coordinate_origin:coordinateOrigin(meta)}, reused:true};
}

function updateAutomaticControls() {
  let valid = false;
  let reason = '';
  let reference;
  try {
    reference = automaticReference(); numeric('diameter', '取样圆直径', true);
    numeric('gap', '间隙'); valid = true;
  } catch (error) { reason = error.message; }
  const active = Boolean(automaticSnapshot?.active_job_id || automaticSnapshot?.camera_busy);
  $('automaticSaveStart').disabled = !ready || busy || automaticEnabled || active || !valid;
  $('automaticStartSaved').disabled = !ready || busy || automaticEnabled || active || !automaticSnapshot?.configured;
  $('automaticStop').disabled = !automaticEnabled && !active;
  $('automaticMode').disabled = busy || automaticEnabled || active || $('automaticSource').value === 'plc_camera';
  $('automaticSource').disabled = busy || automaticEnabled || active;
  $('automaticCycleTimeout').disabled = busy || automaticEnabled || active;
  $('automaticStartHint').textContent = automaticEnabled ? '自动流程已启动。'
    : active ? '当前任务尚未结束，请等待后再启动。'
    : busy ? '正在处理照片，请等待完成。'
    : !ready ? '正在等待识别服务及固定原点加载。'
    : !valid ? reason
    : reference.reused ? `可复用最近一次识别成功的相机照片（${reference.meta.width} × ${reference.meta.height}）；点击启动后才监听 PLC 并下发。`
    : '参数已就绪，点击启动后才开始自动处理。';
  $('automaticSaveStart').title = $('automaticSaveStart').disabled ? $('automaticStartHint').textContent : '';
}

async function automaticRequest(action, body = {}) {
  const response = await fetch(`/api/automatic/${action}`, {method:'POST',
    headers:{'Content-Type':'application/json', 'X-PLC-Request':'1'}, body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw Error(data.error || '自动处理请求失败。');
  return data;
}

function renderAutomatic(data) {
  const initial = automaticSnapshot === null;
  automaticSnapshot = data; automaticEnabled = data.enabled;
  const mode = data.settings?.transfer_mode || 'execute';
  const plcCamera = data.settings?.input_source === 'plc_camera';
  $('automaticBadge').textContent = data.enabled ? plcCamera ? '无人循环运行' : mode === 'coordinates_only' ? '坐标测试运行' : '联动运行' : data.active_job_id || data.camera_busy ? '正在结束当前任务' : '未启动';
  if (data.enabled || data.active_job_id || data.camera_busy || (initial && data.settings)) {
    $('automaticMode').value = mode;
    $('automaticSource').value = data.settings.input_source || 'received_images';
    $('automaticCycleTimeout').value = data.settings.cycle_timeout_s || 1800;
  }
  updateAutomaticFlow();
  $('automaticBadge').classList.toggle('saved', data.enabled);
  const settings = data.settings;
  $('automaticSettings').textContent = settings
    ? `${plcCamera ? 'PLC 到位自动拍照' : mode === 'coordinates_only' ? '坐标覆盖测试' : '联动执行'}；固定参数：${settings.image_width} × ${settings.image_height} px；培养皿 ${settings.calibration.dish_diameter_mm} mm；取样圆 ${settings.diameter_mm} mm；间隙 ${settings.gap_mm} mm。${settings.calibration.coordinate_origin ? '后台自动使用固定螺丝原点；每张照片重新定位皿中心。' : '旧参数使用培养皿中心，请设置螺丝原点后重新保存参数。'}`
    : '首次使用：拍照或上传参考照片后，设置取样参数并启动自动处理。';
  const active = data.jobs.find(job => job.job_id === data.active_job_id);
  const latest = active || data.jobs[0];
  if (active?.plc_status) showPLCStatus(active.plc_status);
  else if (data.enabled && plcCamera && data.camera_trigger?.plc_status) showPLCStatus(data.camera_trigger.plc_status);
  else if (latest?.plc_status) showPLCStatus(latest.plc_status);
  $('automaticStatus').textContent = active ? active.message
    : data.enabled ? plcCamera ? data.camera_trigger.message : `已启动，等待新照片。待处理 ${data.queued} 张。`
    : data.last_error ? `自动处理已停止：${data.last_error}`
    : latest?.state === 'failed' ? `自动处理已停止：${latest.message}`
    : latest?.state === 'interrupted' ? latest.message : '自动处理未启动。';
  if (data.camera_trigger?.image_url && typeof showCameraFrame === 'function') showCameraFrame(data.camera_trigger);
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
  const settings = job.applied_settings || automaticSnapshot.settings;
  const [original, region, regionOutline] = await Promise.all([
    loadImage(job.meta.original), loadImage(job.meta.region), loadImage(job.meta.region_outline)]);
  const photo = {meta:job.meta, original, region, regionOutline,
    glowCanvas:document.createElement('canvas'), plan:job.plan || null, overrides:{},
    fields:{diameter:String(settings.diameter_mm), gap:String(settings.gap_mm)}};
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
    if (automaticEnabled || automaticSnapshot?.active_job_id || automaticSnapshot?.camera_busy) automaticTimer = setTimeout(refreshAutomatic, 1000);
  }
}

async function startAutomatic(useSaved) {
  if (busy) return;
  try {
    if (!ready) throw Error('请等待识别服务及固定原点加载。');
    const reference = useSaved ? null : automaticReference();
    const source = $('automaticSource').value;
    if (source === 'plc_camera') {
      const response = await fetch('/api/health'), capabilities = await response.json();
      if (!capabilities.plc_camera_trigger_available) throw Error('请先重启支持 PLC 到位拍照的新版后台。');
      $('automaticMode').value = 'execute';
    }
    if ($('automaticMode').value === 'coordinates_only') {
      const capabilitiesResponse = await fetch('/api/health');
      const capabilities = await capabilitiesResponse.json();
      if (!capabilities.coordinate_test_available) throw Error('当前后台尚未支持坐标测试，请先重启新版程序。');
    }
    let body = useSaved ? {settings:{...automaticSnapshot.settings, transfer_mode:$('automaticMode').value,
      input_source:source, cycle_timeout_s:numeric('automaticCycleTimeout','设备循环等待时间',true)}} : {};
    if (!useSaved) body.settings = {image_width:reference.meta.width, image_height:reference.meta.height,
      calibration:reference.calibration, diameter_mm:numeric('diameter', '取样圆直径', true),
      gap_mm:numeric('gap', '间隙'), plc_timeout_s:60,
      transfer_mode:$('automaticMode').value, input_source:source,
      cycle_timeout_s:numeric('automaticCycleTimeout','设备循环等待时间',true)};
    renderAutomatic(await automaticRequest('start', body));
    notice(source === 'plc_camera' ? '无人循环已启动：监听 PLC 到位，自动拍照、识别、划圈和下发；等待 PLC 复位后响应下一轮。' : $('automaticMode').value === 'coordinates_only'
      ? '坐标覆盖测试已启动。新照片自动覆盖数量和坐标，4.0 保持 False，不发送联动通知。'
      : '联动执行已启动。新照片自动识别、划圈、写入 PLC 并发送完成信号。');
    refreshAutomatic();
  } catch (error) { notice(error.message, true); }
}

async function uploadAutomatic(files) {
  if (busy || !ready || !files.length) return;
  if (plcCameraControls()) { notice('PLC 到位自动拍照运行中，请先停止后再手动上传。',true); return; }
  busy = true; uploading = true; controls();
  try {
    for (const file of files) {
      if (file.size > maxImageBytes || !/\.(jpe?g|png|bmp)$/i.test(file.name)) throw Error('请选择 100 MB 以内的 JPG、PNG 或 BMP。');
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
  const plcCamera = $('automaticSource').value === 'plc_camera';
  $('automaticCycleOptions').hidden = !plcCamera;
  if (plcCamera) {
    $('automaticMode').value = 'execute';
    $('automaticFlow').textContent = '4.2=True → 自动拍照 → 识别划圈 → 写数量与坐标 → 4.0=True → PLC 执行动作并复位两位 → 下一轮';
    updateAutomaticControls();
    return;
  }
  $('automaticFlow').textContent = $('automaticMode').value === 'coordinates_only'
    ? '接收照片 → 菌丝识别 → 划圈取点 → 覆盖 DB100 数量与坐标 → 等待下一张'
    : '接收照片 → 菌丝识别 → 划圈取点 → 写入 DB100 → 等待 PLC 取数 → 接收下一张';
  updateAutomaticControls();
}
$('automaticSource').addEventListener('change', updateAutomaticFlow);
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
