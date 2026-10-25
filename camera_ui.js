'use strict';
let cameraSnapshot = null;
let cameraWorking = false;

function updateCameraControls() {
  const connected = Boolean(cameraSnapshot?.connected);
  const working = busy || cameraWorking || Boolean(cameraSnapshot?.busy)
    || (typeof plcCameraControls === 'function' && plcCameraControls());
  $('cameraSearch').disabled = working;
  $('cameraSelect').disabled = working || connected;
  $('cameraConnect').disabled = working || connected || !$('cameraSelect').value;
  $('cameraDisconnect').disabled = working || !connected;
  $('cameraCapture').disabled = working || !connected || !ready;
}

async function cameraRequest(action, body = {}) {
  const response = await fetch(`/api/camera/${action}`,{method:'POST',
    headers:{'Content-Type':'application/json','X-PLC-Request':'1'},body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw Error(data.error || '相机操作失败。');
  return data;
}

function showCameraFrame(frame) {
  if (!frame) return;
  $('cameraPreview').src = frame.image_url; $('cameraPreview').hidden = false;
  $('cameraDownload').href = frame.image_url; $('cameraDownload').hidden = false;
}

function renderCamera(status) {
  cameraSnapshot = status;
  const oldSelection = $('cameraSelect').value;
  const devices = [...status.devices];
  if (status.device && !devices.some(device=>device.id===status.device.id)) devices.unshift(status.device);
  $('cameraSelect').replaceChildren();
  if (!devices.length) {
    const option=document.createElement('option'); option.value=''; option.textContent='暂未搜索到相机';
    $('cameraSelect').append(option);
  }
  for (const device of devices) {
    const option=document.createElement('option'); option.value=device.id;
    option.textContent=`${device.model} · ${device.ip} · ${device.serial}`;
    $('cameraSelect').append(option);
  }
  if ([...$('cameraSelect').options].some(o=>o.value===oldSelection)) $('cameraSelect').value=oldSelection;
  if (status.device) $('cameraSelect').value=status.device.id;
  $('cameraBadge').textContent=status.connected ? '已连接' : '未连接';
  $('cameraBadge').classList.toggle('saved',status.connected);
  $('cameraStatus').textContent=status.connected ? `已连接 ${status.device.model} · ${status.device.ip}，可以点击拍照。`
    : status.sdk_loaded ? `SDK ${status.sdk_version} 已加载，搜索到 ${status.devices.length} 台相机。`
    : '点击“搜索相机”加载 MVS SDK 并查找设备。';
  if (status.last_error) $('cameraStatus').textContent=status.last_error;
  else if (status.last_capture?.processing_error) $('cameraStatus').textContent=status.last_capture.processing_error;
  if (status.last_capture) showCameraFrame(status.last_capture);
  updateCameraControls();
}

async function cameraAction(action) {
  if (cameraWorking || busy) return;
  cameraWorking=true;
  if (action==='capture') { busy=true; capture(); controls(); }
  updateCameraControls();
  $('cameraStatus').textContent=action==='capture' ? '正在拍照并接收图像…' : '正在处理相机操作…';
  try {
    if (action==='capture') {
      const frame=await cameraRequest('capture',{capture_id:crypto.randomUUID()});
      showCameraFrame(frame);
      if (frame.processing_error) {
        $('cameraStatus').textContent=frame.processing_error;
        notice(frame.processing_error,true);
      } else if (frame.processing?.kind==='automatic') {
        $('cameraStatus').textContent=`已拍到 ${frame.width} × ${frame.height} 照片，已进入自动处理队列。`;
        notice('照片已接收，正在按当前模式自动识别和处理。');
        refreshAutomatic();
      } else {
        await addRecognizedPhoto(frame.processing.meta);
        $('cameraStatus').textContent=`已拍照并识别：${frame.width} × ${frame.height}。`;
        notice('已拍照并识别。请确认培养皿尺寸与标定；启动自动模式后，下一张照片会直接进入自动处理。');
      }
    } else {
      const body=action==='connect' ? {device_id:$('cameraSelect').value} : {};
      renderCamera(await cameraRequest(action,body));
    }
  } catch (error) {
    $('cameraStatus').textContent=error.message;
    notice(error.message,true);
  } finally {
    if (action==='capture') { busy=false; controls(); }
    cameraWorking=false;
    updateCameraControls();
  }
}

$('cameraSelect').addEventListener('change',updateCameraControls);
for (const [id,action] of [['cameraSearch','search'],['cameraConnect','connect'],
  ['cameraDisconnect','disconnect'],['cameraCapture','capture']]) $(id).addEventListener('click',()=>cameraAction(action));
cameraRequest('status').then(renderCamera).catch(error=>{
  $('cameraStatus').textContent=`相机接口尚未就绪，请确认新版程序已启动：${error.message}`;
});
