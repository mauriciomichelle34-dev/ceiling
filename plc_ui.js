'use strict';
let plcPreparedKey = null;
let plcLastStatus = null;

function plcPlanKey() {
  return current?.plan ? JSON.stringify([current.meta.id, current.plan.plan_id,
    current.plan.points.map(p => [p.x_mm, p.y_mm])]) : null;
}

function updatePLCControls() {
  const automaticLock = typeof automaticLocksPLC === 'function' && automaticLocksPLC();
  const plan = current?.plan;
  const dirty = geometryDirty() || depthDirty();
  const usable = Boolean(plan && plan.count >= 1 && plan.count <= 64 && !dirty);
  if (dirty || plcPreparedKey !== plcPlanKey()) {
    plcPreparedKey = null;
    $('plcPreviewTable').hidden = true;
    $('plcPreviewInfo').textContent = plan?.count > 64
      ? `当前 ${plan.count} 个点超过 DB100 的 64 个容量，无法整批发送；不会截断或自动分批。` : '';
  }
  $('plcPreview').disabled = busy || !usable;
  $('plcSend').disabled = busy || automaticLock || !usable || !plcPreparedKey;
  $('plcTestWrite').disabled = busy || automaticLock || !usable;
  $('plcRead').disabled = busy;
  const plcResetsCycle = typeof automaticSnapshot !== 'undefined' && automaticSnapshot?.settings?.input_source === 'plc_camera';
  $('plcRelease').disabled = busy || automaticLock || plcResetsCycle || !plcLastStatus?.plc_taken;
}

function showPLCStatus(status) {
  plcLastStatus = status;
  $('plcConnection').textContent = `PLC 已响应 · ${status.host || 'DB100'}`;
  $('plcStatus').textContent = `PLC 数量 ${status.count}；PC给完数据 ${status.pc_ready ? 'True' : 'False'}；PLC点位取完 ${status.plc_taken ? 'True' : 'False'}。`;
}

async function plcRequest(action, body = {}) {
  const response = await fetch(`/api/plc/${action}`, {method: 'POST',
    headers: {'Content-Type': 'application/json', 'X-PLC-Request': '1'},
    body: JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw Error(data.error || 'PLC 操作失败，请读取状态后再处理。');
  return data;
}

async function plcAction(action) {
  if (busy) return;
  if (['preview', 'send', 'test-write'].includes(action)
      && (!current?.plan || geometryDirty() || depthDirty())) return;
  if (action === 'send' && plcPreparedKey !== plcPlanKey()) return;
  const body = current?.plan ? {prediction_id: current.meta.id, plan_id: current.plan.plan_id} : {};
  busy = true;
  if (['send', 'test-write'].includes(action)) plcPreparedKey = null;
  controls();
  $('plcStatus').textContent = action === 'send' ? '正在写入并核对坐标，请等待结果…' : '正在处理…';
  try {
    const data = await plcRequest(action, body);
    if (action === 'preview') {
      const rows = $('plcRows'); rows.replaceChildren();
      for (const p of data.points) {
        const row = document.createElement('tr');
        for (const value of [`P[${p.index}]`, p.x_mm, p.y_mm, p.x, p.y]) {
          const cell = document.createElement('td'); cell.textContent = String(value); row.append(cell);
        }
        rows.append(row);
      }
      plcPreparedKey = plcPlanKey();
      $('plcPreviewTable').hidden = false;
      $('plcPreviewInfo').textContent = `${data.count} 个点，相对本张照片的螺丝原点，按原编号顺序发送；1 mm = ${data.units_per_mm}，精度 0.1 mm，按四舍五入转换。补偿字段和 Z 深度不写入。`;
      $('plcStatus').textContent = '坐标已准备好；可选择“仅覆盖坐标（测试）”或“联动发送坐标到 PLC”。';
    } else if (action === 'status') {
      showPLCStatus(data);
    } else if (action === 'send' || action === 'test-write') {
      if (data.status) showPLCStatus(data.status);
      $('plcStatus').textContent = data.message + (data.warning ? ` ${data.warning}`
        : action === 'test-write' ? ' 可继续上传新照片覆盖测试。' : ' 可读取 PLC 状态查看取数反馈。');
      notice(data.message);
    } else {
      plcLastStatus = null;
      $('plcConnection').textContent = 'PLC 已响应 · 请重新读取状态';
      $('plcStatus').textContent = data.message;
    }
  } catch (error) {
    if (action !== 'preview') {
      plcLastStatus = null;
      $('plcConnection').textContent = 'PLC 状态待核对';
    }
    $('plcStatus').textContent = error.message;
    notice(error.message, true);
  } finally {
    busy = false; controls();
  }
}

for (const [id, action] of [['plcPreview', 'preview'], ['plcSend', 'send'], ['plcTestWrite', 'test-write'],
  ['plcRead', 'status'], ['plcRelease', 'release']]) {
  $(id).addEventListener('click', () => plcAction(action));
}
updatePLCControls();
