"""Image receipt -> segmentation -> sampling plan -> verified DB100 transfer.

The PLC owns DBX4.2. This service only withdraws PC-owned DBX4.0 after
acknowledgment, and never controls machine motion or calls a camera SDK.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import queue
import re
import threading
import time

import numpy as np
from PIL import Image

from sampling import make_plan, number, validate_calibration
from plc_transfer import prepare_plan


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


class WorkflowError(RuntimeError):
    pass


class WorkflowStopped(WorkflowError):
    pass


TERMINAL = {'completed', 'failed', 'cancelled', 'interrupted'}


def validate_settings(value):
    if not isinstance(value, dict):
        raise ValueError('自动处理参数必须为对象。')
    width, height = value.get('image_width'), value.get('image_height')
    if any(type(v) is not int or not 1 <= v <= 25000000 for v in (width, height)) or width*height > 25000000:
        raise ValueError('请先上传参考照片并确认图片尺寸。')
    calibration = validate_calibration(value.get('calibration'), width, height)
    mode = value.get('transfer_mode', 'execute')
    if mode not in ('execute', 'coordinates_only'):
        raise ValueError('请选择“坐标覆盖测试”或“联动执行”。')
    return {'image_width': width, 'image_height': height, 'calibration': calibration,
            'transfer_mode': mode,
            'diameter_mm': number(value.get('diameter_mm'), '取样圆直径', positive=True),
            'gap_mm': number(value.get('gap_mm'), '圆边缘间隙'),
            'default_depth_mm': number(value.get('default_depth_mm'), '统一深度', maximum=1000),
            'plc_timeout_s': number(value.get('plc_timeout_s', 60), 'PLC 等待时间', minimum=1, maximum=600)}


class AutomaticWorkflow:
    def __init__(self, predictor, transfer, state_dir):
        self.predictor, self.transfer = predictor, transfer
        self.root = Path(state_dir)
        for name in ('jobs', 'images'):
            (self.root/name).mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.pending = queue.Queue(maxsize=8)
        self.wake = threading.Event()
        self.closed = False
        self.enabled = False  # Explicitly start each program session; settings persist.
        self.active_id = None
        self.settings = None
        self.jobs = {}
        settings_path = self.root/'settings.json'
        if settings_path.is_file():
            self.settings = validate_settings(json.loads(settings_path.read_text(encoding='utf-8')))
        for path in sorted((self.root/'jobs').glob('*.json')):
            job = json.loads(path.read_text(encoding='utf-8'))
            if job['state'] not in TERMINAL:
                job.update(state='interrupted', updated_at=timestamp(),
                           message='上次程序在处理期间退出；本次不会自动重发此图片，请先核对 PLC 状态。')
                atomic_json(path, job)
            self.jobs[job['job_id']] = job
        self.thread = threading.Thread(target=self._worker, name='automatic-image-workflow', daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.lock:
            recent = sorted(self.jobs.values(), key=lambda j: j['created_at'], reverse=True)[:10]
            return {'enabled': self.enabled, 'configured': self.settings is not None,
                    'settings': copy.deepcopy(self.settings), 'active_job_id': self.active_id,
                    'queued': self.pending.qsize(), 'jobs': [self._public(j, details=False) for j in recent]}

    @staticmethod
    def _public(job, details=True):
        omitted = {'settings'} if details else {'settings', 'meta', 'plan'}
        return copy.deepcopy({**{key: val for key, val in job.items() if key not in omitted},
                              **({'applied_settings': job['settings']} if details else {}),
                              'has_result': 'meta' in job, 'has_plan': 'plan' in job,
                              'count': job.get('plan', {}).get('count')})

    def get_job(self, identifier):
        with self.lock:
            if identifier not in self.jobs:
                raise WorkflowError('未找到这张图片的处理记录。')
            return self._public(self.jobs[identifier])

    def start(self, settings=None):
        validated = validate_settings(settings) if settings is not None else None
        with self.lock:
            if self.closed:
                raise WorkflowError('自动处理服务已关闭。')
            if self.active_id or not self.pending.empty():
                raise WorkflowError('当前图片尚未结束，请等待后再修改自动参数。')
            if validated is not None:
                atomic_json(self.root/'settings.json', validated)
                self.settings = validated
            if self.settings is None:
                raise WorkflowError('请先用参考照片确认标定和取样参数。')
            self.wake.clear()
            self.enabled = True
        return self.snapshot()

    def _set(self, job, state, message, **updates):
        with self.lock:
            job.update(state=state, message=message, updated_at=timestamp(), **updates)
            atomic_json(self.root/'jobs'/f"{job['job_id']}.json", job)

    def _cancel_pending(self):
        while True:
            try:
                job_id = self.pending.get_nowait()
            except queue.Empty:
                return
            self._set(self.jobs[job_id], 'cancelled', '自动处理已停止，此图片未发送至 PLC。')
            self.pending.task_done()

    def stop(self):
        with self.lock:
            self.enabled = False
            self.wake.set()
            self._cancel_pending()
        return self.snapshot()

    def submit(self, raw, filename, capture_id):
        if not isinstance(capture_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,96}', capture_id):
            raise ValueError('每次拍摄需提供独立的 X-Capture-ID（字母、数字、下划线或短横线，最多 96 字符）。')
        if not raw or len(raw) > 40*1024*1024:
            raise ValueError('图片应为 40 MB 以内的 JPG 或 PNG。')
        job_id = hashlib.sha256(capture_id.encode()).hexdigest()[:32]
        digest = hashlib.sha256(raw).hexdigest()
        with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                if job['image_sha256'] != digest or job['capture_id'] != capture_id:
                    raise WorkflowError('同一拍摄编号对应了不同图片，请检查相机接图程序。')
                return {**self._public(job), 'duplicate': True}
            if not self.enabled or self.closed:
                raise WorkflowError('自动处理未启动；图片未进入下发流程。')
            if self.pending.full():
                raise WorkflowError('待处理队列已满，图片未接收；请保留拍摄编号，稍后再提交。')
            job = {'job_id': job_id, 'capture_id': capture_id, 'filename': filename,
                   'image_sha256': digest, 'state': 'queued', 'message': '图片已接收，等待自动处理。',
                   'created_at': timestamp(), 'updated_at': timestamp(),
                   'settings': copy.deepcopy(self.settings)}
            (self.root/'images'/f'{job_id}.image').write_bytes(raw)
            atomic_json(self.root/'jobs'/f'{job_id}.json', job)
            self.jobs[job_id] = job
            self.pending.put_nowait(job_id)
            return self._public(job)

    def _check_enabled(self):
        if not self.enabled or self.closed:
            raise WorkflowStopped('自动处理已停止。')

    def _wait_plc(self, job, predicate, state, message, initial=None):
        self._set(job, state, message)
        deadline = time.monotonic() + job['settings']['plc_timeout_s']
        last_error = None
        status = initial
        while True:
            self._check_enabled()
            if status is not None and predicate(status):
                self._set(job, state, message, plc_status=status)
                return status
            if time.monotonic() >= deadline:
                detail = f'；最近一次读取错误：{last_error}' if last_error else ''
                raise WorkflowError(message + '超时' + detail + '。自动处理已停止，请核对 PLC 状态。')
            try:
                status = self.transfer.status()
                last_error = None
                if status != job.get('plc_status'):
                    self._set(job, state, message, plc_status=status)
            except Exception as exc:
                status, last_error = None, str(exc)
            if status is not None and predicate(status):
                return status
            self.wake.wait(0.25)

    def _process(self, job):
        settings = job['settings']
        self._check_enabled()
        self._set(job, 'recognizing', '正在自动识别菌丝区域。')
        raw = (self.root/'images'/f"{job['job_id']}.image").read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in ('JPEG', 'PNG'):
                raise ValueError('只接受 JPG 或 PNG 图片。')
            if image.size != (settings['image_width'], settings['image_height']):
                raise ValueError('新照片尺寸与保存的标定不一致；未下发坐标，请重新确认标定。')
        if not self.predictor.slots.acquire(timeout=10):
            raise WorkflowError('识别服务繁忙，本次未下发坐标。')
        locked = False
        try:
            locked = self.predictor.lock.acquire(timeout=60)
            if not locked:
                raise WorkflowError('等待识别服务超时，本次未下发坐标。')
            self._check_enabled()
            meta = self.predictor.predict(raw, job['filename'])
            self._set(job, 'planning', '正在按固定参数划圈并生成取样坐标。', meta=meta)
            folder = self.predictor.result_root/meta['id']
            with Image.open(folder/'mask.png') as image:
                mask = np.asarray(image)
            plan = make_plan(mask, settings['calibration'], settings['diameter_mm'],
                             settings['gap_mm'], settings['default_depth_mm'])
            plan.update(prediction_id=meta['id'], source_name=meta['name'])
            destination = folder/'plans'/plan['plan_id']
            destination.mkdir(parents=True, exist_ok=False)
            atomic_json(destination/'plan.json', plan)
            self._set(job, 'planning', f"已生成 {plan['count']} 个取样点。", plan=plan)
            prepare_plan(plan)  # Reject zero/over-capacity/invalid values before PLC I/O.
        finally:
            if locked:
                self.predictor.lock.release()
            self.predictor.slots.release()
        if settings['transfer_mode'] == 'coordinates_only':
            self._check_enabled()
            self._set(job, 'writing_coordinates', '正在覆盖并核对坐标，仅测试数据，不发送联动通知。')
            result = self.transfer.write_coordinates(plan)
            self._set(job, 'completed', result['message'] + ' 可提交下一张照片覆盖。',
                      transfer_result=result, plc_status=result['status'])
            return
        self._wait_plc(job, lambda s: not s['pc_ready'] and not s['plc_taken'],
                       'waiting_idle', '等待 PLC 交接位回到初始状态。')
        self._check_enabled()
        self._set(job, 'sending', '正在写入并核对 DB100 坐标和数量。')
        result = self.transfer.send(plan)  # Never retry this operation automatically.
        self._set(job, 'waiting_taken', '坐标已下发，等待 PLC 取数反馈。', transfer_result=result)
        self._wait_plc(job, lambda s: s['plc_taken'], 'waiting_taken',
                       '等待 PLC 取完本批点位。', initial=result.get('status'))
        self._check_enabled()
        self._set(job, 'releasing', 'PLC 已取完点位，正在收回 PC 完成信号。')
        released = self.transfer.release()  # Only PC-owned DBX4.0 is modified.
        self._set(job, 'waiting_reset', '等待 PLC 自行复位取数反馈。', release_result=released)
        self._wait_plc(job, lambda s: not s['pc_ready'] and not s['plc_taken'],
                       'waiting_reset', '等待 PLC 自行复位 DBX4.2。')
        self._set(job, 'completed', f"本批 {plan['count']} 个点已交接完成，可接收下一张照片。")

    def _worker(self):
        while not self.closed:
            try:
                identifier = self.pending.get(timeout=0.2)
            except queue.Empty:
                continue
            with self.lock:
                self.active_id = identifier
                job = self.jobs[identifier]
            try:
                self._process(job)
            except WorkflowStopped:
                notified = job.get('transfer_result', {}).get('notification') == 'sent'
                self._set(job, 'cancelled', '自动处理已停止。' +
                          ('本批已发送到 PLC，停止监视不会撤回数据；继续前请核对握手状态。' if notified
                           else '请核对处理记录；已经开始的底层传输不会被强行中断。'))
            except Exception as exc:
                with self.lock:
                    self.enabled = False
                    self.wake.set()
                    self._set(job, 'failed', str(exc),
                              notification=getattr(exc, 'notification',
                                  job.get('transfer_result', {}).get('notification', 'not_sent')))
                    self._cancel_pending()
            finally:
                with self.lock:
                    self.active_id = None
                self.pending.task_done()

    def close(self):
        self.stop()
        self.closed = True
        self.wake.set()
        self.thread.join(timeout=3)
