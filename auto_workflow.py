"""Image receipt -> segmentation -> sampling plan -> verified DB100 transfer.

The PLC owns DBX4.2. In PLC-camera mode it signals photo position and the PLC
resets both handshake bits after executing the batch. CameraService handles SDK I/O.
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
import uuid

import numpy as np
from PIL import Image

from sampling import make_plan, number, validate_calibration
from plc_transfer import prepare_plan
from fixed_origin import apply_origin, validate_origin, automatic_calibration, DISH_DIAMETER_MM
from image_limits import MAX_IMAGE_BYTES, IMAGE_FORMATS


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
    calibration = validate_calibration(value.get('calibration'), width, height, require_origin=True)
    mode = value.get('transfer_mode', 'execute')
    if mode not in ('execute', 'coordinates_only'):
        raise ValueError('请选择“坐标覆盖测试”或“联动执行”。')
    source = value.get('input_source', 'received_images')
    if source not in ('received_images', 'plc_camera'):
        raise ValueError('请选择接收图片或 PLC 到位自动拍照。')
    if source == 'plc_camera' and mode != 'execute':
        raise ValueError('PLC 到位自动拍照需要使用“联动执行”，完成后由 PLC 复位两位信号。')
    return {'image_width': width, 'image_height': height, 'calibration': calibration,
            'transfer_mode': mode,
            'input_source': source,
            'cycle_timeout_s': number(value.get('cycle_timeout_s', 1800), '设备循环等待时间', minimum=1, maximum=86400),
            'diameter_mm': number(value.get('diameter_mm'), '取样圆直径', positive=True),
            'gap_mm': number(value.get('gap_mm'), '圆边缘间隙'),
            'default_depth_mm': number(value.get('default_depth_mm', 0), '预览深度', maximum=1000),
            'plc_timeout_s': number(value.get('plc_timeout_s', 60), 'PLC 等待时间', minimum=1, maximum=600)}


class AutomaticWorkflow:
    def __init__(self, predictor, transfer, state_dir, camera=None, fixed_origin=None):
        self.predictor, self.transfer = predictor, transfer
        self.camera = camera
        self.fixed_origin = validate_origin(fixed_origin) if fixed_origin is not None else None
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
        self.run_id = None
        self.camera_busy = False
        self.last_error = None
        self.camera_trigger = {'state':'stopped', 'message':'PLC 到位自动拍照未启动。'}
        self.jobs = {}
        settings_path = self.root/'settings.json'
        if settings_path.is_file():
            saved = json.loads(settings_path.read_text(encoding='utf-8'))
            if self.fixed_origin or saved.get('calibration', {}).get('coordinate_origin'):
                try:
                    self.settings = validate_settings(self._configured_settings(saved))
                except ValueError as exc:
                    self.last_error = str(exc)
            else:
                self.last_error = '旧自动参数使用培养皿中心，已停用。请设置螺丝原点后重新保存参数。'
        for path in sorted((self.root/'jobs').glob('*.json')):
            job = json.loads(path.read_text(encoding='utf-8'))
            if job['state'] not in TERMINAL:
                job.update(state='interrupted', updated_at=timestamp(),
                           message='上次程序在处理期间退出；本次不会自动重发此图片，请先核对 PLC 状态。')
                atomic_json(path, job)
            self.jobs[job['job_id']] = job
        self.thread = threading.Thread(target=self._worker, name='automatic-image-workflow', daemon=True)
        self.thread.start()
        self.camera_thread = None
        if camera is not None:
            self.camera_thread = threading.Thread(target=self._camera_worker, name='plc-camera-trigger', daemon=True)
            self.camera_thread.start()

    def snapshot(self):
        with self.lock:
            recent = sorted(self.jobs.values(), key=lambda j: j['created_at'], reverse=True)[:10]
            return {'enabled': self.enabled, 'configured': self.settings is not None,
                    'settings': copy.deepcopy(self.settings), 'active_job_id': self.active_id,
                    'camera_busy': self.camera_busy, 'camera_trigger': copy.deepcopy(self.camera_trigger),
                    'last_error': self.last_error,
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

    def _configured_settings(self, settings):
        if self.fixed_origin and isinstance(settings, dict):
            calibration = dict(settings.get('calibration') or {}, dish_diameter_mm=DISH_DIAMETER_MM)
            return dict(settings, calibration=apply_origin(calibration, self.fixed_origin,
                settings.get('image_width'), settings.get('image_height')))
        return settings

    def start(self, settings=None):
        validated = validate_settings(self._configured_settings(settings)) if settings is not None else None
        with self.lock:
            if self.closed:
                raise WorkflowError('自动处理服务已关闭。')
            if self.active_id or self.camera_busy or not self.pending.empty():
                raise WorkflowError('当前图片尚未结束，请等待后再修改自动参数。')
            candidate = validated if validated is not None else self.settings
            if candidate is not None and candidate['input_source'] == 'plc_camera':
                if self.camera is None or not self.camera.status()['connected']:
                    raise WorkflowError('请先在网页连接海康相机，再启动 PLC 到位自动拍照。')
                if self.camera.status()['busy']:
                    raise WorkflowError('相机正在执行其他操作，请等待拍照完成。')
                try:
                    initial_status = self.transfer.status()
                except Exception as exc:
                    raise WorkflowError(f'启动前读取 PLC 失败：{exc}') from exc
                if initial_status['pc_ready']:
                    raise WorkflowError('DBX4.0 仍为 True，请等 PLC 完成本轮并复位后再启动。')
            if validated is not None:
                atomic_json(self.root/'settings.json', validated)
                self.settings = validated
            if self.settings is None:
                raise WorkflowError('请先用参考照片确认标定和取样参数。')
            self.wake.clear()
            self.run_id = uuid.uuid4().hex
            self.last_error = None
            self.camera_trigger = {'state':'waiting_ready', 'message':'等待 PLC 到拍照位置，将 DBX4.2 置为 True。'}
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
            self.camera_trigger.update(state='stopped', message='PLC 到位自动拍照已停止。')
        return self.snapshot()

    def submit(self, raw, filename, capture_id, *, source='received_images'):
        if not isinstance(capture_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,96}', capture_id):
            raise ValueError('每次拍摄需提供独立的 X-Capture-ID（字母、数字、下划线或短横线，最多 96 字符）。')
        if not raw or len(raw) > MAX_IMAGE_BYTES:
            raise ValueError('图片应为 100 MB 以内的 JPG、PNG 或 BMP。')
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
            if (self.settings['input_source'] == 'plc_camera') != (source == 'plc_camera'):
                raise WorkflowError('PLC 到位模式只接收本轮到位触发的相机照片，请勿手动上传或拍照插入队列。')
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

    def _wait_plc(self, job, predicate, state, message, initial=None, timeout=None):
        self._set(job, state, message)
        deadline = time.monotonic() + (job['settings']['plc_timeout_s'] if timeout is None else timeout)
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
            if image.format not in IMAGE_FORMATS:
                raise ValueError('只接受 JPG、PNG 或 BMP 图片。')
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
            calibration = copy.deepcopy(settings['calibration'])
            if self.fixed_origin:
                calibration = automatic_calibration(meta, self.fixed_origin, mask.shape[1], mask.shape[0])
            elif calibration.get('coordinate_origin'):
                suggestion = meta.get('dish_suggestion')
                if not isinstance(suggestion, dict):
                    raise WorkflowError('本张照片没有培养皿定位结果，未生成或发送坐标。')
                for key in ('center_x_px', 'center_y_px'):
                    calibration[key] = suggestion[key]
            plan = make_plan(mask, calibration, settings['diameter_mm'],
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
        if settings['input_source'] == 'plc_camera':
            self._check_enabled()
            self._set(job, 'sending', '正在到位信号保持期间写入、核对全部坐标和数量。')
            result = self.transfer.send_for_camera_cycle(plan)
            self._set(job, 'waiting_cycle_finish', '数据已发完，等待 PLC 完成设备动作并复位 4.0 和 4.2。', transfer_result=result)
            self._wait_plc(job, lambda s: not s['pc_ready'] and not s['plc_taken'],
                           'waiting_cycle_finish', '等待 PLC 完成本轮并将 DBX4.0、DBX4.2 都复位为 False。',
                           initial=result.get('status'), timeout=settings['cycle_timeout_s'])
            self._set(job, 'completed', f"本轮 {plan['count']} 个点已完成交接，PLC 已复位；等待下一次到位。")
            return
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

    def _camera_worker(self):
        session, job_id, read_failures = None, None, 0
        while not self.closed:
            time.sleep(0.1)
            with self.lock:
                if not self.enabled or self.settings['input_source'] != 'plc_camera':
                    continue
                run_id = self.run_id
                if session != run_id:
                    session, job_id, read_failures = run_id, None, 0
                if job_id is not None:
                    job = self.jobs[job_id]
                    if job['state'] != 'completed':
                        continue
                    # A completed camera job has observed both bits reset. Only now re-arm.
                    job_id = None
                if self.active_id or not self.pending.empty():
                    continue
            try:
                status = self.transfer.status()
                read_failures = 0
            except Exception as exc:
                read_failures += 1
                if read_failures < 3:
                    with self.lock:
                        if self.enabled and self.run_id == run_id:
                            self.camera_trigger.update(state='reading_plc', message=f'读取 PLC 暂时失败，正在重新读取（{read_failures}/3）：{exc}')
                    continue
                self._camera_failed(run_id, f'连续三次读取 PLC 失败：{exc}')
                continue
            try:
                with self.lock:
                    if not self.enabled or run_id != self.run_id:
                        continue
                    self.camera_trigger.update(plc_status=status)
                    if status['pc_ready']:
                        raise WorkflowError('等待拍照时 DBX4.0 仍为 True，停止监听；请先核对 PLC 是否已复位。')
                    if not status['plc_taken']:
                        self.camera_trigger.update(state='waiting_ready', message='等待 PLC 到拍照位置，将 DBX4.2 置为 True。')
                        continue
                    capture_id = 'plc-' + run_id + '-' + uuid.uuid4().hex
                    self.camera_busy = True
                    self.camera_trigger.update(state='capturing', message='已收到到位信号，正在自动拍照。', capture_id=capture_id)

                def receive(raw, filename, identifier):
                    with self.lock:
                        if not self.enabled or self.run_id != run_id:
                            raise WorkflowStopped('拍照期间自动处理已停止，本次照片保留但不下发。')
                        return {'kind':'automatic', 'job':self.submit(raw, filename, identifier, source='plc_camera')}

                frame = self.camera.capture(capture_id, 'plc-camera:'+run_id, receive)
                if frame.get('processing_error'):
                    raise WorkflowError(frame['processing_error'])
                job_id = frame['processing']['job']['job_id']
                with self.lock:
                    self.camera_trigger.update(state='processing', message='本次到位已拍照，等待本轮处理及 PLC 复位，不重复拍摄。', job_id=job_id,
                                               image_url=frame['image_url'])
            except Exception as exc:
                self._camera_failed(run_id, str(exc))
            finally:
                with self.lock:
                    self.camera_busy = False

    def _camera_failed(self, run_id, message):
        with self.lock:
            if self.run_id != run_id or not self.enabled:
                return
            self.enabled = False
            self.last_error = message
            self.camera_trigger.update(state='failed', message=message)
            self.wake.set()
            self._cancel_pending()

    def close(self):
        self.stop()
        self.closed = True
        self.wake.set()
        if self.camera_thread is not None:
            self.camera_thread.join(timeout=self.camera.config.capture_timeout_ms/1000+5)
        self.thread.join(timeout=3)
