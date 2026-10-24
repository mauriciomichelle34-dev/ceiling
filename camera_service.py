"""MVS software-trigger camera control, independent of machine/PLC motion."""
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import sys
import threading

from PIL import Image


class CameraError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraConfig:
    sdk_python_dir: str = 'E:/haikangMVS/MVS/Development/Samples/Python/MvImport'
    runtime_dir: str = 'C:/Program Files (x86)/Common Files/MVS/Runtime/Win64_x64'
    capture_timeout_ms: int = 10000

    def __post_init__(self):
        if type(self.capture_timeout_ms) is not int or not 100 <= self.capture_timeout_ms <= 30000:
            raise ValueError('相机等待时间应为 100–30000 毫秒。')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8-sig')))


def check(code, operation):
    if code:
        raise CameraError(f'{operation}失败，MVS 错误码 0x{code & 0xffffffff:08X}。')


class MVSAdapter:
    def __init__(self, config):
        self.config = config
        runtime, wrappers = Path(config.runtime_dir), Path(config.sdk_python_dir)
        if os.name != 'nt' or ctypes.sizeof(ctypes.c_void_p) != 8:
            raise CameraError('当前相机接口需要 64 位 Windows Python。')
        if not (runtime/'MvCameraControl.dll').is_file() or not (wrappers/'MvCameraControl_class.py').is_file():
            raise CameraError('未找到 MVS 的 64 位运行库或 Python 接口，请核对 camera_config.json。')
        self.dll_directory = os.add_dll_directory(str(runtime))
        # The vendor wrapper loads by basename; prepend the trusted runtime for this process only.
        os.environ['PATH'] = str(runtime) + os.pathsep + os.environ.get('PATH', '')
        sys.path.insert(0, str(wrappers))
        try:
            self.sdk = importlib.import_module('MvCameraControl_class')
            check(self.sdk.MvCamera.MV_CC_Initialize(), '初始化相机 SDK')
        except Exception as exc:
            self.dll_directory.close()
            if isinstance(exc, CameraError):
                raise
            raise CameraError(f'MVS SDK 加载失败：{exc}') from exc
        version = self.sdk.MvCamera.MV_CC_GetSDKVersion()
        self.version = '.'.join(str(version >> shift & 255) for shift in (24,16,8,0))
        self.camera = None
        self.device = None
        self.devices = {}

    @staticmethod
    def _text(value):
        return bytes(value).split(b'\0', 1)[0].decode('utf-8', errors='replace')

    @staticmethod
    def _ip(value):
        return '.'.join(str(value >> shift & 255) for shift in (24,16,8,0))

    def enumerate(self):
        sdk = self.sdk
        listing = sdk.MV_CC_DEVICE_INFO_LIST()
        check(sdk.MvCamera.MV_CC_EnumDevices(sdk.MV_GIGE_DEVICE, listing), '搜索相机')
        devices, native = [], {}
        for index in range(listing.nDeviceNum):
            pointer = ctypes.cast(listing.pDeviceInfo[index], ctypes.POINTER(sdk.MV_CC_DEVICE_INFO))
            info = sdk.MV_CC_DEVICE_INFO.from_buffer_copy(ctypes.string_at(pointer, ctypes.sizeof(sdk.MV_CC_DEVICE_INFO)))
            gige = info.SpecialInfo.stGigEInfo
            serial = self._text(gige.chSerialNumber)
            model = self._text(gige.chModelName)
            identifier = serial or f'{model}-{self._ip(gige.nCurrentIp)}'
            entry = {'id':identifier, 'model':model, 'serial':serial,
                     'ip':self._ip(gige.nCurrentIp), 'interface':'GigE'}
            devices.append(entry)
            native[identifier] = (info, entry)
        self.devices = native
        return devices

    def connect(self, identifier):
        if self.camera is not None:
            if self.device['id'] == identifier:
                return self.device
            raise CameraError('请先断开当前相机，再连接另一台。')
        self.enumerate()
        if identifier not in self.devices:
            raise CameraError('搜索结果中没有这台相机，请重新搜索。')
        info, device = self.devices[identifier]
        camera = self.sdk.MvCamera()
        check(camera.MV_CC_CreateHandle(info), '创建相机句柄')
        try:
            # Do not take over a device already controlled by the MVS client.
            check(camera.MV_CC_OpenDevice(self.sdk.MV_ACCESS_Exclusive, 0),
                  '连接相机（如果已被 MVS 客户端占用，请先在客户端断开）')
        except Exception:
            camera.MV_CC_DestroyHandle()
            raise
        self.camera, self.device = camera, dict(device)
        return self.device

    def _get_enum(self, name, optional=False):
        value = self.sdk.MVCC_ENUMVALUE()
        code = self.camera.MV_CC_GetEnumValue(name, value)
        if code and optional:
            return None
        check(code, '读取 ' + name)
        return value.nCurValue

    def _frame_trigger(self):
        value = self.sdk.MVCC_ENUMVALUE()
        code = self.camera.MV_CC_GetEnumValue('TriggerSelector', value)
        if code:
            check(code, '读取 TriggerSelector')
        symbols = set()
        for number in list(value.nSupportValue)[:value.nSupportedNum]:
            entry = self.sdk.MVCC_ENUMENTRY()
            entry.nValue = number
            check(self.camera.MV_CC_GetEnumEntrySymbolic('TriggerSelector', entry), '读取支持的触发方式')
            symbols.add(self._text(entry.chSymbolic))
        for name in ('FrameStart', 'FrameBurstStart'):
            if name in symbols:
                return name
        raise CameraError('这台相机没有提供 FrameStart 或 FrameBurstStart 触发方式，未修改触发配置。')

    def _frame_to_png(self, frame):
        sdk, camera = self.sdk, self.camera
        info = frame.stFrameInfo
        width = int(info.nExtendWidth or info.nWidth)
        height = int(info.nExtendHeight or info.nHeight)
        if not 0 < width*height <= 25000000 or not frame.pBufAddr:
            raise CameraError('相机返回的图片尺寸无效或超过当前识别程序的 2500 万像素限制。')
        source_length = int(info.nFrameLenEx or info.nFrameLen)
        if info.nLostPacket:
            raise CameraError(f'这帧图片存在丢包（{info.nLostPacket}），未送入识别。')
        mono = {value for name,value in vars(sdk).items() if name.startswith('PixelType_Gvsp_Mono')}
        is_mono = info.enPixelType in mono
        channels = 1 if is_mono else 3
        needed = width*height*channels
        if info.enPixelType == sdk.PixelType_Gvsp_Mono8:
            if source_length < needed:
                raise CameraError('相机帧长度不足。')
            raw = ctypes.string_at(frame.pBufAddr, needed)
        else:
            output = (ctypes.c_ubyte * needed)()
            parameters = sdk.MV_CC_PIXEL_CONVERT_PARAM_EX()
            parameters.nWidth, parameters.nHeight = width, height
            parameters.enSrcPixelType = info.enPixelType
            parameters.pSrcData = frame.pBufAddr
            parameters.nSrcDataLen = source_length
            parameters.enDstPixelType = sdk.PixelType_Gvsp_Mono8 if is_mono else sdk.PixelType_Gvsp_RGB8_Packed
            parameters.pDstBuffer = ctypes.cast(output, ctypes.POINTER(ctypes.c_ubyte))
            parameters.nDstBufferSize = needed
            check(camera.MV_CC_ConvertPixelTypeEx(parameters), '转换相机像素格式')
            if parameters.nDstLen != needed:
                raise CameraError('转换后的图片长度不一致。')
            raw = bytes(output)
        image = Image.frombytes('L' if is_mono else 'RGB', (width,height), raw)
        encoded = io.BytesIO(); image.save(encoded, format='PNG')
        return encoded.getvalue(), {'width':width, 'height':height, 'frame_number':int(info.nFrameNum),
                                    'pixel_type':int(info.enPixelType), 'monochrome':is_mono}

    def capture(self):
        if self.camera is None:
            raise CameraError('请先搜索并连接相机。')
        camera, sdk = self.camera, self.sdk
        selector = self._get_enum('TriggerSelector', optional=True)
        acquisition = self._get_enum('AcquisitionMode')
        trigger_mode = trigger_source = None
        grabbing = False
        frame = sdk.MV_FRAME_OUT()
        has_frame = False
        try:
            if selector is not None:
                # Some CS cameras only expose FrameBurstStart. Query the actual device XML.
                check(camera.MV_CC_SetEnumValueByString('TriggerSelector',self._frame_trigger()), '选择拍照触发')
            trigger_mode = self._get_enum('TriggerMode')
            trigger_source = self._get_enum('TriggerSource')
            check(camera.MV_CC_SetEnumValueByString('TriggerMode','Off'), '配置触发模式')
            check(camera.MV_CC_SetEnumValueByString('AcquisitionMode','Continuous'), '配置采集模式')
            check(camera.MV_CC_SetEnumValueByString('TriggerSource','Software'), '配置软件触发源')
            check(camera.MV_CC_SetEnumValueByString('TriggerMode','On'), '启用软件触发')
            check(camera.MV_CC_StartGrabbing(), '启动相机采集')
            grabbing = True
            check(camera.MV_CC_ClearImageBuffer(), '清除旧图像缓存')
            check(camera.MV_CC_SetCommandValue('TriggerSoftware'), '软件拍照')
            code = camera.MV_CC_GetImageBuffer(frame, self.config.capture_timeout_ms)
            if code:
                raise CameraError(f'等待照片失败或超时，MVS 错误码 0x{code & 0xffffffff:08X}。')
            has_frame = True
            return self._frame_to_png(frame)
        finally:
            original_error = sys.exc_info()[0] is not None
            cleanup = []
            if has_frame:
                code = camera.MV_CC_FreeImageBuffer(frame)
                if code: cleanup.append(f'释放帧缓存 0x{code:08X}')
            if grabbing:
                code = camera.MV_CC_StopGrabbing()
                if code: cleanup.append(f'停止采集 0x{code:08X}')
            # Restore temporary software-trigger changes; never save a camera UserSet.
            restore = ([('TriggerMode',0), ('TriggerSource',trigger_source), ('TriggerMode',trigger_mode)]
                       if trigger_mode is not None else [])
            restore += [('TriggerSelector',selector), ('AcquisitionMode',acquisition)]
            for name,value in restore:
                if value is not None:
                    code = camera.MV_CC_SetEnumValue(name,value)
                    if code: cleanup.append(f'恢复 {name} 0x{code:08X}')
            if cleanup and not original_error:
                raise CameraError('拍照结束但清理或恢复参数失败，请重新连接相机：' + '；'.join(cleanup))

    def disconnect(self):
        if self.camera is not None:
            camera = self.camera
            self.camera = self.device = None
            try:
                check(camera.MV_CC_CloseDevice(), '断开相机')
            finally:
                camera.MV_CC_DestroyHandle()

    def close(self):
        try:
            self.disconnect()
        finally:
            self.sdk.MvCamera.MV_CC_Finalize()
            self.dll_directory.close()


class CameraService:
    def __init__(self, config, capture_root, adapter_factory=MVSAdapter):
        self.config, self.root, self.factory = config, Path(capture_root), adapter_factory
        self.root.mkdir(parents=True,exist_ok=True)
        self.adapter = None
        self.operation = threading.Lock()
        self.state_lock = threading.RLock()
        self.device = None
        self.devices = []
        self.last_error = None
        self.last_capture = None

    @contextmanager
    def _operation(self):
        if not self.operation.acquire(blocking=False):
            raise CameraError('相机正在处理上一项操作，请稍候。')
        try:
            yield
            self.last_error = None
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            self.operation.release()

    def _adapter(self):
        if self.adapter is None:
            self.adapter = self.factory(self.config)
        return self.adapter

    def status(self):
        device, adapter = self.device, self.adapter
        return {'sdk_loaded':self.adapter is not None,
                'sdk_version':adapter.version if adapter else None,
                'connected':device is not None, 'device':dict(device) if device else None,
                'devices':[dict(d) for d in self.devices], 'busy':self.operation.locked(),
                'last_error':self.last_error, 'last_capture':self.last_capture}

    def search(self):
        with self._operation():
            self.devices = self._adapter().enumerate()
        return self.status()

    def connect(self, identifier):
        if not isinstance(identifier,str) or not identifier or len(identifier)>128:
            raise ValueError('请先选择搜索到的相机。')
        with self._operation():
            self.device = self._adapter().connect(identifier)
        return self.status()

    def disconnect(self):
        with self._operation():
            try:
                if self.adapter:
                    self.adapter.disconnect()
            finally:
                self.device = None
        return self.status()

    @staticmethod
    def _save_json(path, record):
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(record,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        temp.replace(path)

    def capture(self, capture_id, route, process):
        if not isinstance(capture_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,96}',capture_id):
            raise ValueError('拍照请求需要有效的唯一编号。')
        key = hashlib.sha256(capture_id.encode()).hexdigest()[:32]
        record_path = self.root/f'{key}.json'
        image_path = self.root/f'{key}.png'
        with self._operation():
            if record_path.is_file():
                record = json.loads(record_path.read_text(encoding='utf-8'))
                if record['capture_id']!=capture_id or record['route']!=route:
                    raise CameraError('该拍照请求已在其他处理模式下执行，请重新点击拍照。')
            else:
                if self.device is None:
                    raise CameraError('请先搜索并连接相机。')
                raw, details = self._adapter().capture()
                with Image.open(io.BytesIO(raw)) as image:
                    if image.format!='PNG' or image.size!=(details['width'],details['height']):
                        raise CameraError('SDK 返回的照片编码或尺寸不一致。')
                image_path.write_bytes(raw)
                record = {'id':key,'capture_id':capture_id,'route':route,
                          'filename':f'camera_{key}.png','image_url':f'/camera/captures/{key}.png',
                          'created_at':datetime.now(timezone.utc).isoformat(),
                          'device':dict(self.device), **details}
                self._save_json(record_path,record)
            if 'processing' not in record and 'processing_error' not in record:
                try:
                    record['processing'] = process(image_path.read_bytes(),record['filename'],capture_id)
                except Exception as exc:
                    record['processing_error'] = f'已拍到照片，但后续处理失败：{exc}'
                self._save_json(record_path,record)
            self.last_capture = record
            return record

    def image_path(self, identifier):
        if not isinstance(identifier,str) or not re.fullmatch(r'[0-9a-f]{32}',identifier):
            raise ValueError('图片编号无效。')
        return self.root/f'{identifier}.png'

    def close(self):
        with self.operation:
            if self.adapter:
                self.adapter.close()
                self.adapter = self.device = None
