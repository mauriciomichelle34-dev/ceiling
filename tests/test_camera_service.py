import ctypes
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PIL import Image
from camera_service import CameraConfig,CameraService,CameraError,MVSAdapter


class FakeAdapter:
    version='simulated-sdk'
    def __init__(self,config):
        self.shots=0; self.connected=False
        self.device={'id':'SIM-1','model':'模拟相机','serial':'SIM-1','ip':'127.0.0.1','interface':'test'}
    def enumerate(self): return [dict(self.device)]
    def connect(self,identifier):
        if identifier!='SIM-1': raise CameraError('不存在')
        self.connected=True; return dict(self.device)
    def disconnect(self): self.connected=False
    def close(self): self.disconnect()
    def capture(self):
        if not self.connected: raise CameraError('未连接')
        self.shots+=1
        image=Image.new('L',(240,240),120); data=io.BytesIO(); image.save(data,format='PNG')
        return data.getvalue(),{'width':240,'height':240,'frame_number':self.shots,'pixel_type':17301505,'monochrome':True}


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.service=CameraService(CameraConfig(),Path(self.temp.name),FakeAdapter)
    def tearDown(self): self.service.close(); self.temp.cleanup()
    def connect(self): self.service.search(); self.service.connect('SIM-1')
    def test_lifecycle_and_lazy_sdk_loading(self):
        self.assertFalse(self.service.status()['sdk_loaded'])
        self.assertEqual(len(self.service.search()['devices']),1)
        self.assertTrue(self.service.connect('SIM-1')['connected'])
        self.assertFalse(self.service.disconnect()['connected'])
    def test_capture_requires_connection(self):
        with self.assertRaises(CameraError): self.service.capture('one','analysis',lambda *a:{})
        self.assertIsNone(self.service.adapter)
    def test_capture_is_saved_and_same_request_does_not_trigger_again(self):
        self.connect(); calls=[]
        def process(*args): calls.append(args); return {'kind':'analysis'}
        first=self.service.capture('frame-1','analysis',process)
        second=self.service.capture('frame-1','analysis',process)
        self.assertEqual(first,second)
        self.assertEqual(self.service.adapter.shots,1)
        self.assertEqual(len(calls),1)
        self.assertTrue(self.service.image_path(first['id']).is_file())
        self.service.capture('frame-2','analysis',process)
        self.assertEqual(self.service.adapter.shots,2)
    def test_existing_capture_cannot_be_reused_as_an_automatic_request(self):
        self.connect(); self.service.capture('one','analysis',lambda *a:{})
        with self.assertRaises(CameraError): self.service.capture('one','automatic:new',lambda *a:{})
        self.assertEqual(self.service.adapter.shots,1)
    def test_recognition_error_keeps_raw_photo_and_is_not_retried(self):
        self.connect()
        def process(*args): raise ValueError('识别失败')
        result=self.service.capture('one','analysis',process)
        self.assertIn('识别失败',result['processing_error'])
        self.assertTrue(self.service.image_path(result['id']).is_file())
        self.service.capture('one','analysis',process)
        self.assertEqual(self.service.adapter.shots,1)
    def test_bad_ids_and_concurrent_actions_rejected(self):
        for value in (None,'../x',''):
            with self.assertRaises(ValueError): self.service.capture(value,'analysis',lambda *a:{})
        self.service.operation.acquire()
        try:
            self.assertTrue(self.service.status()['busy'])
            with self.assertRaises(CameraError): self.service.search()
        finally: self.service.operation.release()
    def test_missing_runtime_does_not_prevent_service_startup(self):
        self.service.close()
        self.service=CameraService(CameraConfig(sdk_python_dir='Z:/not-present',runtime_dir='Z:/not-present'),self.temp.name)
        self.assertFalse(self.service.status()['sdk_loaded'])
        with self.assertRaises(CameraError): self.service.search()


class NativeCamera:
    def __init__(self):
        self.values={'TriggerSelector':2,'AcquisitionMode':0,'TriggerMode':1,'TriggerSource':0}
        self.events=[]; self.timeout=False; self.selector_failure=False; self.convert_failure=False
        self.pixel=17301505
        self.selectors={0:'FrameStart',2:'AcquisitionStart'}
        self.buffer=(ctypes.c_ubyte*12)(*range(12))
    def MV_CC_GetEnumValue(self,name,value):
        value.nCurValue=self.values[name]
        if name=='TriggerSelector':
            value.nSupportValue=list(self.selectors); value.nSupportedNum=len(self.selectors)
        return 0
    def MV_CC_GetEnumEntrySymbolic(self,name,entry):
        entry.chSymbolic=self.selectors[entry.nValue].encode(); return 0
    def MV_CC_SetEnumValueByString(self,name,value):
        self.events.append(('set-string',name,value))
        if self.selector_failure and name=='TriggerSelector': return 0x80000004
        self.values[name]={'FrameStart':0,'FrameBurstStart':6,'Continuous':2,'Off':0,'On':1,'Software':7}[value]; return 0
    def MV_CC_SetEnumValue(self,name,value): self.events.append(('restore',name,value)); self.values[name]=value; return 0
    def MV_CC_StartGrabbing(self): self.events.append(('start',)); return 0
    def MV_CC_ClearImageBuffer(self): self.events.append(('clear',)); return 0
    def MV_CC_SetCommandValue(self,name): self.events.append(('command',name)); return 0
    def MV_CC_GetImageBuffer(self,frame,timeout):
        self.events.append(('get-frame',timeout))
        if self.timeout: return 0x80000007
        frame.stFrameInfo=SimpleNamespace(nWidth=2,nHeight=2,nExtendWidth=0,nExtendHeight=0,
            enPixelType=self.pixel,nFrameNum=12,nFrameLen=12,nFrameLenEx=0,nLostPacket=0)
        frame.pBufAddr=ctypes.cast(self.buffer,ctypes.POINTER(ctypes.c_ubyte)); return 0
    def MV_CC_FreeImageBuffer(self,frame): self.events.append(('free',)); return 0
    def MV_CC_StopGrabbing(self): self.events.append(('stop',)); return 0
    def MV_CC_ConvertPixelTypeEx(self,parameters):
        self.events.append(('convert',parameters.enDstPixelType))
        if self.convert_failure: return 0x80000004
        data=b'\x01\x02\x03\x04'
        ctypes.memmove(parameters.pDstBuffer,data,len(data)); parameters.nDstLen=len(data); return 0


class NativeSequenceTests(unittest.TestCase):
    def setUp(self):
        self.native=NativeCamera()
        self.adapter=MVSAdapter.__new__(MVSAdapter)
        self.adapter.camera=self.native; self.adapter.config=CameraConfig()
        self.adapter.sdk=SimpleNamespace(MVCC_ENUMVALUE=lambda:SimpleNamespace(nCurValue=0),
            MVCC_ENUMENTRY=lambda:SimpleNamespace(),
            MV_FRAME_OUT=lambda:SimpleNamespace(),MV_CC_PIXEL_CONVERT_PARAM_EX=lambda:SimpleNamespace(),
            PixelType_Gvsp_Mono8=17301505,PixelType_Gvsp_Mono12=17825797,PixelType_Gvsp_RGB8_Packed=35127316)
    def test_software_trigger_gets_fresh_frame_and_restores_configuration(self):
        before=dict(self.native.values)
        raw,meta=self.adapter.capture()
        with Image.open(io.BytesIO(raw)) as image:
            self.assertEqual(image.size,(2,2)); self.assertEqual(image.mode,'L')
        self.assertEqual(self.native.values,before)
        self.assertLess(self.native.events.index(('clear',)),self.native.events.index(('command','TriggerSoftware')))
        self.assertIn(('free',),self.native.events); self.assertIn(('stop',),self.native.events)
    def test_timeout_stops_grabbing_restores_state_and_does_not_free_nonexistent_frame(self):
        self.native.timeout=True; before=dict(self.native.values)
        with self.assertRaises(CameraError): self.adapter.capture()
        self.assertEqual(self.native.values,before)
        self.assertNotIn(('free',),self.native.events)
        self.assertIn(('stop',),self.native.events)
    def test_failed_selector_change_does_not_disable_original_trigger(self):
        self.native.selector_failure=True; before=dict(self.native.values)
        with self.assertRaises(CameraError): self.adapter.capture()
        self.assertEqual(self.native.values,before)
        self.assertFalse(any(e[:2]==('restore','TriggerMode') for e in self.native.events))
    def test_cs_camera_with_only_frame_burst_start_can_capture_and_restore(self):
        self.native.selectors={6:'FrameBurstStart'}
        self.native.values['TriggerSelector']=6
        before=dict(self.native.values)
        raw,meta=self.adapter.capture()
        self.assertIn(('set-string','TriggerSelector','FrameBurstStart'),self.native.events)
        self.assertEqual(self.native.values,before)
        self.assertEqual(meta['frame_number'],12)
    def test_unsupported_trigger_selector_does_not_change_trigger_settings(self):
        self.native.selectors={2:'AcquisitionStart'}
        before=dict(self.native.values)
        with self.assertRaisesRegex(CameraError,'没有提供'):
            self.adapter.capture()
        self.assertEqual(self.native.values,before)
        self.assertFalse(any(e[0]=='set-string' for e in self.native.events))
    def test_mono12_conversion_and_error_cleanup(self):
        self.native.pixel=17825797
        raw,_=self.adapter.capture()
        with Image.open(io.BytesIO(raw)) as image: self.assertEqual(image.tobytes(),b'\x01\x02\x03\x04')
        self.native.convert_failure=True
        with self.assertRaises(CameraError): self.adapter.capture()
        self.assertEqual(self.native.events.count(('free',)),2)


if __name__=='__main__': unittest.main(verbosity=2)
