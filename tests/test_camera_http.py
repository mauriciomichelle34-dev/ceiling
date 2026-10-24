import json
from pathlib import Path
import sys
import unittest
from urllib.request import urlopen
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import test_plc_http as fixture
import test_auto_workflow as aw
from auto_workflow import AutomaticWorkflow
from camera_service import CameraConfig,CameraService
from test_camera_service import FakeAdapter


class CameraHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.HTTPTests.setUpClass()
        cls.http=fixture.HTTPTests.http; cls.base=fixture.HTTPTests.base
        cls.http.predictor=aw.TestPredictor(fixture.HTTPTests.results)
        cls.http.automatic=AutomaticWorkflow(cls.http.predictor,cls.http.plc_transfer,fixture.HTTPTests.results/'automatic')
        cls.http.camera=CameraService(CameraConfig(),fixture.HTTPTests.results/'camera',FakeAdapter)
    @classmethod
    def tearDownClass(cls):
        cls.http.camera.close(); cls.http.automatic.close(); fixture.HTTPTests.tearDownClass()
    def post(self,path,body,headers=None): return fixture.HTTPTests.post(self,path,body,headers)
    def test_01_search_connect_capture_analyze_and_original_image(self):
        self.assertFalse(self.post('/api/camera/status',{})[1]['connected'])
        self.assertEqual(self.post('/api/camera/search',{})[0],200)
        self.assertEqual(self.post('/api/camera/connect',{'device_id':'SIM-1'})[0],200)
        code,capture=self.post('/api/camera/capture',{'capture_id':'camera-reference'})
        self.assertEqual(code,200,capture)
        self.assertEqual(capture['processing']['kind'],'analysis')
        with urlopen(self.base+capture['image_url']) as response:
            self.assertEqual(response.headers['Content-Type'],'image/png')
        before=self.http.camera.adapter.shots
        self.assertEqual(self.post('/api/camera/capture',{'capture_id':'camera-reference'})[0],200)
        self.assertEqual(self.http.camera.adapter.shots,before)
        self.assertEqual(fixture.HTTPTests.plc.bit_writes,[])
    def test_02_shot_enters_existing_coordinate_only_pipeline(self):
        self.assertEqual(self.post('/api/automatic/start',{'settings':dict(aw.settings(),transfer_mode='coordinates_only')})[0],200)
        code,capture=self.post('/api/camera/capture',{'capture_id':'camera-automatic'})
        self.assertEqual(code,200,capture)
        job=capture['processing']['job']
        result=aw.wait_until(lambda:self.http.automatic.get_job(job['job_id'])
             if self.http.automatic.get_job(job['job_id'])['state'] in {'completed','failed'} else None)
        self.assertEqual(result['state'],'completed')
        self.assertEqual(result['transfer_result']['mode'],'coordinates_only')
        self.assertEqual(fixture.HTTPTests.plc.bit_writes,[(100,4,0,False)])
        # Previously analyzed reference is not re-used as a new automatic input.
        self.assertEqual(self.post('/api/camera/capture',{'capture_id':'camera-reference'})[0],409)
        self.assertEqual(self.post('/api/automatic/stop',{})[0],200)
    def test_03_analysis_failure_still_returns_photo(self):
        with patch.object(self.http.predictor,'predict',side_effect=ValueError('测试识别失败')):
            code,capture=self.post('/api/camera/capture',{'capture_id':'photo-only'})
        self.assertEqual(code,200,capture)
        self.assertIn('测试识别失败',capture['processing_error'])
        with urlopen(self.base+capture['image_url']) as response: self.assertEqual(response.status,200)
    def test_04_controls_require_local_json_requests(self):
        shots=self.http.camera.adapter.shots
        for headers in ({'Origin':'https://example.com'},{'Host':'example.com'},{'X-PLC-Request':''}):
            self.assertEqual(self.post('/api/camera/capture',{'capture_id':'denied'},headers)[0],403)
        self.assertEqual(self.http.camera.adapter.shots,shots)
        self.assertEqual(self.post('/api/camera/capture',{'capture_id':'../invalid'})[0],400)
    def test_05_script_and_disconnect(self):
        with urlopen(self.base+'/camera_ui.js') as response: self.assertEqual(response.status,200)
        self.assertFalse(self.post('/api/camera/disconnect',{})[1]['connected'])


if __name__=='__main__': unittest.main(verbosity=2)
