import http.client
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from image_limits import MAX_IMAGE_BYTES
from local_app import Predictor
from auto_workflow import AutomaticWorkflow
from plc_transfer import PLCTransfer
from test_plc_transfer import FakeConnection
import test_auto_workflow as aw
import test_plc_http as fixture


def bmp_bytes():
    image=Image.new('RGB',(240,240),(200,210,190))
    output=io.BytesIO();image.save(output,format='BMP')
    return output.getvalue()


class BMPDecodeTests(unittest.TestCase):
    def test_production_predictor_decodes_bmp_without_flips_or_color_channel_swap(self):
        rgb=np.zeros((240,240,3),np.uint8)
        rgb[:120,:,0]=240;rgb[120:,:,2]=180
        raw=io.BytesIO();Image.fromarray(rgb).save(raw,format='BMP')
        predictor=Predictor.__new__(Predictor)
        predictor.network=None;predictor.config={'threshold':.5};predictor.device='cpu';predictor.model_name='bmp-fixture'
        seen=[]
        def probability(network,source,config,device):
            seen.append(np.asarray(source).copy())
            return (np.asarray(source)[...,0]>200).astype(np.float32)
        boundary={'center_x_px':120,'center_y_px':120,'radius_px':100}
        suggestion={'center_x_px':120,'center_y_px':120,'dish_diameter_px':200}
        with tempfile.TemporaryDirectory() as folder:
            predictor.result_root=Path(folder)
            with patch('local_app.detect_dish',return_value=(boundary,suggestion)), \
                 patch('local_app.predict_probability',side_effect=probability), \
                 patch('local_app.detect_infrared',return_value=(np.zeros((240,240),bool),{'status':'not_detected'})):
                meta=predictor.predict(raw.getvalue(),'camera.bmp')
            self.assertTrue(np.array_equal(seen[0],rgb))
            self.assertEqual((meta['width'],meta['height']),(240,240))
            with Image.open(predictor.result_root/meta['id']/'mask.png') as image:
                mask=np.asarray(image)
            self.assertEqual(mask[60,120],1)
            self.assertEqual(mask[180,120],0)

    def test_automatic_workflow_accepts_bmp_and_completes_using_mock_plc(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            workflow=AutomaticWorkflow(aw.TestPredictor(root/'results'),
                PLCTransfer(connection_factory=lambda _:FakeConnection()),root/'state')
            try:
                workflow.start(dict(aw.settings(),transfer_mode='coordinates_only'))
                job=workflow.submit(bmp_bytes(),'camera.bmp','bmp-frame')
                result=aw.wait_until(lambda:workflow.get_job(job['job_id'])
                    if workflow.get_job(job['job_id'])['state'] in ('completed','failed') else None)
                self.assertEqual(result['state'],'completed',result)
                self.assertGreater(result['plan']['count'],0)
            finally:workflow.close()


class BMPHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.HTTPTests.setUpClass();cls.fixture=fixture.HTTPTests;cls.base=cls.fixture.base
        cls.fixture.http.predictor=aw.TestPredictor(cls.fixture.results)
        cls.fixture.http.predictor.model_name='bmp-fixture'
        cls.fixture.http.predictor.model_sha256='test'
        cls.fixture.http.predictor.config={'width':240,'height':240}
        cls.fixture.http.predictor.device=SimpleNamespace(type='cpu')
        cls.received=[]
        def submit(raw,filename,capture_id):
            cls.received.append((raw,filename,capture_id))
            return {'job_id':'b'*32}
        cls.fixture.http.automatic=SimpleNamespace(submit=submit)

    @classmethod
    def tearDownClass(cls):cls.fixture.tearDownClass()

    def test_bmp_manual_route_and_health_upload_limit(self):
        request=Request(self.base+'/api/predict',data=bmp_bytes(),headers={
            'Content-Type':'application/octet-stream','X-File-Name':'camera.bmp'})
        with urlopen(request,timeout=10) as response:meta=json.load(response)
        self.assertEqual(meta['name'],'camera.bmp')
        with urlopen(self.base+'/api/health') as response:health=json.load(response)
        self.assertEqual(health['max_upload_bytes'],100*1024*1024)
        self.assertIn('BMP',health['accepted_image_formats'])
        with urlopen(self.base+'/') as response:html=response.read().decode('utf-8')
        self.assertIn('JPG、PNG、BMP · 单张 ≤ 100 MB',html)
        self.assertIn('image/bmp,.bmp',html)

    def test_automatic_route_accepts_bmp_mime_types(self):
        for content_type in ('image/bmp','image/x-ms-bmp','application/octet-stream'):
            request=Request(self.base+'/api/automatic/images',data=bmp_bytes(),headers={
                'Content-Type':content_type,'X-File-Name':'camera.bmp',
                'X-Capture-ID':'bmp-http','X-PLC-Request':'1'})
            with urlopen(request,timeout=10) as response:self.assertEqual(response.status,202)
            self.assertEqual(self.received[-1],(bmp_bytes(),'camera.bmp','bmp-http'))

    def test_both_routes_reject_oversize_headers_before_reading_or_processing_body(self):
        from urllib.parse import urlsplit
        url=urlsplit(self.base)
        for route in ('/api/predict','/api/automatic/images'):
            client=http.client.HTTPConnection(url.hostname,url.port,timeout=5)
            try:
                client.request('POST',route,body=None,headers={'Content-Length':str(MAX_IMAGE_BYTES+1),
                    'Content-Type':'application/octet-stream','X-PLC-Request':'1'})
                response=client.getresponse()
                self.assertEqual(response.status,413)
                self.assertIn('100 MB',json.loads(response.read())['error'])
            finally:client.close()


if __name__=='__main__':unittest.main(verbosity=2)
