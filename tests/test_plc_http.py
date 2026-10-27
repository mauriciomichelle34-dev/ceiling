"""Exercise the planning HTTP API through a loopback S7 endpoint."""
import json
from pathlib import Path
import socket
import struct
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from http.server import ThreadingHTTPServer
from local_app import Handler
from plc_transfer import PLCConfig, PLCTransfer, prepare_plan
from s7.type import SrvArea
from s7_test_server import BitAwareServer


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.results = Path(cls.temp.name)
        cls.prediction_id = 'a'*32
        folder = cls.results/cls.prediction_id
        folder.mkdir()
        y, x = np.ogrid[:240, :240]
        mask = (((x-120)**2+(y-120)**2) <= 85**2).astype(np.uint8)
        Image.fromarray(mask).save(folder/'mask.png')
        Image.fromarray(np.full((240, 240, 3), 200, dtype=np.uint8)).save(folder/'original.jpg')
        (folder/'metadata.json').write_text(json.dumps({'name': 'test.jpg', 'dish_boundary': {'radius_px': 100}}))
        cls.memory = bytearray(650)
        for index in range(64):
            cls.memory[18+index*10:20+index*10] = struct.pack('>h', 31+index)
        cls.plc = BitAwareServer()
        cls.plc.register_area(SrvArea.DB, 100, cls.memory)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        cls.plc.start_to('127.0.0.1', port)
        cls.http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.http.predictor = SimpleNamespace(result_root=cls.results,
            slots=threading.BoundedSemaphore(6), lock=threading.Lock(),
            model_name='test-fixture', model_sha256='test',
            config={'width': 240, 'height': 240}, device=SimpleNamespace(type='cpu'))
        cls.http.plc_transfer = PLCTransfer(PLCConfig(host='127.0.0.1', port=port))
        cls.http.public_origin_file = None
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.http.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.http.server_close()
        cls.thread.join()
        cls.plc.stop()
        cls.plc.destroy()
        cls.temp.cleanup()

    def post(self, path, body, headers=None):
        options = {'Content-Type': 'application/json', 'X-PLC-Request': '1'}
        if headers:
            options.update(headers)
        request = Request(self.base+path, data=json.dumps(body).encode(), headers=options)
        try:
            with urlopen(request, timeout=20) as response:
                return response.status, json.load(response)
        except HTTPError as response:
            return response.code, json.load(response)

    def test_01_planning_to_plc_full_flow(self):
        code, plan = self.post('/api/plans/create', {
            'prediction_id': self.prediction_id,
            'calibration': {'center_x_px': 120, 'center_y_px': 120,
                            'dish_diameter_px': 200, 'dish_diameter_mm': 90,
                            'coordinate_origin': {'kind':'fixed_screw','x_px':220,'y_px':220,'image_width':240,'image_height':240}},
            'diameter_mm': 20, 'gap_mm': 0.5, 'default_depth_mm': 1})
        self.assertEqual(code, 200, plan)
        self.assertGreater(plan['count'], 0)
        self.assertLessEqual(plan['count'], 64)
        reference = {'prediction_id': self.prediction_id, 'plan_id': plan['plan_id']}
        before = bytes(self.memory)
        code, preview = self.post('/api/plc/preview', reference)
        self.assertEqual(code, 200, preview)
        self.assertEqual(preview['units_per_mm'], 10)
        self.assertIn('0.1 mm', preview['rounding'])
        with urlopen(self.base+'/api/health') as response:
            self.assertEqual(json.load(response)['plc_units_per_mm'], 10)
        self.assertEqual(bytes(self.memory), before)
        # Caller-supplied coordinates are ignored: only the saved plan is sent.
        code, sent = self.post('/api/plc/send', dict(reference, points=[{'x_mm': 999, 'y_mm': 999}]))
        self.assertEqual(code, 200, sent)
        self.assertEqual(sent['count'], plan['count'])
        for point in prepare_plan(plan)['points']:
            offset = point['x_offset']
            self.assertEqual(struct.unpack('>ii', self.memory[offset:offset+8]), (point['x'], point['y']))
            self.assertEqual(self.memory[offset+8:offset+10], before[offset+8:offset+10])
        code, status = self.post('/api/plc/status', {})
        self.assertEqual(code, 200)
        self.assertTrue(status['pc_ready'])
        self.assertFalse(status['plc_taken'])
        snapshot = bytes(self.memory)
        self.assertEqual(self.post('/api/plc/send', reference)[0], 409)
        self.assertEqual(bytes(self.memory), snapshot)
        self.assertEqual(self.post('/api/plc/release', {})[0], 409)
        self.memory[4] |= 4
        self.assertEqual(self.post('/api/plc/release', {})[0], 200)
        self.assertEqual(self.memory[4], 4)
        self.memory[4] = 0  # Simulate the PLC resetting its own acknowledgment.
        self.assertEqual(self.post('/api/plc/send', reference)[0], 200)

    def test_02_local_origin_and_custom_header_required(self):
        before = bytes(self.memory)
        for headers in ({'Origin': 'https://example.com'}, {'Host': 'example.com'},
                        {'X-PLC-Request': ''}, {'Content-Type': 'text/plain'}):
            code, _ = self.post('/api/plc/status', {}, headers)
            self.assertIn(code, (403, 415))
        self.assertEqual(bytes(self.memory), before)

    def test_03_invalid_plan_cannot_access_files(self):
        code, _ = self.post('/api/plc/send', {'prediction_id': '../other', 'plan_id': 'b'*32})
        self.assertEqual(code, 400)
        code, _ = self.post('/api/plc/send', {'prediction_id': self.prediction_id, 'plan_id': 'b'*32})
        self.assertEqual(code, 404)

    def test_04_assets_and_health(self):
        for path in ('/', '/planner.js', '/plc_ui.js', '/planner.css'):
            with urlopen(self.base+path) as response:
                self.assertEqual(response.status, 200)
                self.assertGreater(len(response.read()), 100)
        with urlopen(self.base+'/api/health') as response:
            self.assertTrue(json.load(response)['plc_transfer_available'])

    def test_05_coordinate_test_endpoint_overwrites_without_ack(self):
        plan_id='c'*32
        folder=self.results/self.prediction_id/'plans'/plan_id
        folder.mkdir(parents=True,exist_ok=True)
        stored={'plan_id':plan_id,'count':1,'points':[{'x_mm':7.89,'y_mm':-2.34}],
                'coordinate_frame':{'xy_origin':'fixed_screw','x_positive':'image_left','y_positive':'image_up'}}
        (folder/'plan.json').write_text(json.dumps(stored),encoding='utf-8')
        self.memory[4]=5
        code,data=self.post('/api/plc/test-write',{'prediction_id':self.prediction_id,'plan_id':plan_id})
        self.assertEqual(code,200,data)
        self.assertEqual(data['mode'],'coordinates_only')
        self.assertEqual(self.memory[4],4)
        self.assertEqual(struct.unpack('>ii',self.memory[10:18]),(79,-23))


if __name__ == '__main__':
    unittest.main(verbosity=2)
