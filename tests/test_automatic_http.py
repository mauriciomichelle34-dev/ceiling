import json
from pathlib import Path
import struct
import sys
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import test_plc_http as fixture
import test_auto_workflow as aw
from auto_workflow import AutomaticWorkflow


class AutomaticHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.HTTPTests.setUpClass()
        cls.http = fixture.HTTPTests.http
        cls.base = fixture.HTTPTests.base
        cls.memory = fixture.HTTPTests.memory
        cls.http.predictor = aw.TestPredictor(fixture.HTTPTests.results)
        cls.http.automatic = AutomaticWorkflow(cls.http.predictor, cls.http.plc_transfer,
                                                fixture.HTTPTests.results/'automatic')
        cls.accepted = []
        cls.end_ack = threading.Event()
        def ack():
            while not cls.end_ack.is_set():
                flags = cls.memory[4]
                if flags & 1 and not flags & 4:
                    cls.accepted.append(bytes(cls.memory))
                    cls.memory[4] |= 4
                elif not flags & 1 and flags & 4:
                    cls.memory[4] &= ~4
                cls.end_ack.wait(0.005)
        cls.ack = threading.Thread(target=ack)
        cls.ack.start()

    @classmethod
    def tearDownClass(cls):
        cls.http.automatic.close()
        cls.end_ack.set(); cls.ack.join()
        fixture.HTTPTests.tearDownClass()

    def post(self, path, body, headers=None):
        return fixture.HTTPTests.post(self, path, body, headers)

    def image(self, capture_id, raw=None, headers=None):
        options = {'Content-Type':'application/octet-stream', 'X-PLC-Request':'1',
                   'X-Capture-ID':capture_id, 'X-File-Name':capture_id+'.png'}
        options.update(headers or {})
        request = Request(self.base+'/api/automatic/images', data=raw or aw.picture(), headers=options)
        try:
            with urlopen(request, timeout=10) as response:
                return response.status, json.load(response)
        except HTTPError as response:
            return response.code, json.load(response)

    def job_done(self, job_id):
        return aw.wait_until(lambda:self.post('/api/automatic/job', {'job_id':job_id})[1]
             if self.post('/api/automatic/job', {'job_id':job_id})[1]['state'] in {'completed','failed'} else None)

    def test_01_full_image_api_to_s7_twice(self):
        self.assertEqual(self.image('disabled')[0], 409)
        code, _ = self.post('/api/automatic/start', {'settings':aw.settings()})
        self.assertEqual(code, 200)
        first_code, first = self.image('sdk-1')
        second_code, second = self.image('sdk-2')
        self.assertEqual((first_code, second_code), (202,202))
        self.assertEqual(self.job_done(first['job_id'])['state'], 'completed')
        self.assertEqual(self.job_done(second['job_id'])['state'], 'completed')
        self.assertEqual(len(self.accepted), 2)
        self.assertGreater(struct.unpack('>i', self.accepted[0][:4])[0], 0)
        self.assertEqual(self.accepted[0][18:20], struct.pack('>h', 31))
        code, duplicate = self.image('sdk-1')
        self.assertEqual(code, 200)
        self.assertTrue(duplicate['duplicate'])
        self.assertEqual(len(self.accepted), 2)
        self.assertEqual(self.image('sdk-1', raw=aw.picture((239,240)))[0], 409)
        self.assertEqual(self.post('/api/plc/send', {})[0], 409)
        self.assertEqual(self.post('/api/plc/release', {})[0], 409)
        self.assertEqual(self.post('/api/automatic/stop', {})[0], 200)

    def test_02_image_and_controls_are_local_only(self):
        for headers in ({'Origin':'https://example.com'}, {'Host':'example.com'}, {'X-PLC-Request':''}):
            self.assertEqual(self.post('/api/automatic/start', {}, headers)[0], 403)
            self.assertEqual(self.image('untrusted', headers=headers)[0], 403)
        self.assertEqual(len(self.accepted), 2)

    def test_03_saved_parameters_and_stop_are_available(self):
        code, state = self.post('/api/automatic/status', {})
        self.assertEqual(code, 200)
        self.assertTrue(state['configured'])
        self.assertFalse(state['enabled'])
        self.assertEqual(len(state['jobs']), 2)
        self.assertTrue(all('plan' not in job for job in state['jobs']))
        code, state = self.post('/api/automatic/start', {})
        self.assertEqual(code, 200)
        self.assertTrue(state['enabled'])
        self.assertEqual(self.post('/api/automatic/stop', {})[0], 200)

    def test_04_automatic_script_is_served(self):
        with urlopen(self.base+'/automatic_ui.js') as response:
            self.assertEqual(response.status, 200)
            self.assertIn(b'/api/automatic/images', response.read())

    def test_05_image_input_helper(self):
        from image_input import submit_image
        self.assertEqual(self.post('/api/automatic/start', {'settings':aw.settings()})[0], 200)
        before = len(self.accepted)
        job = submit_image(aw.picture(), 'helper.png', 'helper-frame', self.base)
        self.assertEqual(self.job_done(job['job_id'])['state'], 'completed')
        self.assertTrue(submit_image(aw.picture(), 'helper.png', 'helper-frame', self.base)['duplicate'])
        self.assertEqual(len(self.accepted), before+1)
        with self.assertRaises(ValueError):
            submit_image(b'test', 'one.png', 'one', 'https://example.com')
        self.assertEqual(self.post('/api/automatic/stop', {})[0], 200)

    def test_06_coordinate_mode_two_images_without_any_acknowledger(self):
        self.end_ack.set(); self.ack.join()
        self.memory[4]=1
        bit_start=len(fixture.HTTPTests.plc.bit_writes)
        code,_=self.post('/api/automatic/start',{'settings':dict(aw.settings(),transfer_mode='coordinates_only')})
        self.assertEqual(code,200)
        first=self.image('coordinates-http-1')[1]
        second=self.image('coordinates-http-2')[1]
        self.assertEqual(self.job_done(first['job_id'])['state'],'completed')
        self.assertEqual(self.job_done(second['job_id'])['state'],'completed')
        self.assertEqual(self.memory[4],0)
        self.assertEqual(fixture.HTTPTests.plc.bit_writes[bit_start:],[(100,4,0,False),(100,4,0,False)])
        self.assertEqual(self.post('/api/automatic/stop',{})[0],200)


if __name__ == '__main__':
    unittest.main(verbosity=2)
