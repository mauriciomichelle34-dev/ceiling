"""Local-only light settings and read-only probe; all light sockets are simulated."""
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from camera_service import CameraConfig, CameraService
from light_service import LightConfig, LightService
from local_app import Handler
from test_camera_light import SimulatedLight
from test_camera_service import FakeAdapter


class LightHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state = SimulatedLight()
        self.light = LightService(LightConfig(), Path(self.temp.name) / 'light.json',
                                  connector=self.state.connect, sleeper=lambda _: None)
        self.http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.http.public_origin_file = None
        self.http.camera = CameraService(CameraConfig(), Path(self.temp.name) / 'camera',
                                          FakeAdapter, illumination=self.light)
        self.http.automatic = SimpleNamespace(lock=threading.RLock(), enabled=False,
                                               active_id=None, camera_busy=False, settings=None)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.http.server_port}'
        self.opener = build_opener(ProxyHandler({}))

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()
        self.http.camera.close()
        self.temp.cleanup()

    def post(self, action, body, headers=None):
        options = {'Content-Type': 'application/json', 'X-PLC-Request': '1'}
        options.update(headers or {})
        request = Request(self.base + '/api/camera/light/' + action,
                          json.dumps(body).encode(), headers=options)
        try:
            response = self.opener.open(request, timeout=3)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response)

    def test_save_is_persistent_and_probe_never_switches_light(self):
        code, result = self.post('settings', {'enabled': True, 'brightness': 150})
        self.assertEqual(code, 200, result)
        self.assertEqual(LightConfig.load(self.light.config_path).brightness, 150)
        self.assertEqual(self.state.connections, 0)
        code, result = self.post('probe', {})
        self.assertEqual(code, 200, result)
        self.assertTrue(result['light']['connected'])
        self.assertEqual(self.state.events, [('light', '4', 0)])

    def test_foreign_origin_missing_header_and_invalid_fields_do_no_io(self):
        for headers in ({'Origin': 'https://example.com'}, {'Host': 'example.com'},
                        {'X-PLC-Request': ''}):
            self.assertEqual(self.post('probe', {}, headers)[0], 403)
        for body in ({'enabled': True, 'brightness': 0},
                     {'enabled': True, 'brightness': 100, 'host': '192.168.6.10'}):
            self.assertEqual(self.post('settings', body)[0], 400)
        self.assertEqual(self.state.connections, 0)

    def test_automatic_work_blocks_probe_and_parameter_change(self):
        for field, value in [('enabled', True), ('active_id', 'job'), ('camera_busy', True)]:
            original = getattr(self.http.automatic, field)
            setattr(self.http.automatic, field, value)
            try:
                self.assertEqual(self.post('probe', {})[0], 409)
                self.assertEqual(self.post('settings', {'enabled': False, 'brightness': 80})[0], 409)
            finally:
                setattr(self.http.automatic, field, original)
        self.assertEqual(self.state.connections, 0)
        self.assertTrue(self.light.config.enabled)

    def test_camera_operation_blocks_light_settings(self):
        self.http.camera.operation.acquire()
        try:
            self.assertEqual(self.post('settings', {'enabled': False, 'brightness': 80})[0], 409)
            self.assertEqual(self.post('probe', {})[0], 409)
        finally:
            self.http.camera.operation.release()
        self.assertEqual(self.state.connections, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
