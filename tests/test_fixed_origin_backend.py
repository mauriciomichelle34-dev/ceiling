import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fixed_origin import load_origin, apply_origin, origin_for_image
from auto_workflow import AutomaticWorkflow
from plc_transfer import PLCTransfer
from test_plc_transfer import FakeConnection
import test_auto_workflow as aw
import test_plc_http as fixture

ORIGIN = {'kind':'fixed_screw','x_px':220,'y_px':220,'image_width':240,'image_height':240}


class FixedOriginBackendTests(unittest.TestCase):
    def test_actual_saved_origin_matches_full_camera_frame_and_rejects_changed_frame(self):
        origin = load_origin(ROOT/'coordinate_origin.json')
        self.assertEqual((origin['x_px'],origin['y_px']),(4045,3010))
        self.assertEqual(origin_for_image(origin,5472,3648),origin)
        for size in ((3648,5472),(2736,1824)):
            with self.assertRaises(ValueError): origin_for_image(origin,*size)

    def test_backend_overrides_browser_origin_without_mutating_input(self):
        c = aw.settings()['calibration']
        c['coordinate_origin'].update(x_px=1,y_px=2)
        before = copy.deepcopy(c)
        actual = apply_origin(c,ORIGIN,240,240)
        self.assertEqual(actual['coordinate_origin'],ORIGIN)
        self.assertEqual(c,before)

    def test_automatic_origin_is_injected_and_persists_without_browser_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);connection=FakeConnection()
            transfer=PLCTransfer(connection_factory=lambda _:connection)
            predictor=aw.TestPredictor(root/'results')
            settings=aw.settings();settings['calibration'].pop('coordinate_origin')
            settings['transfer_mode']='coordinates_only'
            workflow=AutomaticWorkflow(predictor,transfer,root/'state',fixed_origin=ORIGIN)
            try:
                workflow.start(settings)
                job=workflow.submit(aw.picture(),'without-selection.png','backend-origin')
                result=aw.wait_until(lambda:workflow.get_job(job['job_id'])
                    if workflow.get_job(job['job_id'])['state'] in ('completed','failed') else None)
                self.assertEqual(result['state'],'completed',result)
                self.assertEqual(result['plan']['calibration']['coordinate_origin'],ORIGIN)
            finally: workflow.close()
            restarted=AutomaticWorkflow(predictor,transfer,root/'state',fixed_origin=ORIGIN)
            try:
                self.assertEqual(restarted.settings['calibration']['coordinate_origin'],ORIGIN)
                self.assertFalse(restarted.enabled)
            finally: restarted.close()

    def test_wrong_image_size_fails_before_plc_connection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);connection=FakeConnection()
            workflow=AutomaticWorkflow(aw.TestPredictor(root/'results'),
                PLCTransfer(connection_factory=lambda _:connection),root/'state',fixed_origin=ORIGIN)
            try:
                settings=aw.settings();settings['image_width']=239
                with self.assertRaises(ValueError):workflow.start(settings)
                self.assertEqual(connection.events,[])
            finally:workflow.close()


class FixedOriginHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.HTTPTests.setUpClass()
        cls.fixture=fixture.HTTPTests
        cls.base=cls.fixture.base
        cls.fixture.http.fixed_origin=dict(ORIGIN)
        path=cls.fixture.results/cls.fixture.prediction_id/'metadata.json'
        meta=json.loads(path.read_text())
        meta['dish_suggestion']={'center_x_px':120,'center_y_px':120,'dish_diameter_px':200}
        path.write_text(json.dumps(meta))

    @classmethod
    def tearDownClass(cls):cls.fixture.tearDownClass()

    def post(self,route,payload):return self.fixture.post(self,route,payload)

    def test_no_origin_controls_and_manual_plan_uses_server_config(self):
        with urlopen(self.fixture.base+'/') as response:html=response.read().decode('utf-8')
        for removed in ('id="originX"','id="originY"','id="pickOrigin"','id="saveOrigin"'):
            self.assertNotIn(removed,html)
        settings=aw.settings();settings['calibration'].pop('coordinate_origin')
        code,plan=self.post('/api/plans/create',dict(settings,prediction_id=self.fixture.prediction_id))
        self.assertEqual(code,200,plan)
        self.assertEqual(plan['calibration']['coordinate_origin'],ORIGIN)
        reference={'prediction_id':self.fixture.prediction_id,'plan_id':plan['plan_id']}
        code,preview=self.post('/api/plc/preview',reference)
        self.assertEqual(code,200,preview)
        self.assertEqual(preview['coordinate_frame']['xy_origin'],'fixed_screw')
        with urlopen(self.fixture.base+'/api/health') as response:
            self.assertEqual(json.load(response)['fixed_coordinate_origin'],ORIGIN)
        plan['calibration']['coordinate_origin']['x_px']=1
        path=self.fixture.results/self.fixture.prediction_id/'plans'/plan['plan_id']/'plan.json'
        path.write_text(json.dumps(plan),encoding='utf-8')
        code,error=self.post('/api/plc/preview',reference)
        self.assertEqual(code,400,error)


if __name__=='__main__':unittest.main(verbosity=2)
