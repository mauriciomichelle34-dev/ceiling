import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fixed_origin import automatic_calibration
from auto_workflow import AutomaticWorkflow
from plc_transfer import PLCTransfer
from test_plc_transfer import FakeConnection
from test_fixed_origin_backend import ORIGIN
import test_auto_workflow as aw
import test_plc_http as fixture


class AutomaticCalibrationTests(unittest.TestCase):
    def test_fixed_diameter_uses_each_detected_circle_and_retains_screw_origin(self):
        for diameter,center in ((200,120),(180,125)):
            meta={'width':240,'height':240,'dish_suggestion':{
                'center_x_px':center,'center_y_px':120,'dish_diameter_px':diameter,'dish_diameter_mm':999}}
            c=automatic_calibration(meta,ORIGIN)
            self.assertEqual(c['dish_diameter_mm'],90)
            self.assertEqual(c['mm_per_pixel'],90/diameter)
            self.assertEqual(c['center_x_px'],center)
            self.assertEqual(c['coordinate_origin'],ORIGIN)

    def test_missing_detection_is_not_replaced_with_browser_calibration(self):
        with self.assertRaises(ValueError):automatic_calibration({'width':240,'height':240},ORIGIN)

    def test_automatic_flow_ignores_old_diameter_and_recalibrates_each_image(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); connection=FakeConnection()
            predictor=aw.TestPredictor(root/'results');predict=predictor.predict
            def changed_circle(raw,name):
                meta=predict(raw,name)
                meta['dish_suggestion']['dish_diameter_px']=200 if predictor.calls==1 else 180
                return meta
            predictor.predict=changed_circle
            workflow=AutomaticWorkflow(predictor,PLCTransfer(connection_factory=lambda _:connection),
                root/'state',fixed_origin=ORIGIN)
            settings=aw.settings();settings['calibration']['dish_diameter_mm']=77
            settings['transfer_mode']='coordinates_only'
            try:
                workflow.start(settings)
                self.assertEqual(workflow.settings['calibration']['dish_diameter_mm'],90)
                for index,diameter in enumerate((200,180)):
                    job=workflow.submit(aw.picture(),f'{index}.png',f'fixed90-{index}')
                    result=aw.wait_until(lambda:workflow.get_job(job['job_id'])
                        if workflow.get_job(job['job_id'])['state'] in ('completed','failed') else None)
                    self.assertEqual(result['state'],'completed',result)
                    c=result['plan']['calibration']
                    self.assertEqual(c['dish_diameter_mm'],90)
                    self.assertEqual(c['mm_per_pixel'],90/diameter)
            finally:workflow.close()


class FixedDiameterHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.HTTPTests.setUpClass();cls.fixture=fixture.HTTPTests;cls.base=cls.fixture.base
        cls.fixture.http.fixed_origin=copy.deepcopy(ORIGIN)
        path=cls.fixture.results/cls.fixture.prediction_id/'metadata.json'
        meta=json.loads(path.read_text())
        meta['dish_suggestion']={'center_x_px':120,'center_y_px':120,'dish_diameter_px':200}
        path.write_text(json.dumps(meta))

    @classmethod
    def tearDownClass(cls):cls.fixture.tearDownClass()

    def post(self,route,payload):return self.fixture.post(self,route,payload)

    def test_area_and_plan_need_no_browser_calibration_and_ignore_client_geometry(self):
        reference={'prediction_id':self.fixture.prediction_id}
        code,area=self.post('/api/area',reference)
        self.assertEqual(code,200,area)
        self.assertEqual(area['calibration']['dish_diameter_mm'],90)
        forged={'center_x_px':1,'center_y_px':1,'dish_diameter_px':1,'dish_diameter_mm':100}
        code,plan=self.post('/api/plans/create',dict(reference,calibration=forged,
            diameter_mm=20,gap_mm=.5,default_depth_mm=1))
        self.assertEqual(code,200,plan)
        self.assertEqual(plan['calibration']['center_x_px'],120)
        self.assertEqual(plan['calibration']['dish_diameter_px'],200)
        self.assertEqual(plan['calibration']['dish_diameter_mm'],90)

    def test_compact_fixed_value_replaces_manual_controls(self):
        with urlopen(self.base+'/') as response:html=response.read().decode('utf-8')
        self.assertIn('90 <small>mm</small>',html)
        for removed in ('dishMm','dishPx','centerX','centerY','pickCenter','pickEdge','saveCalibration','calibrationDetails'):
            self.assertNotIn('id="'+removed+'"',html)
        with urlopen(self.base+'/api/health') as response:health=json.load(response)
        self.assertEqual(health['fixed_dish_diameter_mm'],90)
        self.assertTrue(health['automatic_dish_calibration'])


if __name__=='__main__':unittest.main(verbosity=2)
