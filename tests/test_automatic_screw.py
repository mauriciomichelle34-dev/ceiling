import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from screw_origin import (detect_screw,AutomaticScrewOrigin,ScrewConfig,
                          ScrewDetectionError,origin_from_metadata)
from auto_workflow import AutomaticWorkflow
from plc_transfer import PLCTransfer
from test_plc_transfer import FakeConnection
import test_auto_workflow as aw
import test_plc_http as fixture


def picture(centers=((560,470),)):
    image=Image.new('RGB',(800,600),(30,30,30));draw=ImageDraw.Draw(image)
    for x,y in centers:
        draw.ellipse((x-31,y-31,x+31,y+31),fill=(12,12,12))
        draw.ellipse((x-17,y-17,x+17,y+17),fill=(220,220,220))
        draw.ellipse((x-11,y-11,x+11,y+11),fill=(45,45,45))
    return image


class SilverScrewDetectorTests(unittest.TestCase):
    def test_silver_ring_center_follows_translation(self):
        for x,y in ((560,470),(590,450),(660,490)):
            result=detect_screw(picture(((x,y),)))
            self.assertLess(abs(result['origin']['x_px']-x),2)
            self.assertLess(abs(result['origin']['y_px']-y),2)

    def test_black_bolts_white_disks_and_metal_rectangle_are_not_origins(self):
        for kind in ('black','white','rectangle'):
            image=picture(())
            draw=ImageDraw.Draw(image)
            if kind=='rectangle':draw.rectangle((530,440,600,510),fill='white')
            elif kind=='white':draw.ellipse((543,453,577,487),fill='white')
            else:
                draw.ellipse((529,439,591,501),fill=(12,12,12))
                draw.ellipse((543,453,577,487),fill=(40,40,40))
            with self.subTest(kind=kind),self.assertRaises(ScrewDetectionError):detect_screw(image)

    def test_two_silver_candidates_are_ambiguous(self):
        with self.assertRaisesRegex(ScrewDetectionError,'多个'):
            detect_screw(picture(((560,470),(680,470))))

    def test_circle_inside_dish_is_excluded(self):
        boundary={'center_x_px':560,'center_y_px':470,'radius_px':100,'outer_radius_px':110}
        with self.assertRaises(ScrewDetectionError):detect_screw(picture(),boundary)


class ScrewOriginPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'state.json'
        self.config=ScrewConfig(image_width=800,image_height=600)
    def tearDown(self):self.temp.cleanup()

    def test_first_failure_never_uses_old_fixed_pixel_configuration(self):
        service=AutomaticScrewOrigin(self.config,self.path)
        with self.assertRaisesRegex(ScrewDetectionError,'尚无成功'):
            service.resolve(picture(()))
        self.assertFalse(self.path.exists())

    def test_success_persists_and_failed_detection_reuses_only_successful_position(self):
        service=AutomaticScrewOrigin(self.config,self.path)
        first=service.resolve(picture())
        before=self.path.read_bytes()
        fallback=service.resolve(picture(()))
        self.assertEqual(fallback['status'],'reused_last')
        self.assertEqual(fallback['origin'],first['origin'])
        self.assertEqual(fallback['detection_id'],first['detection_id'])
        self.assertEqual(self.path.read_bytes(),before)
        restarted=AutomaticScrewOrigin(self.config,self.path)
        self.assertEqual(restarted.resolve(picture(()))['origin'],first['origin'])

    def test_ambiguity_reuses_last_and_next_success_updates_it(self):
        service=AutomaticScrewOrigin(self.config,self.path)
        first=service.resolve(picture())
        fallback=service.resolve(picture(((560,470),(680,470))))
        self.assertEqual(fallback['origin'],first['origin'])
        second=service.resolve(picture(((590,450),)))
        self.assertEqual(second['status'],'detected')
        self.assertNotEqual(second['detection_id'],first['detection_id'])
        self.assertGreater(second['origin']['x_px'],first['origin']['x_px']+20)

    def test_wrong_frame_never_reuses_or_rescales_cached_coordinates(self):
        service=AutomaticScrewOrigin(self.config,self.path);service.resolve(picture())
        with self.assertRaisesRegex(ScrewDetectionError,'画幅'):
            service.resolve(picture().resize((400,300)))
        other=AutomaticScrewOrigin(ScrewConfig(image_width=400,image_height=300),self.path)
        self.assertIsNone(other.last)

    def test_cached_record_and_returned_results_are_not_shared_mutable_state(self):
        service=AutomaticScrewOrigin(self.config,self.path);first=service.resolve(picture())
        x=first['origin']['x_px'];first['origin']['x_px']=0
        service.status()['last_success']['origin']['x_px']=1
        self.assertEqual(service.resolve(picture(()))['origin']['x_px'],x)

    def test_corrupt_cache_does_not_supply_coordinates(self):
        self.path.write_text('{broken')
        service=AutomaticScrewOrigin(self.config,self.path)
        self.assertIsNone(service.last)
        with self.assertRaises(ScrewDetectionError):service.resolve(picture(()))


def record(origin,status='detected'):
    return {'status':status,'origin':origin,'detection_id':'test-detection','detected_at':'2026-10-27T00:00:00+00:00'}


class ImageBoundScrewHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.HTTPTests.setUpClass();cls.fixture=fixture.HTTPTests;cls.base=cls.fixture.base
        cls.fixture.http.screw_origin=object()
        cls.origins={}
        for identifier,x,y in ((cls.fixture.prediction_id,220,220),('b'*32,210,225)):
            folder=cls.fixture.results/identifier;folder.mkdir(exist_ok=True)
            if identifier!=cls.fixture.prediction_id:
                (folder/'mask.png').write_bytes((cls.fixture.results/cls.fixture.prediction_id/'mask.png').read_bytes())
            origin={'kind':'fixed_screw','x_px':x,'y_px':y,'image_width':240,'image_height':240}
            meta={'name':identifier+'.png','width':240,'height':240,
                  'dish_boundary':{'radius_px':100,'center_x_px':120,'center_y_px':120},
                  'dish_suggestion':{'center_x_px':120,'center_y_px':120,'dish_diameter_px':200},
                  'screw_origin':record(origin)}
            (folder/'metadata.json').write_text(json.dumps(meta))
            cls.origins[identifier]=origin
    @classmethod
    def tearDownClass(cls):cls.fixture.tearDownClass()
    def post(self,path,body):return self.fixture.post(self,path,body)

    def test_each_plan_and_plc_preview_use_its_own_image_not_latest_cache(self):
        before=bytes(self.fixture.memory)
        plans=[]
        for identifier,origin in self.origins.items():
            code,plan=self.post('/api/plans/create',dict(prediction_id=identifier,diameter_mm=20,gap_mm=.5,
                calibration={'coordinate_origin':{'x_px':9999}}))
            self.assertEqual(code,200,plan)
            self.assertEqual(plan['calibration']['coordinate_origin'],origin)
            self.assertEqual(plan['screw_origin']['origin'],origin)
            plans.append((identifier,plan))
        # Loading image B must not invalidate A or replace A's origin.
        for identifier,plan in plans:
            code,preview=self.post('/api/plc/preview',{'prediction_id':identifier,'plan_id':plan['plan_id']})
            self.assertEqual(code,200,preview)
        self.assertEqual(bytes(self.fixture.memory),before)

    def test_tampered_plan_origin_is_rejected_before_plc_writes(self):
        identifier=self.fixture.prediction_id
        code,plan=self.post('/api/plans/create',dict(prediction_id=identifier,diameter_mm=20,gap_mm=.5))
        self.assertEqual(code,200)
        plan['calibration']['coordinate_origin']['x_px']=1
        path=self.fixture.results/identifier/'plans'/plan['plan_id']/'plan.json'
        path.write_text(json.dumps(plan))
        before=bytes(self.fixture.memory)
        code,_=self.post('/api/plc/test-write',dict(prediction_id=identifier,plan_id=plan['plan_id']))
        self.assertEqual(code,400);self.assertEqual(bytes(self.fixture.memory),before)


class AutomaticScrewWorkflowTests(unittest.TestCase):
    def test_automatic_pipeline_replaces_saved_origin_with_each_photo_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);predictor=aw.TestPredictor(root/'results');predict=predictor.predict
            predictor.screw_origin=object()
            def detected(raw,name):
                meta=predict(raw,name)
                origin=copy.deepcopy(aw.settings()['calibration']['coordinate_origin'])
                origin.update(x_px=220 if predictor.calls==1 else 210,y_px=220 if predictor.calls==1 else 225)
                meta['screw_origin']=record(origin,'detected' if predictor.calls==1 else 'reused_last')
                return meta
            predictor.predict=detected
            connection=FakeConnection()
            workflow=AutomaticWorkflow(predictor,PLCTransfer(connection_factory=lambda _:connection),root/'state')
            try:
                workflow.start(dict(aw.settings(),transfer_mode='coordinates_only'))
                results=[]
                for index in range(2):
                    job=workflow.submit(aw.picture(),f'{index}.png',f'auto-origin-{index}')
                    value=aw.wait_until(lambda:workflow.get_job(job['job_id']) if workflow.get_job(job['job_id'])['state'] in ('completed','failed') else None)
                    self.assertEqual(value['state'],'completed',value)
                    results.append(value['plan'])
                for a,b in zip(results[0]['points'],results[1]['points']):
                    self.assertEqual(a['image_x_px'],b['image_x_px'])
                    self.assertAlmostEqual(b['x_mm']-a['x_mm'],-4.5)
                    self.assertAlmostEqual(b['y_mm']-a['y_mm'],2.25)
            finally:workflow.close()


if __name__=='__main__':unittest.main()
