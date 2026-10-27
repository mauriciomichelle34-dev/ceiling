import copy
import csv
import io
import json
from pathlib import Path
import sys
import unittest
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sampling import make_plan, csv_bytes, layout_svg, with_left_positive_x, with_depths
from plc_transfer import prepare_plan, scaled_coordinate
import test_plc_http as fixture


class CoordinateAxisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        y,x = np.ogrid[:240,:240]
        mask = ((x-120)**2+(y-120)**2<=85**2).astype(np.uint8)
        cls.plan = make_plan(mask,{'center_x_px':120,'center_y_px':120,
            'dish_diameter_px':200,'dish_diameter_mm':90,
            'coordinate_origin':{'kind':'fixed_screw','x_px':120,'y_px':120,'image_width':240,'image_height':240}},20,0.5,1)

    def test_left_is_positive_and_image_geometry_y_depth_are_preserved(self):
        plan = self.plan
        self.assertEqual(plan['coordinate_frame']['x_positive'],'image_left')
        self.assertEqual(plan['coordinate_frame']['xy_origin'],'fixed_screw')
        left = [p for p in plan['points'] if p['image_x_px']<119.999]
        right = [p for p in plan['points'] if p['image_x_px']>120.001]
        self.assertTrue(left); self.assertTrue(right)
        self.assertTrue(all(p['x_mm']>0 for p in left))
        self.assertTrue(all(p['x_mm']<0 for p in right))
        for point in plan['points']:
            if point['image_y_px']<119.999: self.assertGreater(point['y_mm'],0)
            if point['image_y_px']>120.001: self.assertLess(point['y_mm'],0)
            self.assertEqual(point['z_mm'],1)
        svg = ET.fromstring(layout_svg(plan,b'preview'))
        circles = svg.findall('{http://www.w3.org/2000/svg}circle')
        self.assertEqual([(float(c.attrib['cx']),float(c.attrib['cy'])) for c in circles],
                         [(p['image_x_px'],p['image_y_px']) for p in plan['points']])

    def test_csv_and_plc_use_same_signed_coordinates_without_second_flip(self):
        rows = list(csv.DictReader(io.StringIO(csv_bytes(self.plan).decode('utf-8-sig'))))
        plc = prepare_plan(self.plan)
        for point, row, address in zip(self.plan['points'],rows,plc['points']):
            self.assertAlmostEqual(float(row['x_mm']),point['x_mm'],places=5)
            self.assertAlmostEqual(float(row['y_mm']),point['y_mm'],places=5)
            self.assertEqual(address['x'],scaled_coordinate(point['x_mm'],'X'))
            self.assertEqual(address['y'],scaled_coordinate(point['y_mm'],'Y'))

    def test_old_plan_conversion_is_idempotent_and_does_not_mutate_saved_geometry(self):
        old = copy.deepcopy(self.plan)
        old['coordinate_frame']['x_positive'] = 'image_right'
        for point in old['points']: point['x_mm'] = -point['x_mm']
        before = copy.deepcopy(old)
        current = with_left_positive_x(old)
        self.assertEqual(current,self.plan)
        self.assertEqual(with_left_positive_x(current),current)
        self.assertEqual(old,before)
        self.assertEqual(with_depths(old,1,{})['points'],self.plan['points'])


class LegacyCoordinateHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.HTTPTests.setUpClass()
        cls.fixture = fixture.HTTPTests
        cls.base = cls.fixture.base

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDownClass()

    def test_saved_old_plan_json_csv_and_plc_preview_agree_without_rewriting_source(self):
        identifier = 'd'*32
        folder = self.fixture.results/self.fixture.prediction_id/'plans'/identifier
        folder.mkdir(parents=True)
        plan = {'plan_id':identifier,'source_name':'axis.jpg','units':'mm','count':2,
            'coordinate_frame':{'xy_origin':'dish_center','x_positive':'image_right','y_positive':'image_up'},
            'diameter_mm':3,'gap_mm':0.5,'default_depth_mm':1,
            'points':[{'id':1,'x_mm':-12.34,'y_mm':5.67,'z_mm':1,'depth_overridden':False},
                      {'id':2,'x_mm':8.9,'y_mm':-2.3,'z_mm':1,'depth_overridden':False}]}
        path = folder/'plan.json'
        path.write_text(json.dumps(plan),encoding='utf-8')
        saved = path.read_bytes()
        base = self.fixture.base+'/results/'+self.fixture.prediction_id+'/plans/'+identifier
        with urlopen(base+'/plan.json') as response: current = json.load(response)
        with urlopen(base+'/points.csv') as response:
            rows = list(csv.DictReader(io.StringIO(response.read().decode('utf-8-sig'))))
        code, preview = self.fixture.post(self,'/api/plc/preview',{
            'prediction_id':self.fixture.prediction_id,'plan_id':identifier})
        self.assertEqual(code,400,preview)
        self.assertIn('旧的培养皿中心规划已停用',preview['error'])
        self.assertEqual([p['x_mm'] for p in current['points']],[12.34,-8.9])
        self.assertEqual([p['y_mm'] for p in current['points']],[5.67,-2.3])
        self.assertEqual([float(p['x_mm']) for p in rows],[12.34,-8.9])
        self.assertEqual(current['coordinate_frame']['x_positive'],'image_left')
        self.assertEqual(path.read_bytes(),saved)


if __name__=='__main__': unittest.main(verbosity=2)
