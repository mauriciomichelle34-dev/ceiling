import json
from pathlib import Path
import sys
import unittest
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from auto_workflow import validate_settings
import test_auto_workflow as aw
import test_plc_http as fixture


class RemovedUniformDepthTests(unittest.TestCase):
    def test_automatic_settings_no_longer_require_uniform_z(self):
        settings=aw.settings();settings.pop('default_depth_mm')
        settings['diameter_mm']=10;settings['gap_mm']=1
        checked=validate_settings(settings)
        self.assertEqual(checked['default_depth_mm'],0)
        self.assertEqual((checked['diameter_mm'],checked['gap_mm']),(10,1))


class PlanningDefaultsHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.HTTPTests.setUpClass();cls.fixture=fixture.HTTPTests;cls.base=cls.fixture.base

    @classmethod
    def tearDownClass(cls):cls.fixture.tearDownClass()

    def post(self,route,payload):return self.fixture.post(self,route,payload)

    def test_page_defaults_and_sampling_panel_no_uniform_z(self):
        with urlopen(self.base+'/') as response:html=response.read().decode('utf-8')
        self.assertIn('id="diameter" type="number" min="0.001" step="any" value="10"',html)
        self.assertIn('id="gap" type="number" min="0" step="any" value="1"',html)
        self.assertNotIn('id="defaultZ"',html);self.assertNotIn('id="applyDepth"',html)
        self.assertIn('Z 轴由 PLC 控制',html)

    def test_auto_plan_and_manual_regenerate_need_no_depth_or_plc_writes(self):
        before=bytes(self.fixture.memory)
        body={'prediction_id':self.fixture.prediction_id,'calibration':aw.settings()['calibration']}
        counts=[]
        for diameter,gap in ((10,1),(12,2)):
            code,plan=self.post('/api/plans/create',dict(body,diameter_mm=diameter,gap_mm=gap))
            self.assertEqual(code,200,plan)
            self.assertEqual((plan['diameter_mm'],plan['gap_mm']),(diameter,gap))
            self.assertEqual(plan['default_depth_mm'],0)
            self.assertGreater(plan['count'],0)
            counts.append(plan['count'])
        self.assertGreater(counts[0],counts[1])
        self.assertEqual(bytes(self.fixture.memory),before)


if __name__=='__main__':unittest.main(verbosity=2)
