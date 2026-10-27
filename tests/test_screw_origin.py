import copy
import csv
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from sampling import make_plan, measure_area, csv_bytes, validate_calibration
from plc_transfer import prepare_plan, PLCTransfer
from auto_workflow import AutomaticWorkflow, validate_settings
import test_auto_workflow as aw
from test_plc_transfer import FakeConnection, plan as fixture_plan


class ScrewOriginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration = aw.settings()['calibration']
        y, x = np.ogrid[:240, :240]
        cls.mask = ((x-120)**2+(y-120)**2 <= 85**2).astype(np.uint8)
        cls.plan = make_plan(cls.mask, cls.calibration, 20, .5, 1)

    def test_screw_center_drives_xy_csv_and_plc(self):
        self.assertEqual(self.plan['coordinate_frame']['xy_origin'], 'fixed_screw')
        rows = list(csv.DictReader(io.StringIO(csv_bytes(self.plan).decode('utf-8-sig'))))
        prepared = prepare_plan(self.plan)
        for p, row, sent in zip(self.plan['points'], rows, prepared['points']):
            self.assertAlmostEqual(p['x_mm'], (220-p['image_x_px'])*.45)
            self.assertAlmostEqual(p['y_mm'], (220-p['image_y_px'])*.45)
            self.assertAlmostEqual(float(row['x_mm']), p['x_mm'], places=5)
            self.assertAlmostEqual(float(row['y_mm']), p['y_mm'], places=5)
            self.assertLessEqual(abs(sent['x']-p['x_mm']*10), .500001)
            self.assertLessEqual(abs(sent['y']-p['y_mm']*10), .500001)

    def test_changing_origin_only_translates_xy_preserving_all_image_geometry_and_area(self):
        moved = copy.deepcopy(self.calibration)
        moved['coordinate_origin'].update(x_px=210, y_px=225)
        other = make_plan(self.mask, moved, 20, .5, 1)
        self.assertEqual(measure_area(self.mask, self.calibration)['foreground_pixels'],
                         measure_area(self.mask, moved)['foreground_pixels'])
        self.assertEqual(self.plan['count'], other['count'])
        for a, b in zip(self.plan['points'], other['points']):
            for key in ('id','image_x_px','image_y_px','layer','z_mm'):
                self.assertEqual(a[key], b[key])
            self.assertAlmostEqual(b['x_mm']-a['x_mm'], -4.5)
            self.assertAlmostEqual(b['y_mm']-a['y_mm'], 2.25)

    def test_missing_or_invalid_origin_refuses_new_plan(self):
        for origin in (None, {}, {'kind':'dish_center'},
                       dict(self.calibration['coordinate_origin'], image_width=239),
                       dict(self.calibration['coordinate_origin'], x_px=240),
                       dict(self.calibration['coordinate_origin'], y_px=float('nan'))):
            c = copy.deepcopy(self.calibration)
            if origin is None: c.pop('coordinate_origin')
            else: c['coordinate_origin'] = origin
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                make_plan(self.mask, c, 20, .5, 1)

    def test_dish_measurement_can_run_before_origin_is_selected(self):
        c = dict(self.calibration)
        c.pop('coordinate_origin')
        self.assertGreater(measure_area(self.mask, c)['mycelium_area_mm2'], 0)
        with self.assertRaises(ValueError):
            validate_settings(dict(aw.settings(), calibration=c))

    def test_old_or_unknown_frame_never_connects_or_writes_plc(self):
        connection = FakeConnection()
        transfer = PLCTransfer(connection_factory=lambda _: connection)
        for frame in (None, {'xy_origin':'dish_center'},
                      {'xy_origin':'fixed_screw','x_positive':'image_right','y_positive':'image_up'}):
            old = dict(fixture_plan(), coordinate_frame=frame)
            for send in (transfer.send, transfer.write_coordinates):
                with self.subTest(frame=frame), self.assertRaises(ValueError): send(old)
        self.assertEqual(connection.events, [])


class ScrewWorkflowTests(unittest.TestCase):
    def test_dish_moves_but_saved_screw_and_scale_stay_fixed_in_automatic_flow(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            predictor = aw.TestPredictor(root/'results')
            predict = predictor.predict
            def shifted(raw, name):
                meta = predict(raw, name)
                meta['dish_suggestion'].update(center_x_px=125, center_y_px=118)
                return meta
            predictor.predict = shifted
            connection = FakeConnection()
            transfer = PLCTransfer(connection_factory=lambda _: connection)
            workflow = AutomaticWorkflow(predictor, transfer, root/'state')
            try:
                workflow.start(dict(aw.settings(), transfer_mode='coordinates_only'))
                job = workflow.submit(aw.picture(), 'shifted.png', 'shifted-origin')
                result = aw.wait_until(lambda: workflow.get_job(job['job_id'])
                    if workflow.get_job(job['job_id'])['state'] in ('completed','failed') else None)
                self.assertEqual(result['state'], 'completed', result)
                c = result['plan']['calibration']
                self.assertEqual((c['center_x_px'], c['center_y_px']), (125,118))
                self.assertEqual(c['coordinate_origin'], aw.settings()['calibration']['coordinate_origin'])
                self.assertEqual(c['mm_per_pixel'], .45)
                self.assertEqual(workflow.settings['calibration']['center_x_px'],120)
                first = result['plan']['points'][0]
                expected = prepare_plan(result['plan'])['points'][0]
                self.assertAlmostEqual(first['x_mm'], (220-first['image_x_px'])*.45)
                self.assertEqual(struct.unpack('>ii',connection.memory[10:18]),(expected['x'],expected['y']))
            finally: workflow.close()

    def test_old_settings_do_not_start_or_silently_use_dish_origin_after_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = aw.settings(); old['calibration'].pop('coordinate_origin')
            (root/'settings.json').write_text(json.dumps(old))
            connection = FakeConnection()
            workflow = AutomaticWorkflow(aw.TestPredictor(root/'results'),
                PLCTransfer(connection_factory=lambda _: connection), root)
            try:
                self.assertIsNone(workflow.settings)
                self.assertIn('已停用',workflow.last_error)
                self.assertFalse(workflow.enabled)
                self.assertEqual(connection.events, [])
            finally: workflow.close()


if __name__ == '__main__': unittest.main(verbosity=2)
