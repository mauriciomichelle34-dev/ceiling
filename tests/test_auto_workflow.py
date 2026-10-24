import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from auto_workflow import AutomaticWorkflow, WorkflowError, validate_settings
from plc_transfer import PLCTransfer
from test_plc_transfer import FakeConnection, plan


def settings(timeout=2):
    return {'image_width':240, 'image_height':240,
            'calibration':{'center_x_px':120, 'center_y_px':120,
                           'dish_diameter_px':200, 'dish_diameter_mm':90},
            'diameter_mm':20, 'gap_mm':0.5, 'default_depth_mm':1, 'plc_timeout_s':timeout}


def picture(size=(240, 240)):
    image = Image.new('RGB', size, (200, 210, 190))
    buffer = io.BytesIO(); image.save(buffer, format='PNG')
    return buffer.getvalue()


class TestPredictor:
    def __init__(self, root):
        self.result_root = Path(root)
        self.result_root.mkdir(parents=True, exist_ok=True)
        self.slots, self.lock = threading.BoundedSemaphore(6), threading.Lock()
        self.calls = 0

    def predict(self, raw, name):
        self.calls += 1
        identifier = uuid.uuid4().hex
        folder = self.result_root/identifier; folder.mkdir()
        y, x = np.ogrid[:240, :240]
        Image.fromarray((((x-120)**2+(y-120)**2) <= 85**2).astype(np.uint8)).save(folder/'mask.png')
        Image.open(io.BytesIO(raw)).save(folder/'original.jpg')
        meta = {'id':identifier, 'name':name, 'width':240, 'height':240,
                'dish_boundary':{'radius_px':100},
                **{key:f'/results/{identifier}/{filename}' for key,filename in
                   [('original','original.jpg'),('region','region.png'),('region_outline','region_outline.png')]}}
        (folder/'metadata.json').write_text(json.dumps(meta))
        return meta


def wait_until(function, timeout=6):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        value = function()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError('Timed out waiting for workflow condition')


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.predictor = TestPredictor(self.root/'results')
        self.connection = FakeConnection()
        self.transfer = PLCTransfer(connection_factory=lambda config:self.connection)
        self.workflow = AutomaticWorkflow(self.predictor, self.transfer, self.root/'state')
        self.ack_stop = threading.Event()
        self.ack_thread = None
        self.accepted = []

    def tearDown(self):
        self.workflow.close()
        self.ack_stop.set()
        if self.ack_thread:
            self.ack_thread.join()
        self.temp.cleanup()

    def start_acknowledger(self):
        def run():
            while not self.ack_stop.is_set():
                flags = self.connection.memory[4]
                if flags & 1 and not flags & 4:
                    self.accepted.append(bytes(self.connection.memory))
                    self.connection.memory[4] |= 4
                elif not flags & 1 and flags & 4:
                    self.connection.memory[4] &= ~4
                self.ack_stop.wait(0.005)
        self.ack_thread = threading.Thread(target=run)
        self.ack_thread.start()

    def done(self, job_id):
        return wait_until(lambda:self.workflow.get_job(job_id)
                          if self.workflow.get_job(job_id)['state'] in {'completed','failed','cancelled'} else None)

    def test_first_setup_and_saved_restart(self):
        with self.assertRaises(WorkflowError): self.workflow.start()
        self.workflow.start(settings())
        self.workflow.close()
        self.workflow = AutomaticWorkflow(self.predictor, self.transfer, self.root/'state')
        self.assertFalse(self.workflow.enabled)
        self.assertTrue(self.workflow.snapshot()['configured'])
        self.workflow.start()
        self.assertTrue(self.workflow.enabled)

    def test_receive_two_images_and_complete_without_manual_send(self):
        self.workflow.start(settings())
        self.start_acknowledger()
        first = self.workflow.submit(picture(), 'first.png', 'camera-frame-1')
        second = self.workflow.submit(picture(), 'second.png', 'camera-frame-2')
        self.assertEqual(self.done(first['job_id'])['state'], 'completed')
        self.assertEqual(self.done(second['job_id'])['state'], 'completed')
        self.assertEqual(len(self.accepted), 2)
        self.assertEqual(self.predictor.calls, 2)
        self.assertEqual(self.connection.memory[4], 0)
        for memory in self.accepted:
            self.assertGreater(struct.unpack('>i', memory[:4])[0], 0)
            self.assertEqual(memory[18:20], struct.pack('>h', -32))
        signals = [event for event in self.connection.events if event[0] == 'signal']
        self.assertEqual(signals, [('signal',True),('signal',False),('signal',True),('signal',False)])

    def test_coordinate_mode_accepts_two_images_without_plc_ack(self):
        self.connection.memory[4] = 1
        self.workflow.start(dict(settings(), transfer_mode='coordinates_only'))
        with patch('auto_workflow.make_plan', side_effect=[plan(2),plan(1)]):
            first=self.workflow.submit(picture(),'first.png','coordinates-first')
            second=self.workflow.submit(picture(),'second.png','coordinates-second')
            self.assertEqual(self.done(first['job_id'])['state'],'completed')
            self.assertEqual(self.done(second['job_id'])['state'],'completed')
        self.assertEqual(struct.unpack('>i',self.connection.memory[:4])[0],1)
        self.assertEqual(self.connection.memory[4],0)
        self.assertEqual([e for e in self.connection.events if e[0]=='signal'],[('signal',False),('signal',False)])

    def test_repeated_capture_id_never_resends_even_after_restart(self):
        self.workflow.start(settings()); self.start_acknowledger()
        job = self.workflow.submit(picture(), 'one.png', 'same-frame')
        self.done(job['job_id'])
        self.assertTrue(self.workflow.submit(picture(), 'one.png', 'same-frame')['duplicate'])
        self.workflow.close()
        self.workflow = AutomaticWorkflow(self.predictor, self.transfer, self.root/'state')
        self.assertTrue(self.workflow.submit(picture(), 'one.png', 'same-frame')['duplicate'])
        self.assertEqual(len(self.accepted), 1)
        with self.assertRaises(WorkflowError):
            self.workflow.submit(picture((239,240)), 'different.png', 'same-frame')

    def test_wrong_image_dimensions_halts_before_plc_access(self):
        self.workflow.start(settings())
        job = self.workflow.submit(picture((241,240)), 'wrong.png', 'wrong-size')
        result = self.done(job['job_id'])
        self.assertEqual(result['state'], 'failed')
        self.assertIn('尺寸', result['message'])
        self.assertEqual(self.predictor.calls, 0)
        self.assertEqual(self.connection.events, [])
        self.assertFalse(self.workflow.enabled)

    def test_empty_or_over_64_points_halts_without_plc_io(self):
        for count in (0, 65):
            self.workflow.start(settings())
            with patch('auto_workflow.make_plan', return_value=plan(count)):
                job = self.workflow.submit(picture(), 'test.png', f'count-{count}')
                result = self.done(job['job_id'])
            self.assertEqual(result['state'], 'failed')
            self.assertEqual(self.connection.events, [])

    def test_ack_timeout_does_not_withdraw_or_repeat_ready(self):
        self.workflow.start(settings(timeout=1))
        job = self.workflow.submit(picture(), 'timeout.png', 'timeout')
        result = self.done(job['job_id'])
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(result['notification'], 'sent')
        self.assertTrue(self.connection.memory[4] & 1)
        self.assertEqual([e for e in self.connection.events if e[0]=='signal'], [('signal',True)])

    def test_uncertain_commit_halts_and_never_retries(self):
        self.connection.fail_signal = True
        self.workflow.start(settings())
        job = self.workflow.submit(picture(), 'lost.png', 'lost-response')
        result = self.done(job['job_id'])
        self.assertEqual(result['notification'], 'unknown')
        self.assertFalse(self.workflow.enabled)
        self.assertTrue(self.workflow.submit(picture(), 'lost.png', 'lost-response')['duplicate'])
        self.assertEqual([e for e in self.connection.events if e[0]=='signal'], [('signal',True)])

    def test_stop_after_send_cancels_queue_and_does_not_retract_data(self):
        self.workflow.start(settings(timeout=5))
        first = self.workflow.submit(picture(), 'one.png', 'one')
        wait_until(lambda:self.workflow.get_job(first['job_id'])['state']=='waiting_taken')
        second = self.workflow.submit(picture(), 'two.png', 'two')
        self.workflow.stop()
        self.assertEqual(self.done(first['job_id'])['state'], 'cancelled')
        self.assertEqual(self.workflow.get_job(second['job_id'])['state'], 'cancelled')
        self.assertTrue(self.connection.memory[4] & 1)

    def test_interrupted_job_is_not_replayed_on_restart(self):
        self.workflow.start(settings()); self.start_acknowledger()
        job = self.workflow.submit(picture(), 'one.png', 'interrupted')
        self.done(job['job_id']); self.workflow.close()
        path = self.root/'state'/'jobs'/f"{job['job_id']}.json"
        record = json.loads(path.read_text(encoding='utf-8')); record['state']='sending'
        path.write_text(json.dumps(record), encoding='utf-8')
        self.workflow = AutomaticWorkflow(self.predictor, self.transfer, self.root/'state')
        self.assertEqual(self.workflow.get_job(job['job_id'])['state'], 'interrupted')
        self.assertEqual(len(self.accepted), 1)

    def test_parameters_cannot_change_during_active_capture(self):
        self.connection.memory[4] = 1
        self.workflow.start(settings())
        job = self.workflow.submit(picture(), 'one.png', 'active')
        wait_until(lambda:self.workflow.get_job(job['job_id'])['state']=='waiting_idle')
        with self.assertRaises(WorkflowError): self.workflow.start(settings())

    def test_disabled_rejects_new_image(self):
        with self.assertRaises(WorkflowError): self.workflow.submit(picture(), 'one.png', 'disabled')

    def test_invalid_settings_and_capture_id(self):
        for invalid in ({}, dict(settings(),image_width=True), dict(settings(),diameter_mm=0),
                        dict(settings(),plc_timeout_s=0), dict(settings(),transfer_mode='invalid')):
            with self.subTest(settings=invalid), self.assertRaises(ValueError): validate_settings(invalid)
        self.workflow.start(settings())
        for identifier in (None, '../bad', ''):
            with self.assertRaises(ValueError): self.workflow.submit(picture(), 'one.png', identifier)


if __name__ == '__main__':
    unittest.main(verbosity=2)
