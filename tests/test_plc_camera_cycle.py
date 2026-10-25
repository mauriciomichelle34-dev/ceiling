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
from urllib.error import HTTPError
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from auto_workflow import AutomaticWorkflow,WorkflowError,validate_settings
from camera_service import CameraConfig,CameraService,CameraError
from plc_transfer import PLCTransfer,TransferError
import test_auto_workflow as aw
import test_plc_http as http_fixture
from test_plc_transfer import FakeConnection,plan
from test_camera_service import FakeAdapter


def cycle_settings(timeout=3):
    return dict(aw.settings(),input_source='plc_camera',transfer_mode='execute',cycle_timeout_s=timeout)


class CameraCycleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.connection=FakeConnection()
        self.transfer=PLCTransfer(connection_factory=lambda config:self.connection)
        self.predictor=aw.TestPredictor(self.root/'results')
        self.camera=CameraService(CameraConfig(),self.root/'camera',FakeAdapter)
        self.camera.search();self.camera.connect('SIM-1')
        self.workflow=AutomaticWorkflow(self.predictor,self.transfer,self.root/'state',camera=self.camera)

    def tearDown(self):
        self.workflow.close();self.camera.close();self.temp.cleanup()

    def job(self):
        return aw.wait_until(lambda:self.workflow.snapshot()['jobs'][0] if self.workflow.snapshot()['jobs'] else None)

    def wait_job(self,identifier,state):
        return aw.wait_until(lambda:self.workflow.get_job(identifier) if self.workflow.get_job(identifier)['state']==state else None)

    def signal_writes(self):
        return [event for event in self.connection.events if event[0] in ('write','signal')]

    def test_five_complete_cycles_keep_one_photo_per_ready_and_plc_resets_both_bits(self):
        self.connection.memory[4]=0b10110000
        original=bytes(self.connection.memory)
        self.workflow.start(cycle_settings())
        for index in range(5):
            self.connection.memory[4] |= 4
            aw.wait_until(lambda:len(self.workflow.jobs)==index+1)
            identifier=self.workflow.snapshot()['jobs'][0]['job_id']
            result=self.wait_job(identifier,'waiting_cycle_finish')
            self.assertEqual(self.camera.adapter.shots,index+1)
            self.assertEqual(self.connection.memory[4]&5,5)
            self.assertEqual(struct.unpack('>i',self.connection.memory[:4])[0],result['count'])
            for point in result['plan']['points']:
                offset=10+(point['id']-1)*10
                self.assertEqual(self.connection.memory[offset+8:offset+10],original[offset+8:offset+10])
            time.sleep(.25)
            self.assertEqual(self.camera.adapter.shots,index+1)
            self.connection.memory[4] &= ~5  # The PLC owns the end-of-cycle reset.
            self.wait_job(identifier,'completed')
        self.assertEqual([event for event in self.signal_writes() if event[0]=='signal'],[('signal',True)]*5)
        self.assertEqual(self.connection.memory[4],0b10110000)
        self.assertEqual(len({job['capture_id'] for job in self.workflow.jobs.values()}),5)

    def test_ready_already_true_at_start_is_processed_once(self):
        self.connection.memory[4]=4
        self.workflow.start(cycle_settings())
        identifier=self.job()['job_id']
        self.wait_job(identifier,'waiting_cycle_finish')
        time.sleep(.3)
        self.assertEqual(self.camera.adapter.shots,1)

    def test_stale_pc_completion_refuses_start_without_clearing_it(self):
        self.connection.memory[4]=5
        with self.assertRaisesRegex(WorkflowError,'4.0'):
            self.workflow.start(cycle_settings())
        self.assertEqual(self.connection.memory[4],5)
        self.assertEqual(self.camera.adapter.shots,0)
        self.assertEqual(self.signal_writes(),[])

    def test_camera_must_be_connected_and_test_only_mode_is_rejected(self):
        self.camera.disconnect()
        with self.assertRaisesRegex(WorkflowError,'连接海康相机'):
            self.workflow.start(cycle_settings())
        with self.assertRaises(ValueError):
            validate_settings(dict(cycle_settings(),transfer_mode='coordinates_only'))

    def test_startup_plc_error_is_explained_without_enabling_or_capturing(self):
        with patch.object(self.transfer,'status',side_effect=TransferError('连接失败')):
            with self.assertRaisesRegex(WorkflowError,'启动前读取 PLC 失败'):
                self.workflow.start(cycle_settings())
        self.assertFalse(self.workflow.enabled)
        self.assertEqual(self.camera.adapter.shots,0)
        self.assertEqual(self.signal_writes(),[])

    def test_camera_error_stops_listener_and_is_not_retried(self):
        with patch.object(self.camera.adapter,'capture',side_effect=CameraError('相机超时')) as capture:
            self.connection.memory[4]=4
            self.workflow.start(cycle_settings())
            aw.wait_until(lambda:not self.workflow.enabled)
            time.sleep(.3)
            self.assertEqual(capture.call_count,1)
        self.assertIn('相机超时',self.workflow.snapshot()['last_error'])
        self.assertEqual(self.signal_writes(),[])

    def test_prediction_error_keeps_raw_photo_without_writing_plc(self):
        with patch.object(self.predictor,'predict',side_effect=ValueError('识别未通过')):
            self.connection.memory[4]=4
            self.workflow.start(cycle_settings())
            identifier=self.job()['job_id']
            result=self.wait_job(identifier,'failed')
        self.assertIn('识别未通过',result['message'])
        self.assertEqual(len(list((self.root/'camera').glob('*.png'))),1)
        self.assertEqual(self.signal_writes(),[])

    def test_over_capacity_rejected_without_partial_plc_writes(self):
        with patch('auto_workflow.make_plan',return_value=plan(65)):
            self.connection.memory[4]=4
            self.workflow.start(cycle_settings())
            self.wait_job(self.job()['job_id'],'failed')
        self.assertEqual(self.signal_writes(),[])

    def test_lost_completion_response_does_not_resend_or_take_second_photo(self):
        self.connection.fail_signal=True
        self.connection.memory[4]=4
        self.workflow.start(cycle_settings())
        result=self.wait_job(self.job()['job_id'],'failed')
        self.assertEqual(result['notification'],'unknown')
        time.sleep(.3)
        self.assertEqual(self.camera.adapter.shots,1)
        self.assertEqual([e for e in self.signal_writes() if e[0]=='signal'],[('signal',True)])

    def test_plc_ready_disappearing_during_recognition_prevents_transfer(self):
        predict=self.predictor.predict
        def ready_lost(*args):
            result=predict(*args)
            self.connection.memory[4] &= ~4
            return result
        with patch.object(self.predictor,'predict',side_effect=ready_lost):
            self.connection.memory[4]=4
            self.workflow.start(cycle_settings())
            self.wait_job(self.job()['job_id'],'failed')
        self.assertEqual(self.signal_writes(),[])

    def test_stop_during_capture_keeps_photo_but_submits_no_job(self):
        entered,finish=threading.Event(),threading.Event()
        original=self.camera.adapter.capture
        def delayed_capture():
            entered.set();finish.wait(3);return original()
        with patch.object(self.camera.adapter,'capture',side_effect=delayed_capture):
            self.connection.memory[4]=4
            self.workflow.start(cycle_settings())
            self.assertTrue(entered.wait(3))
            self.workflow.stop()
            with self.assertRaises(WorkflowError):self.workflow.start(cycle_settings())
            finish.set()
            aw.wait_until(lambda:not self.workflow.camera_busy)
        self.assertFalse(self.workflow.jobs)
        self.assertEqual(self.signal_writes(),[])
        self.assertTrue(list((self.root/'camera').glob('*.png')))

    def test_stop_after_sending_does_not_clear_device_handshake(self):
        self.connection.memory[4]=4
        self.workflow.start(cycle_settings())
        identifier=self.job()['job_id']
        self.wait_job(identifier,'waiting_cycle_finish')
        self.workflow.stop()
        self.wait_job(identifier,'cancelled')
        self.assertEqual(self.connection.memory[4]&5,5)

    def test_cycle_reset_timeout_stops_without_repeating_capture(self):
        self.connection.memory[4]=4
        self.workflow.start(cycle_settings(timeout=1))
        result=self.wait_job(self.job()['job_id'],'failed')
        self.assertIn('超时',result['message'])
        self.assertEqual(self.connection.memory[4]&5,5)
        time.sleep(.25)
        self.assertEqual(self.camera.adapter.shots,1)

    def test_three_failed_reads_stop_monitor_without_camera_or_plc_write(self):
        self.workflow.start(cycle_settings())
        with patch.object(self.transfer,'status',side_effect=OSError('PLC 离线')):
            aw.wait_until(lambda:not self.workflow.enabled)
        self.assertIn('连续三次',self.workflow.snapshot()['last_error'])
        self.assertEqual(self.camera.adapter.shots,0)
        self.assertEqual(self.signal_writes(),[])

    def test_restart_restores_parameters_but_never_replays_ready_automatically(self):
        self.workflow.start(cycle_settings())
        self.workflow.close()
        self.connection.memory[4]=4
        self.workflow=AutomaticWorkflow(self.predictor,self.transfer,self.root/'state',camera=self.camera)
        time.sleep(.3)
        self.assertFalse(self.workflow.enabled)
        self.assertEqual(self.workflow.settings['input_source'],'plc_camera')
        self.assertEqual(self.camera.adapter.shots,0)


class CameraCycleHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        http_fixture.HTTPTests.setUpClass()
        cls.fixture=http_fixture.HTTPTests
        cls.http,cls.base=cls.fixture.http,cls.fixture.base
        cls.http.predictor=aw.TestPredictor(cls.fixture.results)
        cls.http.camera=CameraService(CameraConfig(),cls.fixture.results/'camera',FakeAdapter)
        cls.http.camera.search();cls.http.camera.connect('SIM-1')
        cls.http.automatic=AutomaticWorkflow(cls.http.predictor,cls.http.plc_transfer,cls.fixture.results/'auto',camera=cls.http.camera)

    @classmethod
    def tearDownClass(cls):
        cls.http.automatic.close();cls.http.camera.close();cls.fixture.tearDownClass()

    def post(self,path,body):return self.fixture.post(self,path,body)

    def test_manual_camera_and_image_inputs_locked_during_plc_cycle(self):
        self.assertEqual(self.post('/api/automatic/start',{'settings':cycle_settings()})[0],200)
        for action in ('capture','disconnect','search','connect'):
            self.assertEqual(self.post('/api/camera/'+action,{'capture_id':'manual','device_id':'SIM-1'})[0],409)
        request=Request(self.base+'/api/automatic/images',data=aw.picture(),headers={
            'Content-Type':'image/png','X-PLC-Request':'1','X-Capture-ID':'manual-upload'})
        with self.assertRaises(HTTPError) as response:urlopen(request)
        self.assertEqual(response.exception.code,409)
        self.assertEqual(self.http.camera.adapter.shots,0)
        self.assertEqual(self.post('/api/automatic/stop',{})[0],200)
        self.fixture.memory[4]=5
        self.assertEqual(self.post('/api/plc/release',{})[0],409)
        self.assertEqual(self.fixture.memory[4],5)
        self.fixture.memory[4]=0
        self.assertEqual(self.post('/api/camera/capture',{'capture_id':'manual-after-stop'})[0],200)


if __name__=='__main__':unittest.main(verbosity=2)
