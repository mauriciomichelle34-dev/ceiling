"""Exercise DCS3 framing and cleanup with simulated sockets/camera only."""
from pathlib import Path
import socket
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from camera_service import CameraService, CameraConfig, CameraError
from light_service import LightConfig, LightService, LightError, command_frame, read_value
import test_camera_service as native_fixtures


class SimulatedLight:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.brightness = 75
        self.on = False
        self.off_failures = 0
        self.on_failure = False
        self.corrupt_read = False
        self.reject_on = False
        self.connections = 0

    def connect(self, address, timeout):
        self.connections += 1
        return SimulatedSocket(self)


class SimulatedSocket:
    def __init__(self, state):
        self.state, self.buffer, self.fail, self.closed = state, bytearray(), False, False
    def settimeout(self, value):
        pass
    def close(self):
        self.closed = True
    def sendall(self, frame):
        if self.closed:
            raise OSError('closed socket')
        command, channel, value = chr(frame[1]), int(chr(frame[2])), int(frame[3:6], 16)
        if frame != command_frame(command, channel, value):
            raise AssertionError('bad request checksum')
        state = self.state
        state.events.append(('light', command, value))
        self.fail = False
        if command == '4':
            reply = command_frame('4', channel, state.brightness)
            self.buffer = bytearray(reply[:-1] + b'Z' if state.corrupt_read else reply)
        elif command == '3':
            state.brightness = value
            self.buffer = bytearray(b'$')
        elif command == '1':
            state.on = True
            self.fail = state.on_failure
            self.buffer = bytearray(b'&' if state.reject_on else b'$')
        elif command == '2':
            self.fail = state.off_failures > 0
            state.off_failures = max(0, state.off_failures - 1)
            if not self.fail:
                state.on = False
            self.buffer = bytearray(b'$')
        else:
            raise AssertionError(command)
    def recv(self, size):
        if self.fail:
            raise socket.timeout('simulated timeout')
        part = bytes(self.buffer[:min(size, 2)])
        del self.buffer[:len(part)]
        return part


class ProtocolTests(unittest.TestCase):
    def test_manual_examples_match_byte_for_byte(self):
        for args, expected in [(('3', 1, 100), b'$3106414'),
                               (('2', 2, 41), b'$220291F'),
                               (('1', 3, 100), b'$1306414'),
                               (('4', 2, 100), b'$4206410')]:
            self.assertEqual(command_frame(*args), expected)
        self.assertEqual(read_value(b'$4206410', '4', 2), 100)
        for reply in (b'$4206411', b'$4106413', b'&', b'$42GGG10'):
            with self.assertRaises(LightError):
                read_value(reply, '4', 2)

    def test_probe_reads_fragmented_reply_without_switching_light(self):
        state = SimulatedLight()
        light = LightService(LightConfig(), connector=state.connect, sleeper=lambda _: None)
        status = light.probe()
        self.assertEqual(status['actual_brightness'], 75)
        self.assertTrue(status['connected'])
        self.assertEqual(state.events, [('light', '4', 0)])
        self.assertFalse(state.on)

    def test_configuration_persists_without_any_network_operation(self):
        state = SimulatedLight()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'light.json'
            light = LightService(LightConfig(), path, connector=state.connect)
            light.configure({'enabled': False, 'brightness': 123})
            self.assertEqual(LightConfig.load(path).brightness, 123)
            self.assertFalse(LightConfig.load(path).enabled)
            self.assertEqual(state.connections, 0)
            for body in ({'enabled': True, 'brightness': 0}, {'enabled': 1, 'brightness': 100},
                         {'enabled': True, 'brightness': 100.5}, {'host': '127.0.0.1'}):
                with self.assertRaises(ValueError):
                    light.configure(body)
            self.assertEqual(LightConfig.load(path).brightness, 123)


class CaptureLightTests(unittest.TestCase):
    def setUp(self):
        native_fixture = native_fixtures.NativeSequenceTests()
        native_fixture.setUp()
        self.native, self.adapter = native_fixture.native, native_fixture.adapter
        self.adapter.version = 'simulated'
        self.state = SimulatedLight(self.native.events)
        self.light = LightService(LightConfig(brightness=100), connector=self.state.connect,
                                  sleeper=lambda seconds: self.native.events.append(('delay', seconds)))
        convert = self.adapter._frame_to_png
        def encode(frame):
            self.assertFalse(self.state.on, 'light must be off before image conversion')
            self.native.events.append(('encode',))
            return convert(frame)
        self.adapter._frame_to_png = encode

    def test_light_brackets_trigger_and_frame_but_not_conversion(self):
        raw, info = self.adapter.capture(illumination=self.light)
        events = self.native.events
        self.assertLess(events.index(('clear',)), events.index(('light', '1', 0)))
        self.assertLess(events.index(('light', '1', 0)), events.index(('delay', .1)))
        self.assertLess(events.index(('delay', .1)), events.index(('command', 'TriggerSoftware')))
        self.assertLess(events.index(('get-frame', 10000)), events.index(('light', '2', 0)))
        self.assertLess(events.index(('light', '2', 0)), events.index(('encode',)))
        self.assertTrue(info['illumination']['off_confirmed'])
        self.assertEqual(self.state.brightness, 100)

    def test_camera_timeout_still_turns_off_and_restores_camera(self):
        self.native.timeout = True
        before = dict(self.native.values)
        with self.assertRaises(CameraError):
            self.adapter.capture(illumination=self.light)
        self.assertFalse(self.state.on)
        self.assertTrue(self.light.status()['off_confirmed'])
        self.assertEqual(self.native.values, before)
        self.assertNotIn(('encode',), self.native.events)

    def test_uncertain_on_ack_gets_off_attempt_and_never_triggers_camera(self):
        self.state.on_failure = True
        with self.assertRaises(LightError):
            self.adapter.capture(illumination=self.light)
        self.assertFalse(self.state.on)
        self.assertIn(('light', '2', 0), self.native.events)
        self.assertNotIn(('command', 'TriggerSoftware'), self.native.events)

    def test_corrupt_read_or_nack_prevents_camera_trigger(self):
        for field in ('corrupt_read', 'reject_on'):
            with self.subTest(field=field):
                self.setUp()
                setattr(self.state, field, True)
                with self.assertRaises(LightError):
                    self.adapter.capture(illumination=self.light)
                self.assertFalse(self.state.on)
                self.assertNotIn(('command', 'TriggerSoftware'), self.native.events)

    def test_off_retry_is_bounded_and_successful_over_new_socket(self):
        self.state.off_failures = 1
        self.adapter.capture(illumination=self.light)
        self.assertFalse(self.state.on)
        self.assertEqual(self.state.connections, 2)
        self.assertEqual(self.native.events.count(('light', '2', 0)), 2)

    def test_off_failure_blocks_encoding_and_downstream_processing(self):
        self.state.off_failures = 2
        processed = []
        with tempfile.TemporaryDirectory() as temp:
            service = CameraService(CameraConfig(), temp, illumination=self.light)
            service.adapter, service.device = self.adapter, {'id': 'SIM'}
            with self.assertRaisesRegex(LightError, '未能确认关灯'):
                service.capture('failed-off', 'analysis', lambda *args: processed.append(args))
            self.assertEqual(processed, [])
            self.assertEqual(list(Path(temp).glob('*.json')), [])
        self.assertFalse(self.light.status()['off_confirmed'])
        self.assertIn(('free',), self.native.events)
        self.assertNotIn(('encode',), self.native.events)

    def test_disabled_light_does_no_io_and_capture_still_works(self):
        self.light.configure({'enabled': False, 'brightness': 100})
        self.adapter.capture(illumination=self.light)
        self.assertEqual(self.state.connections, 0)
        self.assertFalse(any(event[0] == 'light' for event in self.native.events))

    def test_duplicate_capture_does_not_repeat_light_or_processing(self):
        processed = []
        with tempfile.TemporaryDirectory() as temp:
            service = CameraService(CameraConfig(), temp, illumination=self.light)
            service.adapter, service.device = self.adapter, {'id': 'SIM'}
            first = service.capture('once', 'analysis', lambda *args: processed.append(args) or {})
            second = service.capture('once', 'analysis', lambda *args: processed.append(args) or {})
            self.assertEqual(first, second)
            self.assertEqual(len(processed), 1)
            self.assertTrue(first['illumination']['off_confirmed'])
        self.assertEqual(self.native.events.count(('light', '1', 0)), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
