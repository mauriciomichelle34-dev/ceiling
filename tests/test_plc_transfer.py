import copy
from decimal import Decimal
import json
from pathlib import Path
import socket
import struct
import sys
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from plc_transfer import PLCConfig, PLCTransfer, TransferError, prepare_plan, scaled_coordinate


def plan(count=2):
    return {'plan_id': 'b'*32, 'count': count,
            'coordinate_frame': {'xy_origin':'fixed_screw','x_positive':'image_left','y_positive':'image_up'},
            'points': [{'id': i+1, 'x_mm': 12.345 if i % 2 == 0 else -3.215,
                        'y_mm': -0.005 if i % 2 == 0 else 5, 'z_mm': 3}
                       for i in range(count)]}


class FakeConnection:
    def __init__(self, config=None):
        self.memory = bytearray(650)
        for index in range(64):
            self.memory[18+index*10:20+index*10] = struct.pack('>h', index-32)
        self.events = []
        self.fail_write_at = None
        self.corrupt_offset = None
        self.fail_signal = False
        self.flip_feedback_on_signal = False

    def connect(self):
        self.events.append(('connect',))

    def disconnect(self):
        self.events.append(('disconnect',))

    def read(self, offset, size):
        result = bytes(self.memory[offset:offset+size])
        if self.corrupt_offset == offset:
            return bytes([result[0] ^ 1]) + result[1:]
        return result

    def write(self, offset, data):
        self.events.append(('write', offset, bytes(data)))
        if self.fail_write_at == offset:
            raise OSError('injected write failure')
        self.memory[offset:offset+len(data)] = data

    def set_ready(self, value):
        self.events.append(('signal', value))
        if self.flip_feedback_on_signal:
            self.memory[4] |= 4
        self.memory[4] = (self.memory[4] | 1) if value else (self.memory[4] & ~1)
        if self.fail_signal:
            raise OSError('injected lost response after accepting signal')


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.connection = FakeConnection()
        self.transfer = PLCTransfer(connection_factory=lambda config: self.connection)

    def writes(self):
        return [e for e in self.connection.events if e[0] in ('write', 'signal')]

    def test_rounding_sign_and_precision(self):
        for raw, expected in [('12.345', 123), ('-12.345', -123),
                              ('12.35', 124), ('-12.35', -124),
                              ('-0.05', -1), ('0.049', 0), ('1', 10)]:
            self.assertEqual(scaled_coordinate(Decimal(raw), 'X'), expected)

    def test_dint_boundaries(self):
        self.assertEqual(scaled_coordinate(Decimal('-214748364.8'), 'X'), -2**31)
        self.assertEqual(scaled_coordinate(Decimal('214748364.7'), 'X'), 2**31-1)
        for value in [Decimal('214748364.75'), Decimal('-214748364.85'),
                      float('inf'), float('nan'), True, None, '1.2']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                scaled_coordinate(value, 'X')

    def test_validate_entire_plan_before_connection(self):
        for invalid in [plan(0), plan(65), {'points': [None]},
                        {'count': 3, 'points': plan()['points']},
                        {'points': [{'x_mm': 1, 'y_mm': float('nan')}]},
                        {'points': [{'x_mm': 1}]}]:
            with self.subTest(plan=invalid), self.assertRaises(ValueError):
                self.transfer.send(invalid)
        self.assertEqual(self.connection.events, [])

    def test_send_all_64_offsets_preserve_compensation_and_unused_bytes(self):
        self.connection.memory[4] = 0b10110010
        self.connection.memory[5:10] = b'abcde'
        before = bytes(self.connection.memory)
        result = self.transfer.send(plan(64))
        self.assertTrue(result['verified'])
        self.assertEqual(struct.unpack('>i', self.connection.memory[:4])[0], 64)
        self.assertEqual(self.connection.memory[4], 0b10110011)
        self.assertEqual(self.connection.memory[5:10], b'abcde')
        for index in range(64):
            offset = 10 + index * 10
            expected = (123, 0) if index % 2 == 0 else (-32, 50)
            self.assertEqual(struct.unpack('>ii', self.connection.memory[offset:offset+8]), expected)
            self.assertEqual(self.connection.memory[offset+8:offset+10], before[offset+8:offset+10])
        writes = self.writes()
        self.assertEqual(writes[-2], ('write', 0, struct.pack('>i', 64)))
        self.assertEqual(writes[-1], ('signal', True))

    def test_short_batch_does_not_touch_unused_points(self):
        self.connection.memory[30:] = bytes([0x5A]) * 620
        self.transfer.send(plan())
        self.assertEqual(self.connection.memory[30:], bytes([0x5A]) * 620)

    def test_busy_flags_refuse_all_writes(self):
        for flags in (1, 4, 5):
            self.connection.memory[4] = flags
            with self.assertRaises(TransferError):
                self.transfer.send(plan())
        self.assertEqual(self.writes(), [])

    def test_partial_write_never_signals_ready(self):
        self.connection.fail_write_at = 20
        with self.assertRaises(TransferError) as caught:
            self.transfer.send(plan())
        self.assertEqual(caught.exception.notification, 'not_sent')
        self.assertFalse(self.connection.memory[4] & 1)
        self.assertFalse(any(e[0] == 'signal' for e in self.writes()))

    def test_coordinate_readback_mismatch_never_signals(self):
        self.connection.corrupt_offset = 10
        with self.assertRaises(TransferError):
            self.transfer.send(plan())
        self.assertFalse(self.connection.memory[4] & 1)

    def test_count_write_failure_never_signals(self):
        self.connection.fail_write_at = 0
        with self.assertRaises(TransferError):
            self.transfer.send(plan())
        self.assertFalse(self.connection.memory[4] & 1)

    def test_lost_signal_response_reports_unknown_without_retry(self):
        self.connection.fail_signal = True
        with self.assertRaises(TransferError) as caught:
            self.transfer.send(plan())
        self.assertEqual(caught.exception.notification, 'unknown')
        self.assertEqual([e for e in self.writes() if e[0] == 'signal'], [('signal', True)])

    def test_concurrent_feedback_not_overwritten(self):
        self.connection.flip_feedback_on_signal = True
        self.transfer.send(plan())
        self.assertEqual(self.connection.memory[4], 5)

    def test_status_is_read_only(self):
        self.connection.memory[4] = 4
        result = self.transfer.status()
        self.assertTrue(result['plc_taken'])
        self.assertEqual(self.writes(), [])

    def test_release_requires_ack_and_changes_pc_bit_only(self):
        self.connection.memory[4] = 1
        with self.assertRaises(TransferError):
            self.transfer.release()
        self.assertEqual(self.writes(), [])
        self.connection.memory[4] = 0b10100111
        self.transfer.release()
        self.assertEqual(self.connection.memory[4], 0b10100110)

    def test_concurrent_operation_rejected(self):
        self.transfer.lock.acquire()
        try:
            with self.assertRaises(TransferError):
                self.transfer.send(plan())
        finally:
            self.transfer.lock.release()
        self.assertEqual(self.writes(), [])

    def test_config_validation(self):
        for values in ({'host': 'invalid'}, {'db': True}, {'port': 0}, {'slot': 32}):
            with self.assertRaises(ValueError):
                PLCConfig(**values)

    def test_coordinate_test_overwrites_pending_data_and_keeps_sibling_bits(self):
        self.connection.memory[4] = 0b10110111
        before = bytes(self.connection.memory)
        result = self.transfer.write_coordinates(plan(2))
        self.assertEqual(result['mode'], 'coordinates_only')
        self.assertEqual(result['notification'], 'not_sent')
        self.assertEqual(self.connection.memory[4], 0b10110110)
        self.assertEqual(self.writes()[0], ('signal', False))
        self.assertEqual(self.connection.memory[18:20], before[18:20])
        smaller = dict(plan(1), points=[{'x_mm':7.89,'y_mm':-2.34}])
        self.transfer.write_coordinates(smaller)
        self.assertEqual(struct.unpack('>ii',self.connection.memory[10:18]),(79,-23))
        self.assertEqual(struct.unpack('>i',self.connection.memory[:4])[0],1)
        self.assertEqual([e for e in self.writes() if e[0]=='signal'], [('signal',False),('signal',False)])

    def test_coordinate_test_validates_before_clearing_flag(self):
        self.connection.memory[4] = 1
        with self.assertRaises(ValueError): self.transfer.write_coordinates(plan(65))
        self.assertEqual(self.connection.memory[4], 1)
        self.assertEqual(self.connection.events, [])

    def test_coordinate_test_cannot_write_if_ready_cannot_be_cleared(self):
        self.connection.memory[4] = 1
        self.connection.set_ready = lambda value: None
        with self.assertRaises(TransferError): self.transfer.write_coordinates(plan())
        self.assertFalse(any(e[0]=='write' for e in self.connection.events))

    def test_coordinate_test_failure_never_signals_true(self):
        self.connection.memory[4] = 1
        self.connection.fail_write_at = 20
        with self.assertRaises(TransferError): self.transfer.write_coordinates(plan())
        self.assertFalse(self.connection.memory[4] & 1)
        self.assertEqual([e for e in self.writes() if e[0]=='signal'], [('signal',False)])

    def test_coordinate_test_readback_mismatch_keeps_ready_low(self):
        self.connection.corrupt_offset = 10
        with self.assertRaises(TransferError): self.transfer.write_coordinates(plan())
        self.assertEqual([e for e in self.writes() if e[0]=='signal'], [('signal',False)])


class RealS7Tests(unittest.TestCase):
    def test_coordinate_test_overwrite_uses_only_false_bit_telegrams(self):
        from s7_test_server import BitAwareServer
        from s7.type import SrvArea
        server = BitAwareServer()
        memory = bytearray(650); memory[4]=5
        server.register_area(SrvArea.DB,100,memory)
        with socket.socket() as reserve:
            reserve.bind(('127.0.0.1',0)); port=reserve.getsockname()[1]
        try:
            server.start_to('127.0.0.1',port)
            transfer=PLCTransfer(PLCConfig(host='127.0.0.1',port=port))
            transfer.write_coordinates(plan(2))
            transfer.write_coordinates(dict(plan(1), points=[{'x_mm':8,'y_mm':-9}]))
            self.assertEqual(struct.unpack('>ii',memory[10:18]),(80,-90))
            self.assertEqual(struct.unpack('>i',memory[:4])[0],1)
            self.assertEqual(memory[4],4)
            self.assertEqual(server.bit_writes,[(100,4,0,False),(100,4,0,False)])
        finally:
            server.stop(); server.destroy()

    def test_loopback_s7_transfer_and_atomic_bit_write(self):
        from s7_test_server import BitAwareServer
        from s7.type import SrvArea
        server = BitAwareServer()
        memory = bytearray(650)
        memory[4] = 0b10110010
        for index in range(64):
            memory[18+index*10:20+index*10] = struct.pack('>h', index+100)
        server.register_area(SrvArea.DB, 100, memory)
        with socket.socket() as reserve:
            reserve.bind(('127.0.0.1', 0))
            port = reserve.getsockname()[1]
        try:
            server.start_to('127.0.0.1', port)
            transfer = PLCTransfer(PLCConfig(host='127.0.0.1', port=port))
            result = transfer.send(plan(64))
            self.assertTrue(result['verified'])
            self.assertEqual(memory[4], 0b10110011)
            self.assertEqual(struct.unpack('>ii', memory[640:648]), (-32, 50))
            self.assertEqual(struct.unpack('>h', memory[648:650])[0], 163)
            self.assertEqual(transfer.status()['count'], 64)
            memory[4] |= 4
            transfer.release()
            self.assertEqual(memory[4], 0b10110110)
            self.assertEqual(server.bit_writes, [(100, 4, 0, True), (100, 4, 0, False)])
        finally:
            server.stop()
            server.destroy()


if __name__ == '__main__':
    unittest.main(verbosity=2)
