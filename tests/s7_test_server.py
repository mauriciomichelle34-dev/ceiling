"""Loopback test server with BIT semantics absent from python-snap7 3.3's server.

This is test code only. The unmodified bundled server reduces BIT writes to a
whole-byte assignment; the client emits a correctly addressed S7 BIT telegram.
Capture and validate that telegram, then emulate PLC bit-level assignment.
"""
from s7.server import Server
from s7.datatypes import S7WordLen


class BitAwareServer(Server):
    def __init__(self):
        super().__init__()
        self.bit_writes = []

    def _parse_address_specification(self, raw):
        result = super()._parse_address_specification(raw)
        if result and result['word_len'] == S7WordLen.BIT:
            result['bit_offset'] = int.from_bytes(raw[9:12], 'big') % 8
        return result

    def _parse_write_address(self, request):
        parsed = super()._parse_write_address(request)
        spec = request.get('parameters', {}).get('address_spec', {})
        if parsed and spec.get('word_len') == S7WordLen.BIT:
            area, db, offset, count, data = parsed
            if count != 1 or len(data) != 1 or data[0] not in (0, 1):
                raise ValueError('Test server expects one S7 BIT value')
            bit = spec['bit_offset']
            self.bit_writes.append((db, offset, bit, bool(data[0])))
            key = (area, db)
            with self.area_locks[key]:
                existing = self.memory_areas[key][offset]
                value = existing | (1 << bit) if data[0] else existing & ~(1 << bit)
            return area, db, offset, count, bytearray([value])
        return parsed
