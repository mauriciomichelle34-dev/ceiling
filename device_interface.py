"""Future device boundary: coordinate intentions only, no physical I/O."""
from typing import Protocol


class SamplingDevice(Protocol):
    def execute(self, plan: dict) -> None:
        """A future adapter must map frames, check travel, and implement tool actions."""
        ...


class PreviewDevice:
    def preview(self, plan):
        return {'schema_version': 1, 'mode': 'preview_only', 'hardware_ready': False,
                'plan_id': plan['plan_id'], 'units': 'mm', 'coordinate_frame': plan['coordinate_frame'],
                'sampling_order': plan.get('sampling_order', {'strategy': 'legacy_order'}),
                'required_before_execution': ['sample_to_machine_calibration', 'travel_limits',
                    'safe_lift_height', 'controller_protocol', 'tool_sampling_action'],
                'actions': [{'operation': 'sample_at', 'point_id': p['id'],
                             **({'layer': p['layer']} if 'layer' in p else {}),
                             'x_mm': p['x_mm'], 'y_mm': p['y_mm'], 'depth_mm': p['z_mm']}
                            for p in plan['points']]}

    def execute(self, plan):
        raise NotImplementedError('尚未接入硬件；当前只提供取样位置和深度预览。')
