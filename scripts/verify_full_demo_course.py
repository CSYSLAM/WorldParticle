"""Check timeline rotation, split isolation, exact endpoints and window uploads."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    import numpy as np
    import torch
    from wp_vbd.multirate_window import ReferenceGrid
    from wp_vbd.full_demo_course import timeline_catalog,select_timeline_plan,device_window,authored_frame_horizon
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    checked=0
    for dt,count,events in [(1/600,3601,()),(1/60,361,()),(1/1800,5401,(500,1500,3000))]:
        grid=ReferenceGrid(np.arange(count)*dt,((0,count-1),))
        catalog=timeline_catalog(grid,events=events)
        assert not catalog['missing_phase_splits']
        assert {select_timeline_plan(catalog,i)['phase_bin'] for i in range(6)}==set(range(6))
        terminal=select_timeline_plan(catalog,5,complete_motion_sampling=True)
        earlier=select_timeline_plan(catalog,5)
        assert terminal['start']>earlier['start']
        assert select_timeline_plan(catalog,0,complete_motion_sampling=True)['start']==0
        assert select_timeline_plan(catalog,5,development=True,complete_motion_sampling=True)==select_timeline_plan(catalog,5,development=True)
        for split in ('training','development'):
            for plan in catalog[split]['windows']:
                start=plan['start'];end=plan['endpoint_indices'][-1]
                assert abs(sum(plan['durations'])-(grid.times[end]-grid.times[start]))<1e-12
                for first,last in zip([start,*plan['endpoint_indices'][:-1]],plan['endpoint_indices']):
                    assert not any(first<event<last for event in events)
                if split=='training':
                    for block in catalog['reserved_development_blocks']:
                        assert end<block['first'] or start>block['last']
                checked+=1
    # Constant expanded arrays, empty history widths and typed controls retain
    # every temporal row after slicing. Only this window is transferred.
    states={'x':torch.arange(60.).reshape(10,2,3),
            'history__dual':torch.zeros(10,0,6),'input_mask':torch.zeros(9,2,dtype=torch.bool),
            'input_force':torch.ones(1,2,3).expand(9,2,3)}
    window=device_window(states,3,4,'cpu')
    torch.testing.assert_close(window['x'],states['x'][3:8])
    assert window['input_mask'].dtype==torch.bool and window['history__dual'].shape==(5,0,6)
    with tempfile.TemporaryDirectory() as directory:
        source=Path(directory)/'demo.py'
        source.write_text("PARAMS={'num_frames':360, 'unrelated': build_asset()}\nclass Example:\n    FPS=60\n    RAMP_TIME=3.0\n    HOLD_TIME=5.0\nparser.set_defaults(num_frames=int(Example.FPS*(Example.RAMP_TIME+Example.HOLD_TIME))+30)\n")
        assert authored_frame_horizon(source)==510
        source.write_text("PARAMS={'num_frames':360}\nparser.set_defaults(num_frames=PARAMS['num_frames'])\n")
        assert authored_frame_horizon(source)==360
        source.write_text('unrelated_call_never_executes()\n')
        assert authored_frame_horizon(source)==100
    result={'status':'PASS','checked_plans':checked,'six_phase_rotation':True,
            'temporal_split_nonoverlap':True,'hard_event_isolation':True,'typed_window_uploads':True,
            'authored_frame_horizon_without_running_source':True}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
