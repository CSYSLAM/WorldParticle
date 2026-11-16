"""Convert a screened, fully audited native grasp tape to causal training data."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),'/home/oem/code/engine/newton']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import json
    import torch
    import warp as wp
    import newton
    sys.path.append('/home/oem/code/engine/newton/.venv/lib/python3.12/site-packages')
    torch.set_num_threads(1);wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device('cpu')
    def forbidden(*unused,**kwargs):raise AssertionError('Data conversion must not construct or execute a dynamics solver')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):cls.__init__=forbidden;cls.step=forbidden
    from wp_vbd.franka_reference_data import prepare,load_reference
    result=prepare(args.recording,args.review,args.output,wp,newton)
    scene,states=load_reference(args.output,wp,newton)
    assert len(states['x'])==result['physical_steps']+1 and scene.mimic_spec is not None
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
