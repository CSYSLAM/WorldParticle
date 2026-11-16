"""Create an immutable operator prototype without changing trained NN weights."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=ROOT/'checkpoints/shared_cycle_v10/shared_strict.pt')
    parser.add_argument('--variant',choices=('linear','material','velocity','chain','curvature','precision','flexure'),required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--report-dir',default='reports/rod_boundary_v11')
    args=parser.parse_args()
    if args.checkpoint.exists() or args.config.exists():raise FileExistsError('Use new paths to preserve existing experiment evidence')
    parent=torch.load(args.parent,weights_only=True,map_location='cpu');probe=copy.deepcopy(parent)
    flags={'rod_boundary_lift':True,'rod_material_twist_modes':args.variant!='linear','rod_material_velocity':args.variant in ('velocity','chain','curvature','precision','flexure')}
    flags['rod_chain_shear_modes']=args.variant in ('chain','curvature','precision','flexure')
    flags['rod_curvature_ports']=args.variant in ('curvature','precision','flexure')
    flags['rod_geometry_fp64']=args.variant in ('precision','flexure')
    flags['rod_curvature_flexure']=args.variant=='flexure'
    probe['step']=0;probe['architecture']='shared_boundary_v11';probe['config']['name']='shared_boundary_'+args.variant+'_probe'
    probe['config']['model'].update(flags)
    if args.variant=='chain':probe['architecture']='shared_chain_v12'
    if args.variant=='curvature':probe['architecture']='shared_curvature_v13'
    if args.variant=='precision':probe['architecture']='shared_precision_v14'
    if args.variant=='flexure':probe['architecture']='shared_flexure_v15'
    probe['config'].update(initialize_from=str(args.parent.resolve()),report_dir=args.report_dir,checkpoint_dir=str(args.checkpoint.parent))
    probe['operator_probe_provenance']={'status':'UNTRAINED_ARCHITECTURE_PROTOTYPE','parent_checkpoint_sha256':hashlib.sha256(args.parent.read_bytes()).hexdigest(),'parent_selected_update':parent['step'],'operator_stage_updates':0,'weights_changed':False,'new_neural_parameters':0}
    probe['config']['scope']['operator_probe']='Runtime/geometry experiment only; no new optimizer updates are claimed'
    assert all(torch.equal(value,parent['model'][key]) for key,value in probe['model'].items())
    args.checkpoint.parent.mkdir(parents=True,exist_ok=True);args.config.parent.mkdir(parents=True,exist_ok=True)
    torch.save(probe,args.checkpoint);args.config.write_text(json.dumps(probe['config'],indent=2)+'\n')
    print(json.dumps({'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),**probe['operator_probe_provenance']},indent=2))


if __name__=='__main__':main()
