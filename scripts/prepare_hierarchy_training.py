"""Initialize, admit and freeze one multi-type hierarchical experiment."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from wp_vbd.mixed_solver import MixedSolver

DIRECTORY=ROOT/'reports/shared_hierarchy_v20'
CHECKPOINTS=ROOT/'checkpoints/shared_hierarchy_v20'
CONFIG=ROOT/'configs/shared_hierarchy_v20.json'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize():
    path=CHECKPOINTS/'architecture_probe.pt'
    if path.exists() or CONFIG.exists():raise FileExistsError('Keep initialization and the declared experiment immutable')
    parent=ROOT/'checkpoints/shared_vector_v19/shared_strict.pt';stored=torch.load(parent,weights_only=True,map_location='cpu')
    cfg=copy.deepcopy(stored['config']);cfg.update(name='shared_hierarchy_v20',seed=20261108,report_dir='reports/shared_hierarchy_v20',checkpoint_dir='checkpoints/shared_hierarchy_v20',initialize_from=str(path))
    cfg['model'].update(shared_hierarchy=True,maximum_super_tokens=128)
    cfg['training'].update(steps=128,learning_rate=1e-4,validation_interval=64,batch_seed=20261108,max_seconds=3600,record_sample_history=True)
    cfg['training']['validation_windows_by_demo']['vbd.example_cable_bend_twist_analytic']={'starts':[0,1080,2280,3780],'steps':120}
    for key in ('selected_stage_update','completed_stage_updates','selected_update','completed_updates'):cfg['scope'].pop(key,None)
    cfg['scope'].update(shared_hierarchy='Four type tokenizers, material/relation FiLM, three shared coarse attention blocks and two full-resolution cross decoders with hyperedge bypass. Only latent features merge; inherited physical/vector response budgets remain unchanged.',
        training_curriculum='Warm-start v19 and add a zero-initialized typed hierarchy readout; train the complete protected response on qualified multi-type official episodes and five-type replay.',
        official_coverage='Derived from this run and selected-checkpoint sampling counts. All 51 demos are integration targets; unsupported operators are never silently removed.',
        teacher_target='Qualified same-h multi-type and analytic rod reference pairs listed in data_manifest.json, plus prior transfer and finite mixed replay. Preliminary trajectory admission, not full P1 certification.',
        validation_selection='Per-object mixed/new-official endpoint or mean of the four rod-window endpoints, with predeclared 5% regression caps. No individual-window or full-trajectory guarantee.',
        test_isolation='Old material1.33 and mixed09 are development. Newly frozen held-out materials are separate from optimizer and selection.',
        acceptance='Architecture expansion experiment. Reserved dynamic relation kinds do not admit missing physics/history/geometry or external solver operators.',default_promoted=False)
    torch.manual_seed(cfg['seed']);model=MixedSolver(cfg['model'])
    missing=model.load_state_dict(stored['model'],strict=False)
    assert not missing.unexpected_keys and all(key.startswith(('protected_response.hierarchy.','protected_response.transport.')) for key in missing.missing_keys)
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in stored['model'].items())
    DIRECTORY.mkdir(parents=True,exist_ok=True);CHECKPOINTS.mkdir(parents=True,exist_ok=True)
    CONFIG.write_text(json.dumps(cfg,indent=2)+'\n')
    torch.save({'architecture':model.architecture_id,'model':model.state_dict(),'config':cfg,'step':0},path)
    record={'status':'UNTRAINED_HIERARCHY_AUGMENTATION','checkpoint_sha256':sha(path),'parent_checkpoint_sha256':sha(parent),'new_optimizer_updates':0,'active_parameters':sum(p.numel() for p in model.protected_response.parameters()),'new_hierarchy_parameters':sum(p.numel() for p in model.protected_response.hierarchy.parameters()),'new_vector_transport_parameters':sum(p.numel() for p in model.protected_response.transport.parameters()),'inherited_tensors_exact':True}
    (DIRECTORY/'initialization.json').write_text(json.dumps(record,indent=2)+'\n')
    holdout={'frozen_before_optimization':True,'newton_sha':cfg['newton_sha'],'official_material_factors':{'softbody.example_softbody_hanging':1.07,'vbd.example_cloth_stiff_material_hanging':1.07,'basic.example_basic_dzhanibekov':1.07,'vbd.example_cable_twist_transfer':1.41},'scope':'Separate future material test protocol. No claim of new topology or complete geometry acceptance. References must pass the same fixed-h qualification; no model test is used for weight selection.'}
    (ROOT/'configs/shared_hierarchy_holdout_v20.json').write_text(json.dumps(holdout,indent=2)+'\n')
    print(json.dumps(record,indent=2))


def admit():
    if (DIRECTORY/'training.json').exists():raise FileExistsError('Keep optimizer evidence immutable')
    operator=read(DIRECTORY/'integration_cpu.json');assert operator['status']=='PASS'
    for name in ('backbone_cpu','backbone_cuda','vector_transport_cpu','vector_transport_cuda','runtime_initial','window_verification','training_graph_verification'):
        assert read(DIRECTORY/(name+'.json'))['status']=='PASS'
    rows=copy.deepcopy(read(ROOT/'reports/shared_vector_v19/data_manifest.json')['scenes'])
    reference=read(ROOT/'configs/shared_multitype_reference_v20.json');excluded=[]
    for family in reference['families']:
        family_rows=[]
        for episode in reference['episodes']:
            record=read(ROOT/reference['report_dir']/(family['name']+'_'+episode['name']+'_refinement.json'))
            if family['name'] in ('cloth','rigid'):
                query_path=ROOT/'reports/qualified_windows_v20'/(record['name']+'.json')
                query=read(query_path)
                assert query['status']=='PASS' and query['all_windows_qualified'] and sha(ROOT/query['path'])==query['sha256']
                family_rows.append({'dataset_id':family['demo']+'/qualified_'+episode['name'],'demo':family['demo'],'path':query['path'],'sha256':query['sha256'],'split':episode['split'],'status':'PASS','parameter_factor':episode['factor'],'teacher_refinement_report':str(query_path.relative_to(ROOT)),'data_layout':'teacher_queried_windows_v1','excluded_full_episode_refinement':record})
                continue
            good=record['within_refinement_limits'] and record['same_static_and_current_inputs'] and record['finite'] and record['original_terminal_checks_passed'] and (record['minimum_detF'] is None or record['minimum_detF']>=.01)
            if not good:excluded.append({'name':record['name'],'refinement':record});continue
            source=record['sources'][-1];assert sha(ROOT/source['path'])==source['sha256']
            family_rows.append({'dataset_id':family['demo']+'/refined_'+episode['name'],'demo':family['demo'],'path':source['path'],'sha256':source['sha256'],'split':episode['split'],'status':'PASS','parameter_factor':episode['factor'],'teacher_refinement_report':str((ROOT/reference['report_dir']/(record['name']+'_refinement.json')).relative_to(ROOT))})
        if not any(row['split']=='train' for row in family_rows) or not any(row['split']=='validation' for row in family_rows):
            (DIRECTORY/'reference_admission_failure.json').write_text(json.dumps({'status':'FAIL','family':family['name'],'excluded':excluded,'scope':'Do not train unqualified source labels or discard a required validation split.'},indent=2)+'\n')
            raise RuntimeError('Required family reference qualification failed: '+family['name'])
        rows.extend(family_rows)
    analytic=read(ROOT/'configs/refined_analytic_data_v20.json')
    for episode in analytic['episodes']:
        record=read(ROOT/analytic['report_dir']/(episode['name']+'_episode.json'))
        assert record['within_refinement_limits']
        source=record['sources'][-1];assert source['terminal_check']=='PASS' and sha(ROOT/source['path'])==source['sha256']
        rows.append({'dataset_id':analytic['demo']+'/refined_'+episode['name'],'demo':analytic['demo'],'path':source['path'],'sha256':source['sha256'],'split':episode['split'],'status':'PASS','parameter_factor':episode['factor'],'history_contract':'rod_compliant_alm_v1','teacher_refinement_report':record['refinement_report']})
    for index,row in enumerate(rows):row['index']=index
    manifest={'status':'PASS','scenes':rows,'newton_sha':reference['newton_sha'],'active_architecture':'shared_hierarchy_v20','all_51_training_targets':'reports/all_vbd_integration_v20/registry.json','teacher_admission_scope':'Preliminary P4 experiment: refined complete softbody/rod episodes and explicitly disjoint two-step same-input queried cloth/rigid windows. Failed full-trajectory refinement remains excluded as supervision. Not full P1 continuum/time-step or high-stiffness arithmetic certification.','excluded_reference_episodes':excluded,'historical_original_demo_training_count':10,'all_51_demo_acceptance':False}
    (DIRECTORY/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (DIRECTORY/'verification.json').write_text(json.dumps({'status':'PASS','initial_checkpoint_sha256':sha(CHECKPOINTS/'architecture_probe.pt'),'source_checks':['integration_cpu','backbone_cpu','backbone_cuda','runtime_initial','training_graph_verification','window_verification'],'scope':'Admit one bounded shared experiment; dynamic contacts, self-contact, general joint ALM and external exchange remain unimplemented.'},indent=2)+'\n')
    print(json.dumps({'status':'ADMITTED_TO_BOUNDED_TRAINING','official_train_demos':sorted({row['demo'] for row in rows if row['split']=='train' and row.get('demo')}),'split_counts':{split:sum(row['split']==split for row in rows) for split in ('train','validation','test')}},indent=2))


def freeze():
    target=CHECKPOINTS/'shared_strict.pt'
    if target.exists():raise FileExistsError('Preserve frozen candidate')
    report=read(DIRECTORY/'training.json');assert report['status']=='COMPLETED' and report['completed_updates']==128
    selected=torch.load(CHECKPOINTS/'shared.pt',weights_only=True,map_location='cpu')
    assert selected['step']==report['selected_checkpoint_step']
    assert selected['step']>0,'No trained checkpoint met the frozen regression caps; retain the failed experiment instead of deploying its initialization'
    row=next(row for row in report['history'] if row['step']==selected['step']);assert row['validation']['selection']['eligible']
    selected['config']['scope'].update(completed_updates=128,selected_update=selected['step'],default_promoted=False)
    sources=('mixed_solver.py','rod_solver.py','rod_rows.py','rod_chain.py','protected_response.py','inverse_response.py','vector_response.py','vector_ports.py','heterogeneous_backbone.py','hierarchical_response.py','vector_transport.py','unified_graph.py','runtime.py')
    selected['implementation_sha256']={name:sha(ROOT/'src/wp_vbd'/name) for name in sources}
    selected['training_report_sha256']=sha(DIRECTORY/'training.json')
    torch.save(selected,target)
    (ROOT/'configs/shared_hierarchy_candidate.json').write_text(json.dumps(selected['config'],indent=2)+'\n')
    record={'status':'FROZEN_EXPERIMENT_ONLY','checkpoint_sha256':sha(target),'selected_update':selected['step'],'completed_updates':128,'default_promoted':False,'implementation_sha256':selected['implementation_sha256'],'training_report_sha256':selected['training_report_sha256']}
    (DIRECTORY/'candidate.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('initialize','admit','freeze'));args=parser.parse_args();globals()[args.command]()
