"""Replay AppendixD.4 conditioning and stored-run checks; no training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import experiment as E
import unit_scale as U
from records import load_results


def target_state(module):
    return {'pairs':[{'center':list(c),'angle':angle,'half_gap':0.0,'completed':True}
                     for c,angle,a,b in module.MANUFACTURED_JET['pairs']],
            'singletons':[list(c) for c,a in module.MANUFACTURED_JET['singletons']],
            'coefficients':[w for c,angle,a,b in module.MANUFACTURED_JET['pairs'] for w in (a,b)]
                           +[a for c,a in module.MANUFACTURED_JET['singletons']]}


def conditioning(module):
    state = target_state(module)
    points,weights = module.quadrature(32)
    values,gradients,_ = module.basis_fields(state,points)
    bp,bw,bn = module.boundary_quadrature(32)
    bv = module.basis_fields(state,bp)[0]
    force = module.manufactured_jet_load(points[:,0],points[:,1])
    neumann = module.manufactured_neumann_data(bp,bn)
    coefficients,info = module.fit_coefficients(values,gradients,weights,force,bv,bw,neumann)
    pq,wq = module.quadrature(96); bq,bqw,bqn = module.boundary_quadrature(96)
    return {'coefficient_condition':info['gradient_basis_condition'],
            'max_absolute_coefficient_error':float(np.max(np.abs(coefficients-state['coefficients']))),
            'relative_coefficient_error':float(np.linalg.norm(coefficients-state['coefficients'])/np.linalg.norm(state['coefficients'])),
            'target_mean':module.TARGET_MEAN,
            'neumann_compatibility_error':float(abs(np.dot(wq,module.manufactured_jet_load(pq[:,0],pq[:,1]))
                                            +np.dot(bqw,module.manufactured_neumann_data(bq,bqn)))),
            'target_h1_norm':module.energy(state,96)['h1_norm'],
            'target_energy':module.energy(state,96)['energy']}


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=root/'data/poisson')
    parser.add_argument('--output',type=Path,default=root/'outputs/audits/poisson.json')
    args = parser.parse_args()
    data = load_results(args.input); unit = load_results(args.input/'audits/unit_scale')
    # Transfer records retain the measured change and post-transfer state.
    transfers = [t for r in data['runs'] for t in r['transfers']]
    report = {'training_rerun':False,'scale045':conditioning(E),'unit_scale_historical_target':conditioning(U),
              'scale_comparison_note':'The historical unit-scale target and anchors also differ; these are two documented configurations, not a controlled scale-only comparison.',
              'stored_run_checks':{'runs':len(data['runs']),
                'largest_order64_order96_energy_difference':max(r['energy_order64_order96_difference'] for r in data['runs']),
                'largest_coefficient_condition':max(r['maximum_gradient_basis_condition'] for r in data['runs']),
                'maximum_coefficient_solves':max(r['coefficient_solves'] for r in data['runs']),
                'transfers_per_completed_start':[len(r['transfers']) for r in data['runs'] if r['method']=='completion'],
                'measured_h1_transfer_change_range':[min(t['bound']['measured_h1_state_change'] for t in transfers),
                                                     max(t['bound']['measured_h1_state_change'] for t in transfers)]},
              'unit_scale_endpoints':[{'seed':r['seed'],'method':r['method'],'status':r['status'],
                                      'energy':r['independent_quadrature'][-1]['energy'],
                                      'singleton_centres':r['final_state']['singletons'],
                                      'singleton_coefficients':r['final_state']['coefficients'][-2:]}
                                     for r in unit['runs']]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
