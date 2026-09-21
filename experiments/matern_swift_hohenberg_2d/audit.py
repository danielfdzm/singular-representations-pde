"""Replay the Matérn checks reported in AppendixD.3; no training."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import experiment as E
import gaussian_target as G
import matern
from records import load_results


def exact_target_energy(parts, order):
    points, weights = E.quadrature(order)
    u, ux, uy, lap, lx, ly, lap2 = parts(points[:, 0], points[:, 1])
    f = (1+E.RHO)*u+2*lap+lap2+u**3
    volume = np.dot(weights, .5*(u+lap)**2+.5*E.RHO*u*u+.25*u**4-f*u)
    bp, bw, normals = E.boundary_quadrature(order)
    u, ux, uy, lap, lx, ly, _ = parts(bp[:, 0], bp[:, 1])
    dn = ux*normals[:, 0]+uy*normals[:, 1]
    q = -((ux+lx)*normals[:, 0]+(uy+ly)*normals[:, 1])
    return float(volume-np.dot(bw, (u+lap)*dn+q*u))


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=root/'data/matern')
    parser.add_argument('--output', type=Path, default=root/'outputs/audits/matern.json')
    args = parser.parse_args()
    data = load_results(args.input)
    energies = {name: {str(order): exact_target_energy(parts, order) for order in (32,48,64,80,96,128)}
                for name, parts in [('matern', E._manufactured_jet_parts), ('gaussian', G._manufactured_jet_parts)]}
    run, pair = min(((run,pair) for run in data['runs'] if run['method']=='ordinary'
                    for pair in run['final_state']['pairs']), key=lambda item: item[1]['half_gap'])
    points, weights = E.quadrature(64)
    r = points-np.asarray(pair['center'])
    direction = np.array([math.cos(pair['angle']), math.sin(pair['angle'])]); h = pair['half_gap']
    a,b,delta,t = matern.pair_geometry(r,direction,h)
    stable = matern.pair_combination('f0',a,b,delta,t)[1]
    zero = matern.pair_combination('f0',*matern.pair_geometry(r,direction,0.0))[1]
    naive = (matern.radial('f0',np.linalg.norm(r-h*direction,axis=1))
             -matern.radial('f0',np.linalg.norm(r+h*direction,axis=1)))/(2*h)
    norm = np.sqrt(np.dot(weights,zero*zero))
    report = {'training_rerun':False,'exact_target_energy':energies,
              'energy_differences':{k:{'32_to_48':abs(v['32']-v['48']),'96_to_128':abs(v['96']-v['128'])} for k,v in energies.items()},
              'cancellation':{'seed':run['seed'],'half_gap':h,'quadrature_order':64,
                              'metric':'relative quadrature L2 error against zero-gap column',
                              'naive_relative_error':float(np.sqrt(np.dot(weights,(naive-zero)**2))/norm),
                              'stable_relative_error':float(np.sqrt(np.dot(weights,(stable-zero)**2))/norm),
                              'naive_relative_max_norm_error':float(np.max(np.abs(naive-zero))/np.max(np.abs(zero))),
                              'stable_relative_max_norm_error':float(np.max(np.abs(stable-zero))/np.max(np.abs(zero)))},
              'stored_run_checks':{'runs':len(data['runs']),
                'largest_order80_order96_energy_difference':max(r['energy_order80_order96_difference'] for r in data['runs']),
                'largest_coefficient_condition':max(r['maximum_gradient_basis_condition'] for r in data['runs']),
                'maximum_coefficient_solves':max(r['coefficient_solves'] for r in data['runs']),
                'completed_starts':sum(bool(r['transfers']) for r in data['runs'] if r['method']=='completion')}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
