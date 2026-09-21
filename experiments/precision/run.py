"""Same-state Gaussian cancellation experiment, with independent MP replay.

No training is performed. Dyadic gaps and central-stencil weights are exactly
representable in both tested IEEE formats, including the order-three weights.
Native raw and removable-singularity formulas evaluate the identical state.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, platform, sys, time
from pathlib import Path
from fractions import Fraction
import mpmath as mp
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
from gaussian_gram import Atom, subtract, operator, norm_interval, interval_record
DEFAULT=ROOT/'outputs/precision'
MAX_K={(precision,order):38 for precision in ('float32','float64') for order in (1,3)}

def lift(x):
    a,b=float(x).as_integer_ratio()
    return mp.mpf(a)/b

def hermite(n,x):
    a,b=mp.mpf(1),x
    if n==0:return a
    for j in range(1,n):a,b=b,x*b-j*a
    return b

def native(order,k,x,precision,method):
    d=np.dtype(precision).type
    x=np.asarray(x,dtype=d); h=d(math.ldexp(1.,-k)); h2=h*h
    if method=='raw':
        out=np.zeros((3,len(x)),dtype=d)
        for j in range(order+1):
            c=d(((-1)**(order-j))*math.comb(order,j)*math.ldexp(1.,order*k))
            b=d((j-order/2)*float(h)); z=x-b; e=np.exp(d(-.5)*z*z)
            out[0]+=c*e; out[1]+=c*(-z*e); out[2]+=c*((z*z-d(1))*e)
        return out
    if method=='completed':
        e=np.exp(d(-.5)*x*x)
        he=[np.ones_like(x),x]
        for n in range(1,order+2):he.append(x*he[-1]-d(n)*he[-2])
        return np.stack([he[order]*e,-he[order+1]*e,he[order+2]*e])
    if method!='stable':raise ValueError(method)
    z=d(.5)*h*x
    sc=np.divide(np.sinh(z),z,out=np.ones_like(z),where=z!=0)
    y=x*sc; c=np.cosh(z); e=np.exp(d(-.5)*x*x-h2/d(8))
    if order==1:
        f=y; fp=c; fpp=h2/d(4)*y
    elif order==3:
        r=-np.expm1(-h2)/h2; b=np.exp(-h2)
        f=b*y*y*y-d(3)*r*y
        fp=d(3)*(b*y*y-r)*c
        fpp=d(6)*b*y*c*c+d(3)*(b*y*y-r)*(h2/d(4))*y
    else:raise ValueError(order)
    return np.stack([e*f,e*(fp-x*f),e*(fpp-d(2)*x*fp+(x*x-d(1))*f)])

def split(value):
    hi=float(value)
    return hi,float(value-lift(hi))

def mp_reference(order,k,x,dps=100):
    """Direct signed Gaussian sum, independent of native stable identities.

    Return high/low binary64 pairs (~106-bit storage) and already-subtracted
    reference model errors. Subtraction against native values retains the low
    part, avoiding a binary64 reference floor in the replay itself.
    """
    n=len(x); hi=np.empty((3,n)); lo=np.empty_like(hi)
    thi=np.empty_like(hi); tlo=np.empty_like(hi); model=np.empty_like(hi)
    with mp.workdps(dps):
        h=mp.mpf(2)**(-k)
        weights=[mp.mpf((-1)**(order-j)*math.comb(order,j))/h**order for j in range(order+1)]
        centers=[(mp.mpf(j)-mp.mpf(order)/2)*h for j in range(order+1)]
        for i,xx in enumerate(x):
            xx=lift(xx); e=mp.exp(-xx*xx/2)
            target=[(-1)**s*hermite(order+s,xx)*e for s in range(3)]
            val=[mp.mpf(0) for _ in range(3)]
            for a,b in zip(weights,centers):
                z=xx-b; v=a*mp.exp(-z*z/2)
                val[0]+=v;val[1]-=z*v;val[2]+=(z*z-1)*v
            for s in range(3):
                hi[s,i],lo[s,i]=split(val[s])
                thi[s,i],tlo[s,i]=split(target[s])
                model[s,i]=float(val[s]-target[s])
    return hi,lo,thi,tlo,model

def norms(v,w,den_h1,den_res):
    h1=np.sqrt(np.dot(w,v[0]**2+v[1]**2))/den_h1
    res=np.sqrt(np.dot(w,(v[0]-v[2])**2))/den_res
    return float(h1),float(res)

def continuous(order,k):
    h=math.ldexp(1.,-k)
    state=[Atom(Fraction((-1)**(order-j)*math.comb(order,j))*Fraction(2)**(order*k),(j-order/2)*h) for j in range(order+1)]
    target=[Atom((-1)**order,0.,order)]
    diff=subtract(state,target)
    result={}
    for name,parts,tparts,s in [('h1',diff,target,1),('residual',operator(diff),operator(target),0)]:
        a=norm_interval(parts,s,140);b=norm_interval(tparts,s,140)
        result[name]=interval_record(a/b)
    return result

def run(out:Path,nodes:int=1025):
    start=time.perf_counter();out.mkdir(parents=True,exist_ok=True)
    x=np.linspace(-12.,12.,nodes);step=x[1]-x[0]
    w=np.full(nodes,step);w[[0,-1]]*=.5
    assert np.array_equal(x.astype(np.float32).astype(float),x),'Use dyadic grid'
    rows=[];checks=[];max_quad_rel=0.;max_identity=0.
    existing=out/'sweep.csv'
    if existing.exists():
        with existing.open() as f:
            for record in csv.DictReader(f):
                rows.append({name:(value if name in ('precision','method') else int(value) if name in ('order','k') else float(value)) for name,value in record.items()})
    cached={(r['order'],r['k']) for r in rows}
    cached_precisions={(r['order'],r['k'],r['precision']) for r in rows}
    for order,k,precision in cached_precisions:
        methods=[r['method'] for r in rows if (r['order'],r['k'],r['precision'])==(order,k,precision)]
        assert sorted(methods)==['completed','raw','stable'],(order,k,precision,methods)
    for order in (1,3):
        for k in range(max(v for (p,q),v in MAX_K.items() if q==order)+1):
            expected_precisions=[p for p in ('float32','float64') if k<=MAX_K[(p,order)]]
            if all((order,k,p) in cached_precisions for p in expected_precisions):
                cached_rows=[r for r in rows if r['order']==order and r['k']==k]
                expected=3*sum(k<=MAX_K[(p,order)] for p in ('float32','float64'))
                assert len(cached_rows)==expected, (order,k,len(cached_rows),expected)
                first=cached_rows[0]
                for metric in ('h1','residual'):
                    mid=(first['continuous_'+metric+'_lo']+first['continuous_'+metric+'_hi'])/2
                    max_quad_rel=max(max_quad_rel,abs(first['model_'+metric]-mid)/mid)
                if k in (0,4,8,12,MAX_K[('float64',order)]):
                    subset=x[::16];low=mp_reference(order,k,subset,100);high=mp_reference(order,k,subset,160)
                    delta=max(float(np.max(np.abs(a-b))) for a,b in zip(low,high))
                    assert delta<1e-70,(order,k,delta)
                    checks.append({'order':order,'k':k,'nodes':len(subset),'reference_dps':[100,160],'maximum_stored_pair_difference':delta})
                continue
            hi,lo,thi,tlo,model=mp_reference(order,k,x)
            target=thi+tlo
            dh=np.sqrt(np.dot(w,target[0]**2+target[1]**2));dr=np.sqrt(np.dot(w,(target[0]-target[2])**2))
            mh,mr=norms(model,w,dh,dr);cert=continuous(order,k)
            for metric,observed in [('h1',mh),('residual',mr)]:
                midpoint=sum(cert[metric])/2
                max_quad_rel=max(max_quad_rel,abs(observed-midpoint)/midpoint)
            if k in (0,4,8,12,MAX_K[('float64',order)]):
                # Recompute at higher precision on the same subset of dyadic nodes.
                idx=np.arange(0,nodes,16)
                check=mp_reference(order,k,x[idx],160)
                max_pair=max(float(np.max(np.abs(a[:,idx]-b))) for a,b in zip((hi,lo,thi,tlo,model),check))
                checks.append({'order':order,'k':k,'nodes':len(idx),'reference_dps':[100,160],'maximum_stored_pair_difference':max_pair})
                assert max_pair<1e-70,(order,k,max_pair)
            for precision in ('float32','float64'):
                if k>MAX_K[(precision,order)] or (order,k,precision) in cached_precisions:continue
                base={'order':order,'precision':precision,'k':k,'gap':math.ldexp(1.,-k),'coefficient_norm':math.sqrt(math.comb(2*order,order))*math.ldexp(1.,order*k),'model_h1':mh,'model_residual':mr,'continuous_h1_lo':cert['h1'][0],'continuous_h1_hi':cert['h1'][1],'continuous_residual_lo':cert['residual'][0],'continuous_residual_hi':cert['residual'][1]}
                for method in ('raw','stable','completed'):
                    v=native(order,k,x,precision,method).astype(float)
                    total=(v-thi)-tlo
                    eh,er=norms(total,w,dh,dr)
                    forward=(v-hi)-lo if method!='completed' else total
                    fh,fr=norms(forward,w,dh,dr)
                    rows.append({**base,'method':method,'total_h1':eh,'total_residual':er,'evaluation_h1':fh,'evaluation_residual':fr})
            print(f'order {order}, k {k}: model H1 {mh:.5e}',flush=True)
    with (out/'sweep.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summaries=[]
    for precision in ('float32','float64'):
        for order in (1,3):
            r=[v for v in rows if v['precision']==precision and v['order']==order and v['method']=='raw']
            best=min(r,key=lambda v:v['total_h1'])
            stable=[v for v in rows if v['precision']==precision and v['order']==order and v['method']=='stable']
            completed=next(v for v in rows if v['precision']==precision and v['order']==order and v['method']=='completed')
            small=[v for v in stable if v['model_h1']<np.finfo(precision).eps]
            summaries.append({'precision':precision,'order':order,'raw_best_sampled_gap':best['gap'],'raw_minimum_sampled_h1':best['total_h1'],'raw_best_model_h1':best['model_h1'],'raw_best_evaluation_h1':best['evaluation_h1'],'stable_minimum_sampled_h1':min(v['total_h1'] for v in stable),'stable_small_gap_evaluation_max':max((v['evaluation_h1'] for v in small),default=None),'stable_small_gap_count':len(small),'completed_h1':completed['total_h1'],'raw_best_sampled_residual':min(v['total_residual'] for v in r),'nominal_eps':float(np.finfo(precision).eps),'asymptotic_gap_scale':float(np.finfo(precision).eps**(1/(order+2))),'asymptotic_error_scale':float(np.finfo(precision).eps**(2/(order+2)))})
    assert max_quad_rel<1e-10,max_quad_rel
    summary={'protocol':{'description':'Fixed dyadic central stencils; no optimization or format-dependent parameters. Raw and stable evaluators represent identical functions. Completed derivative is a distinct limiting state.','orders':[1,3],'gap':'h=2^{-k}; k=0,...,max_k','max_k':{f'{p}_order{q}':v for (p,q),v in MAX_K.items()},'nodes':nodes,'interval':[-12,12],'quadrature':'composite trapezoid on dyadic nodes/weights','norms':['relative H1','relative L2 residual for L=1-dxx'],'primary_norm':'H1','references':'100-digit direct Gaussian sums; 160-digit subset check; 140-digit outward whole-line Gaussian Gram enclosures','reference_storage':'Binary64 high/low pairs; pre-subtracted exact model error; ~106-bit reference storage.','rounding_control':'All nodes, centers, coefficients are exactly representable in both float32/64. Only arithmetic changes.','completed_control':'Evaluates He_q(x) exp(-x^2/2) directly; it changes the finite-gap state to its derivative limit.','scope':'The observed optimal gaps/floors apply to these sampled naive evaluators and stencils; they are not global optimization lower bounds.'},'summary':summaries,'reference_checks':checks,'maximum_grid_vs_continuous_model_relative_difference':max_quad_rel,'seconds_this_invocation':time.perf_counter()-start,'resumed_cached_order_gap_cases':len(cached),'range_design_note':'All four order/precision sweeps now cover k=0,...,38 so plotted curves reach the same horizontal boundary. The earlier ranges ended at k26 (order-one binary32), k38 (order-one binary64), k15 (order-three binary32), and k32 (order-three binary64). Every earlier sample is retained. The order-three binary64 pilot had previously been extended from k22 to k32. These are range extensions, not endpoint selection.','environment':{'python':sys.version,'numpy':np.__version__,'mpmath':mp.__version__,'platform':platform.platform()},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(out/'grid.npz',x=x,weights=w)
    print(json.dumps(summaries,indent=2),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--nodes',type=int,default=1025)
    a=p.parse_args();run(a.output,a.nodes)
