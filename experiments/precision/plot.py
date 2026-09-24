"""Redraw the controlled precision experiment from the complete recorded sweep."""
from pathlib import Path
import argparse,csv,json
import numpy as np
from plot_style import apply_style, save_figure, FULL_WIDTH, BLUE,RED,GREEN,GRAY
apply_style()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator,NullFormatter
from matplotlib.text import Text
from matplotlib.transforms import Bbox
ROOT=Path(__file__).resolve().parents[2]
DEFAULT_INPUT=ROOT/'data/precision'
DEFAULT_OUTPUT=ROOT/'outputs/precision/precision_sweep'

def texnum(x,digits=2):
    exponent=int(np.floor(np.log10(x)));mantissa=x/10.**exponent
    return rf'{mantissa:.{digits-1}f}\times10^{{{exponent}}}'

def label_box(annotation,renderer):
    """The numerical text's extent, excluding its leader to the marked point."""
    return Text.get_window_extent(annotation,renderer=renderer)

def label_is_clear(annotation,axis,renderer,padding_points=2.5):
    """Check rendered text plus whitespace against every plotted curve."""
    text_box=label_box(annotation,renderer)
    pad=padding_points*axis.figure.dpi/72
    box=Bbox.from_extents(text_box.x0-pad,text_box.y0-pad,
                          text_box.x1+pad,text_box.y1+pad)
    if not (axis.bbox.contains(box.x0,box.y0) and axis.bbox.contains(box.x1,box.y1)):
        return False
    for line in axis.lines:
        path=line.get_path().transformed(line.get_transform())
        half_width=line.get_linewidth()*axis.figure.dpi/144
        curve_box=Bbox.from_extents(box.x0-half_width,box.y0-half_width,
                                    box.x1+half_width,box.y1+half_width)
        if path.intersects_bbox(curve_box,filled=False):return False
        if any(curve_box.contains(*point) for point in path.vertices):return False
    for other in axis.texts:
        if other is not annotation and box.overlaps(other.get_window_extent(renderer)):
            return False
    return True

def place_minimum_labels(figure,requests):
    """Find nearby empty regions rather than cover the numerical curves."""
    figure.canvas.draw()
    renderer=figure.canvas.get_renderer()
    report=[]
    candidates=[(dx,dy) for dy in (24,30,36,42,48,54,-26,-32,-38,-44,-50)
                for dx in (0,-12,12,-24,24,-36,36)]
    for axis,x,y,panel in requests:
        annotation=axis.annotate(r'$'+texnum(y)+r'$',xy=(x,y),xytext=(0,24),
                    textcoords='offset points',ha='center',va='bottom',
                    fontsize=8,color=BLUE,
                    arrowprops={'arrowstyle':'-','color':BLUE,'lw':.6,
                                'shrinkA':3,'shrinkB':4})
        for offset in candidates:
            annotation.set_position(offset)
            annotation.update_positions(renderer)
            if label_is_clear(annotation,axis,renderer):break
        else:raise RuntimeError(f'No curve-free numerical label position for {panel}')
        report.append({'panel':panel,'offset_points':list(offset),
                       'padding_from_curves_points':2.5,
                       'label_curve_clearance_passed':True})
    figure.canvas.draw()
    renderer=figure.canvas.get_renderer()
    for (axis,_,_,panel),entry in zip(requests,report):
        annotation=axis.texts[-1]
        assert label_is_clear(annotation,axis,renderer),panel
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=DEFAULT_INPUT,
                        help='Directory containing sweep.csv and summary.json')
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT,
                        help='Output stem for PDF, SVG, PNG, and metadata JSON')
    args=parser.parse_args()
    with (args.input/'sweep.csv').open() as f:data=list(csv.DictReader(f))
    summary=json.loads((args.input/'summary.json').read_text())
    common_xlim=(max(float(record['gap']) for record in data),
                 min(float(record['gap']) for record in data))
    common_ylim=(1e-17,2.)
    fig,axes=plt.subplots(2,2,figsize=(FULL_WIDTH,4.75),sharex=True,sharey=True)
    fig.subplots_adjust(left=.095,right=.985,bottom=.125,top=.86,hspace=.12,wspace=.08)
    derivative_labels=('First derivative','Third derivative')
    precision_labels=('single precision','double precision')
    labels=[f'{derivative} · {precision}' for derivative in derivative_labels
            for precision in precision_labels]
    minimum_labels=[]
    for row,q in enumerate((1,3)):
        for col,precision in enumerate(('float32','float64')):
            ax=axes[row,col]
            curves={}
            for method,color,ls in [('raw',BLUE,'-'),('stable',RED,'-'),('completed',GREEN,'-.')]:
                records=sorted([r for r in data if r['precision']==precision and int(r['order'])==q and r['method']==method],key=lambda r:int(r['k']))
                h=np.array([float(r['gap']) for r in records]);e=np.array([float(r['total_h1']) for r in records]);curves[method]=(h,e,records)
                if (h[0],h[-1])!=common_xlim:
                    raise ValueError(f'Incomplete gap range for order {q}, {precision}, {method}; extend the precision sweep before plotting')
                ax.loglog(h,e,color=color,ls=ls,lw=1.5,zorder=3 if method=='stable' else 2)
            h,e,records=curves['raw'];ideal=np.array([float(r['model_h1']) for r in records])
            ax.loglog(h,ideal,color=GRAY,ls='--',lw=1.25,zorder=1)
            best=int(np.argmin(e));ax.plot(h[best],e[best],marker='o',ms=4,color=BLUE,zorder=5)
            minimum_labels.append((ax,h[best],e[best],labels[2*row+col]))
            ax.set_xlim(common_xlim);ax.set_ylim(common_ylim)
            if row==0:
                ax.set_title(precision_labels[col],fontsize=9)
                ax.tick_params(axis='x',which='both',bottom=False,top=False,
                               labelbottom=False,labeltop=False)
            else:
                ax.set_xlabel(r'adjacent separation $h$')
            if col==0:
                ax.set_ylabel(derivative_labels[row]+'\n'+r'relative $H^1$ error',
                              linespacing=2.0)
            else:
                ax.tick_params(axis='y',which='both',left=False,right=False,
                               labelleft=False,labelright=False)
            ax.xaxis.set_major_locator(LogLocator(base=10,numticks=5));ax.yaxis.set_major_locator(LogLocator(base=10,numticks=5))
            ax.xaxis.set_minor_formatter(NullFormatter());ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(axis='y',which='major',color='#e2e2e2',lw=.45)
    handles=[Line2D([],[],color=BLUE,label='raw Gaussian sum'),Line2D([],[],color=RED,label='stable evaluation: same state'),Line2D([],[],color=GRAY,ls='--',label='exact finite-gap error'),Line2D([],[],color=GREEN,ls='-.',label='completed derivative')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,1.005),ncol=2,columnspacing=2.5,handlelength=2.7,fontsize=8)
    label_audit=place_minimum_labels(fig,minimum_labels)
    paths=save_figure(fig,args.output)
    args.output.with_name(args.output.name+'_metadata.json').write_text(json.dumps({'figure':paths,'panels':labels,'rows_in_complete_sweep':len(data),'raw_best_sampled_endpoints':summary['summary'],'numerical_label_clearance':label_audit,'shared_axis_limits':{'x':common_xlim,'y':common_ylim},'all_curves_cover_full_horizontal_range':True,'subplot_spacing':{'hspace':fig.subplotpars.hspace,'wspace':fig.subplotpars.wspace},'display_note':'All four panels share x and y limits. The separation decreases to the right. Every curve is evaluated across the complete common gap range and meets the plot boundary; no extrapolation is used. All values, including out-of-range points, remain in sweep.csv.'},indent=2)+'\n')
    print(json.dumps(paths,indent=2))
if __name__=='__main__':main()
