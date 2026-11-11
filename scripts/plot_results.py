"""Question-led figures using complete paired records and verified spatial evidence."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import argparse
P=Path(__file__).resolve().parents[1];R=P/'results';O=P/'runs/reproduced';SP=R/'figure_data'
BLUE,RED,GRAY,PALE,INK='#2166AC','#B2182B','#969696','#C6DBEF','#252525'
GREEN,PURPLE,ORANGE='#1B7F5A','#6A51A3','#D95F02'
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif'],'mathtext.fontset':'stix',
 'font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
 'legend.fontsize':7.5,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
 'axes.spines.right':False,'axes.linewidth':.6,'lines.linewidth':1.25,'savefig.facecolor':'white'})
FILES={'P10-G44_discrete_modes': 'multirate_rd.json', 'P10-G43_independent_views': 'independent_views.json', 'P10-G48_lgsc_independent_views': 'independent_views_lgsc.json', 'P10-G40_reference_reuse_analysis': 'reference_reuse.json', 'P10-G45_cohort_representation': 'representation_ablation.json', 'P10-B12_recent_baselines': 'recent_public_codecs.json', 'P10-B14_recent_external': 'recent_external_codecs.json', 'P10-G42_lgsc_external': 'external_multicodec.json'}
def load(n):return json.loads((R/FILES[n]).read_text())
def save(fig,n):
 for e in ['pdf','png']:fig.savefig(O/f'{n}.{e}',dpi=350,bbox_inches='tight',pad_inches=.025)
 plt.close(fig)
def grid(ax):ax.grid(alpha=.12,lw=.45);ax.set_axisbelow(True)
def legend(ax,loc='best',**kw):ax.legend(loc=loc,frameon=True,facecolor='white',edgecolor='none',framealpha=.92,borderpad=.25,handlelength=1.3,labelspacing=.2,**kw)
def ci(v):
 v=np.asarray(v);rng=np.random.default_rng(20261101)
 return np.quantile(rng.choice(v,(10000,len(v)),replace=True).mean(1),[.025,.975])

def rd():
 d=load('P10-G44_discrete_modes');recent=load('P10-B12_recent_baselines')['statistics']
 fig,ax=plt.subplots(2,2,figsize=(3.5,3.42),layout='constrained');curves={}
 for backend in ['FCGS','L-GSC']:
  rows=d[backend]['rows'];n=len(rows[0]['baseline'])
  curves[(backend,'intra')]=([np.mean([8*r['baseline'][j]['bytes']/r['target_gaussians'] for r in rows]) for j in range(n)],[np.mean([r['baseline'][j]['psnr_db'] for r in rows]) for j in range(n)])
  curves[(backend,'selected')]=([np.mean([8*r['selected'][j]['bytes']/r['target_gaussians'] for r in rows]) for j in range(n)],[np.mean([r['selected'][j]['psnr_db'] for r in rows]) for j in range(n)])
 flex=sorted((v['rate_bpg'],v['psnr_db']) for v in recent['FlexGaussian'].values());gs=recent['GSCodec_rp1']
 styles=[('FlexGaussian',*[list(v) for v in zip(*flex)],GRAY,'^',':'),('FCGS',*curves[('FCGS','intra')],BLUE,'o','--'),('RevisitGS + FCGS',*curves[('FCGS','selected')],RED,'D','-'),('L-GSC',*curves[('L-GSC','intra')],GREEN,'s','-.'),('RevisitGS + L-GSC',*curves[('L-GSC','selected')],ORANGE,'P','-')]
 for panel in ax[0]:
  for label,xx,yy,c,m,ls in styles:panel.plot(xx,yy,c=c,marker=m,ms=3.1,ls=ls,mfc='white' if 'RevisitGS' not in label else c,label=label)
  panel.scatter(gs['rate_bpg'],gs['psnr_db'],c=PURPLE,marker='*',s=36,edgecolors='white',linewidths=.35,zorder=6,label='GSCodec RP1');grid(panel)
 ax[0,0].set(title='(a) Public-codec landscape',xlabel='Rate (bits/Gaussian)',ylabel='PSNR (dB)',xlim=(8,210),ylim=(12,54));legend(ax[0,0],'lower right',fontsize=5.6,ncol=2,columnspacing=.55,handletextpad=.3)
 ax[0,1].set(title='(b) High-quality operating region',xlabel='Rate (bits/Gaussian)',ylabel='PSNR (dB)',xlim=(72,132),ylim=(34,50));ax[0,1].text(.96,.06,'Streams only',transform=ax[0,1].transAxes,ha='right',va='bottom',fontsize=6.3,color=GRAY)
 for k,backend in enumerate(['FCGS','L-GSC']):
  rows=d[backend]['rows'];n=len(rows[0]['baseline'])
  for key,c,m,label in [('conditional',GRAY,'o','Conditional'),('selected',RED,'D','Selected')]:
   dx=np.array([[100*(r[key][j]['bytes']/r['baseline'][j]['bytes']-1) for j in range(n)] for r in rows]);dy=np.array([[r[key][j]['psnr_db']-r['baseline'][j]['psnr_db'] for j in range(n)] for r in rows])
   ax[1,k].scatter(dx,dy,color=c,marker=m,s=7,alpha=.22,edgecolors='none');ax[1,k].plot(dx.mean(0),dy.mean(0),c=c,ls='-' if key=='selected' else '--',marker=m,ms=3.7,label=label)
  ax[1,k].axvline(0,c=BLUE,lw=.65);ax[1,k].axhline(0,c=BLUE,lw=.65);ax[1,k].set(title=f'({chr(99+k)}) {backend}: paired shifts',xlabel='Rate change (%)',ylabel='PSNR change (dB)');legend(ax[1,k],'lower right',fontsize=6.4);grid(ax[1,k])
 save(fig,'discrete_rd')

def transfer():
 d=json.loads((R/'backend_transfer.json').read_text())
 a={r['pair']:(r['psnr_gain'],r['ssim_gain']) for r in d['FCGS']}
 b={r['pair']:(r['psnr_gain'],r['ssim_gain']) for r in d['L-GSC']}
 ids=sorted(set(a)&set(b));fig,axs=plt.subplots(1,2,figsize=(3.5,2.35),layout='constrained')
 for ax,data,title in zip(axs,[a,b],['(a) FCGS','(b) L-GSC L1']):
  v=np.array([data[i] for i in ids]);good=(v[:,0]>0)&(v[:,1]>0)
  ax.axvline(0,c=BLUE,lw=.7);ax.axhline(0,c=BLUE,lw=.7)
  ax.scatter(v[good,0],1000*v[good,1],c=RED,s=20,marker='o',edgecolors='white',linewidths=.4)
  ax.scatter(v[~good,0],1000*v[~good,1],c=GRAY,s=22,marker='x',lw=.8)
  ax.scatter([0],[0],c=BLUE,s=32,marker='P',edgecolors='white',linewidths=.4,zorder=5)
  ax.set(title=title,xlabel='PSNR gain (dB)',ylabel=r'SSIM gain ($10^{-3}$)');grid(ax)
 save(fig,'cohort_evidence')

def external_public():
 d=load('P10-G42_lgsc_external');recent=load('P10-B14_recent_external')['means']
 fig,axs=plt.subplots(1,2,figsize=(3.5,2.05),layout='constrained')
 specifications=[('CL-Splats','clsplats_captured','(a) CL-Splats / reference RGB'),('ScanNet-GSReg','scannet_target','(b) ScanNet-GSReg / target render')]
 for index,(dataset,key,title) in enumerate(specifications):
  ax=axs[index];means=recent[dataset];flex=sorted((v['rate_bpg'],v['psnr_db']) for name,v in means.items() if name.startswith('FlexGaussian:'))
  ax.plot(*zip(*flex),c=GRAY,marker='^',mfc='white',ms=3.5,ls=':',label='FlexGaussian')
  gs=means['GSCodec:rp1'];ax.scatter(gs['rate_bpg'],gs['psnr_db'],c=PURPLE,marker='*',s=38,edgecolors='white',linewidths=.35,zorder=6,label='GSCodec RP1')
  for backend,c,m in [('FCGS',BLUE,'o'),('L-GSC',GREEN,'s')]:
   records=d['backends'][backend][key];intra=records['intra'];selected=records['selected']
   ax.scatter(intra['mean_rate_bpg'],intra['mean_psnr_db'],facecolors='white',edgecolors=c,marker=m,s=26,linewidths=1.1,label=backend)
   ax.scatter(selected['mean_rate_bpg'],selected['mean_psnr_db'],c=RED if backend=='FCGS' else ORANGE,marker='D' if backend=='FCGS' else 'P',s=27,edgecolors='white',linewidths=.35,label=f'RevisitGS + {backend}')
   ax.plot([intra['mean_rate_bpg'],selected['mean_rate_bpg']],[intra['mean_psnr_db'],selected['mean_psnr_db']],c=c,lw=.75,alpha=.75)
  ax.set(title=title,xlabel='Rate (bits/Gaussian)',ylabel='PSNR (dB)');grid(ax)
  if index==0:legend(ax,'lower right',fontsize=5.55,ncol=2,columnspacing=.55,handletextpad=.3)
 save(fig,'external_public_rd')

def independent():
 datasets=[('FCGS',load('P10-G43_independent_views')),('L-GSC L1',load('P10-G48_lgsc_independent_views'))]
 fig,ax=plt.subplots(2,2,figsize=(3.5,3.5),layout='constrained')
 for k,(name,d) in enumerate(datasets):
  allx=[];ally=[];selection=[];means=[];bounds=[]
  for r in d['rows']:
   active=r['selected_mode']=='conditional';sv=[v['conditional_psnr']-v['intra_psnr'] for v in r['views'] if v['role']=='selection'];vv=[v for v in r['views'] if v['role']=='independent']
   y=np.array([v['conditional_psnr']-v['intra_psnr'] if active else 0 for v in vv]);x=np.array([v['index']/max(r['selection_indices']) for v in vv]);allx.extend(x);ally.extend(y)
   selection.append(np.mean(sv) if active else 0);means.append(y.mean());bounds.append(np.quantile(y,[.1,.9]))
  x=np.array(allx);y=np.array(ally);bins=np.linspace(0,1,9);centers=(bins[:-1]+bins[1:])/2;quant=np.array([np.quantile(y[(x>=a)&(x<b)],[.1,.5,.9]) for a,b in zip(bins[:-1],bins[1:])])
  a=ax[0,k];a.scatter(x,y,s=3,c=GRAY,alpha=.18,rasterized=True);a.fill_between(centers,quant[:,0],quant[:,2],color=PALE,alpha=.7,label='10–90% range');a.plot(centers,quant[:,1],c=RED,marker='o',ms=2.5,label='View median');a.axhline(0,c=BLUE,lw=.7,ls='--')
  a.set(title=f'({chr(97+k)}) {name}',xlabel='Camera index (normalized)',ylabel='PSNR gain (dB)');legend(a,'lower left',fontsize=6.2)
  a=ax[1,k];b=np.array(bounds);selection=np.array(selection);means=np.array(means)
  for xx,lo,hi in zip(selection,b[:,0],b[:,1]):a.plot([xx,xx],[lo,hi],c=GRAY,alpha=.35,lw=.6)
  a.scatter(selection,means,c=RED,s=10,edgecolors='white',linewidths=.25,zorder=3)
  lo=min(0,selection.min(),means.min());hi=max(0,selection.max(),means.max());pad=max((hi-lo)*.1,.05);a.plot([lo-pad,hi+pad],[lo-pad,hi+pad],c=BLUE,ls='--',lw=.75)
  a.axhline(0,c=GRAY,lw=.6);a.set(title=f'({chr(99+k)}) Mean transfer',xlabel='Selection gain (dB)',ylabel='Independent gain (dB)')
  a.text(.04,.96,rf'$\Delta R={d["mean_selected_rate_change_percent"]:+.2f}\%$',transform=a.transAxes,ha='left',va='top',fontsize=7)
  for a in ax[:,k]:grid(a)
 save(fig,'independent_views')

def assignment():
 files=sorted(SP.glob('assignment_*.npz'));d=np.load(SP/'assignment_04.npz');methods=['nearest','greedy_k8','hungarian_attribute_oracle']
 su=d['source_uv'];tu=d['target_uv'];nearest=set(d['nearest_rows'].tolist());greedy=d['greedy_k8_rows'];eligible=np.array([i for i in greedy if i not in nearest and d['source_z'][i]>0 and 0<su[i,0]<960 and 0<su[i,1]<540])
 seed=eligible[np.argmin(np.linalg.norm(su[eligible]-[480,270],axis=1))]
 eligible=np.flatnonzero((d['source_z']>0)&(su[:,0]>0)&(su[:,0]<960)&(su[:,1]>0)&(su[:,1]<540));take=eligible[np.argsort(np.linalg.norm(su[eligible]-su[seed],axis=1))[:12]]
 selected={m:{int(r):int(c) for r,c in zip(d[m+'_rows'],d[m+'_cols'])} for m in methods}
 endpoints=np.array([tu[mp[int(i)]] for mp in selected.values() for i in take if int(i) in mp]);xy=np.r_[su[take],endpoints];pad=np.maximum(np.ptp(xy,axis=0)*.13,3);lo=xy.min(0)-pad;hi=xy.max(0)+pad
 fig=plt.figure(figsize=(3.5,2.95));gs=fig.add_gridspec(2,6,height_ratios=[1.05,1],hspace=.65,wspace=.7)
 ax=fig.add_subplot(gs[0,:2]);ax.imshow(d['rgb']);ax.add_patch(Rectangle(lo,*(hi-lo),fill=False,ec=RED,lw=.9));ax.axis('off');ax.set_title('(a) Scene patch',fontsize=8)
 ax=fig.add_subplot(gs[0,3:]);labels=['Nearest','Greedy $k=8$','Exact attributes']
 for method,label,c,ls in zip(methods,labels,[GRAY,RED,BLUE],[':','-','--']):
  cost=np.concatenate([np.load(f)[method+'_cost'] for f in files]);cost=np.sort(cost)
  ax.plot(cost,np.arange(1,len(cost)+1)/len(cost),c=c,ls=ls,lw=1.1,label=label)
 ax.set(xlabel='Retained residual cost',ylabel='Cumulative fraction',title='(b) All bounded graphs');legend(ax,'lower right',fontsize=6.2);grid(ax)
 for k,(m,label) in enumerate(zip(methods,labels)):
  ax=fig.add_subplot(gs[1,2*k:2*k+2]);mp=selected[m]
  for i in take:
   if int(i) not in mp:continue
   a,b=su[i],tu[mp[int(i)]];ax.plot([a[0],b[0]],[a[1],b[1]],c=GRAY,lw=.85);ax.scatter(*b,c=RED,marker='x',s=19,lw=.8)
  ax.scatter(su[take,0],su[take,1],facecolors='white',edgecolors=BLUE,s=21,lw=.9,zorder=4)
  ax.set(xlim=(lo[0],hi[0]),ylim=(hi[1],lo[1]));ax.axis('off');ax.set_title(f'({chr(99+k)}) '+label,fontsize=8)
  ax.text(.5,-.02,f'{sum(int(i) in mp for i in take)}/{len(take)} assigned',transform=ax.transAxes,ha='center',va='top',fontsize=7)
 fig.subplots_adjust(left=.03,right=.97,top=.91,bottom=.08);save(fig,'assignment_analysis')

def reuse():
 d=load('P10-G40_reference_reuse_analysis');rows=d['variants']+[d['global_selection']]
 labels=['Nearest','Geometry-only','No allocation','Uniform q8','RevisitGS raw','RevisitGS selected']
 fig,ax=plt.subplots(2,1,figsize=(3.5,3.15),gridspec_kw={'height_ratios':[1.2,1]},layout='constrained')
 offsets=[(2.1,.08),(2.1,.73),(.8,-.19),(5.0,-.18),(4.2,.30),(4.2,.57)]
 for k,(r,label,offset) in enumerate(zip(rows,labels,offsets)):
  x=np.array([v['rate_change_percent'] for v in r['rows']]);y=np.array([v['matched_rate_psnr_gain_db'] for v in r['rows']]);xx,yy=x.mean(),y.mean();xc,yc=ci(x),ci(y)
  c=RED if k>=4 else BLUE if k==1 else GRAY;m='*' if k==5 else 'D' if k==4 else 'o'
  ax[0].errorbar(xx,yy,xerr=[[xx-xc[0]],[xc[1]-xx]],yerr=[[yy-yc[0]],[yc[1]-yy]],color=c,lw=.9,capsize=2)
  ax[0].scatter(xx,yy,c=c,marker=m,s=48 if k>=4 else 24,zorder=4)
  ax[0].annotate(label,(xx,yy),xytext=offset,textcoords='data',fontsize=7.5,color=c,arrowprops={'arrowstyle':'-','color':c,'lw':.5})
 ax[0].axhline(0,c=GRAY,lw=.6);ax[0].axvline(0,c=GRAY,lw=.6);ax[0].set(xlabel='Mean rate change (%)',ylabel='Matched-rate gain (dB)',title='(a) Complete-stream trade-offs');ax[0].set(xlim=(-.8,7.9),ylim=(-.28,.92))
 raw={r['pair']:r['matched_rate_psnr_gain_db'] for r in rows[4]['rows']}
 for j,k in enumerate([0,1,2,3]):
  v=np.array([raw[r['pair']]-r['matched_rate_psnr_gain_db'] for r in rows[k]['rows']]);rng=np.random.default_rng(j)
  ax[1].scatter(v,j+rng.uniform(-.13,.13,len(v)),s=5,c=GRAY,alpha=.4);lo,hi=ci(v)
  ax[1].plot([lo,hi],[j,j],c=RED,lw=2.4);ax[1].plot(v.mean(),j,'D',c=RED,ms=4,mec='white',mew=.4)
 ax[1].set(yticks=range(4),yticklabels=['Nearest','Geometry-only','No allocation','Uniform q8'],xlabel='Raw RevisitGS gain minus control (dB)',title='(b) Paired component contrasts');ax[1].invert_yaxis();ax[1].axvline(0,c=BLUE,ls='--',lw=.8)
 for a in ax:grid(a)
 save(fig,'reference_reuse_baselines')

def representation():
 d=load('P10-G45_cohort_representation');rr=[{p['profile']:p for p in r['profiles']} for r in d['rows']];im=np.load(SP/'ablation_06.npz')
 fig=plt.figure(figsize=(3.5,3.35));gs=fig.add_gridspec(3,6,height_ratios=[.67,.67,1.6],wspace=.65,hspace=.45)
 specs=[('without_opacity_revision','No opacity'),('without_directional_revision','No SH revision'),('default_rank4_q4','RevisitGS')]
 # A common central region per camera, fixed independently of method differences.
 for i in range(2):
  for j,(key,label) in enumerate(specs):
   ax=fig.add_subplot(gs[i,2*j:2*j+2]);crop=im[key][i];ref=im['target'][i]
   ax.imshow(crop,interpolation='nearest');ax.axis('off')
   if i==0:ax.set_title(label,fontsize=7.7,pad=3)
   if j==0:ax.text(-.08,.5,['View A','View B'][i],rotation=90,ha='right',va='center',transform=ax.transAxes,fontsize=7)
   ax.text(.5,-.025,f'MAE {np.mean(abs(crop-ref)):.4f}',ha='center',va='top',transform=ax.transAxes,fontsize=7)
 for ax,keys,labs,title in [(fig.add_subplot(gs[2,:3]),['without_scale_revision','without_opacity_revision','without_rotation_revision'],['Scale','Opacity','Rotation'],'(a) Geometric attributes'),(fig.add_subplot(gs[2,3:]),['without_directional_revision','nested_rank2_q4','rank4_q8','uniform_q8'],['No SH','Rank 2','8-bit SH','Uniform 8'],'(b) Direction / precision')]:
  for j,key in enumerate(keys):
   v=np.array([r['default_rank4_q4']['mean_psnr_db']-r[key]['mean_psnr_db'] for r in rr]);rng=np.random.default_rng(j)
   ax.scatter(j+rng.uniform(-.14,.14,len(v)),v,s=5,c=GRAY,alpha=.45);lo,hi=ci(v)
   ax.plot([j,j],[lo,hi],c=RED,lw=2);ax.plot(j,v.mean(),'D',c=RED,ms=4,mec='white',mew=.4)
  ax.axhline(0,c=BLUE,lw=.7,ls='--');ax.set(xticks=range(len(keys)),xticklabels=labs,ylabel='PSNR difference (dB)' if len(keys)==3 else '',title=title);ax.tick_params(axis='x',labelrotation=35,labelsize=7);grid(ax)
 fig.subplots_adjust(left=.14,right=.99,top=.93,bottom=.17);save(fig,'ablation_evidence')

def complexity():
 sets=[('FCGS',load('P10-G43_independent_views')),('L-GSC L1',load('P10-G48_lgsc_independent_views'))]
 fig,ax=plt.subplots(2,2,figsize=(3.5,3.45),layout='constrained')
 for k,(name,d) in enumerate(sets):
  rows=sorted(d['rows'],key=lambda r:r['target_gaussians']);x=np.array([r['target_gaussians']/1e6 for r in rows]);base=np.array([r['decode_intra_wall_seconds'] for r in rows]);birth=np.array([r['decode_birth_wall_seconds'] for r in rows]);core=np.array([r['core_decode_seconds'] for r in rows]);render=np.array([r['selection_render_metric_seconds'] for r in rows]);a=ax[0,k]
  for xx,aa,bb in zip(x,base,birth+core):a.plot([xx,xx],[aa,bb],c=GRAY,alpha=.5,lw=.6)
  a.scatter(x,base,c=BLUE,marker='o',s=9,label='Public intra');a.scatter(x,birth+core,c=RED,marker='D',s=9,label='RevisitGS');a.set_yscale('log');a.set_ylim(.08,25);a.set_yticks([.1,1,10],['0.1','1','10']);a.set(xlabel='Target Gaussians (M)',ylabel='Decoder time (s, log)',title=f'({chr(97+k)}) {name}');legend(a,'lower right',fontsize=6.7);grid(a)
  a=ax[1,k];total=base+birth+core+render;bottom=np.zeros(len(x))
  for v,c,label in zip([base,birth,core,render],[GRAY,BLUE,RED,PALE],['Intra','Birth','Revision','Views']):
   share=100*v/total;a.bar(range(len(x)),share,bottom=bottom,width=.88,color=c,label=label);bottom+=share
  a.set(xlabel='Map-pair size order',ylabel='Decision-stage share (%)',title=f'({chr(99+k)}) Stage composition',xlim=(-1,len(x)),ylim=(0,140));a.set_yticks([0,50,100]);a.set_xticks([0,20,41],[1,21,42]);legend(a,'upper left',ncol=2,fontsize=6.1);a.text(.97,.78,f'Median {np.median(total):.2f} s',transform=a.transAxes,ha='right',va='top',fontsize=6.5);grid(a)
 save(fig,'combined_analysis')

def main():
 global O
 parser=argparse.ArgumentParser(description="Regenerate the numerical paper figures from released measurements.")
 parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args();O=args.output;O.mkdir(parents=True,exist_ok=True)
 for f in [rd,transfer,external_public,independent,assignment,reuse,representation,complexity]:
  f();print(f.__name__,flush=True)
if __name__=='__main__':main()
