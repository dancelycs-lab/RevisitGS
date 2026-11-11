"""Full-scene context, genuine decoded RGB zooms, and calibrated local errors.

Reads the verified render cache only; crops never depend on method errors.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import argparse
P=Path(__file__).resolve().parents[1];R=P/'results';O=P/'runs/reproduced'
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif'],
 'mathtext.fontset':'stix','font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
def load(p):return json.loads(p.read_text())
# Regions identify visible objects in the target render; all methods use the same coordinates.
ROI={'pair_06':(.21,.25,.65,.80),'pair_23':(.30,.22,.69,.76),'pair_39':(.27,.25,.70,.76),
 'add':(.36,.48,.62,.83),'delete':(.36,.29,.63,.64),'move':(.34,.27,.64,.67),'multi':(.34,.26,.64,.66)}
LABEL={'add':'Addition','delete':'Removal','move':'Relocation','multi':'Mixed changes',
 'pair_06':'Sofa / floor','pair_23':'Chair / texture','pair_39':'Room structure'}
def plot(dataset,names,filename):
 # A narrow context column and two comparison columns fit a real paper column.
 # Each scene has FCGS / FCGS+RevisitGS above L-GSC / L-GSC+RevisitGS.
 fig=plt.figure(figsize=(3.5,1.43*len(names)+.14))
 outer=fig.add_gridspec(len(names),1,hspace=.36)
 for i,name in enumerate(names):
  gs=outer[i].subgridspec(2,3,width_ratios=[.59,1,1],wspace=.09,hspace=.90)
  path=R/'figure_data'/name;data=np.load(path/'images.npz');target=data['target'];crops=data['crops'];met=load(path/'metrics.json');rate=met['stream_mib']
  h,w=target.shape[:2];x0,y0,x1,y1=ROI[name]
  # The same target-defined horizontal detail is shown for every decoded stream.
  cy=(y0+y1)/2; half=(x1-x0)*w/(2*2.4*h)
  y0,y1=max(0,cy-half),min(1,cy+half)
  x0,x1=int(x0*w),int(x1*w);y0,y1=int(y0*h),int(y1*h)
  ax=fig.add_subplot(gs[:,0])
  if dataset=='clsplats':
   mask=np.any(target<.93,axis=2);ys,xs=np.where(mask);l=max(0,xs.min()-12);r=min(w,xs.max()+13);t=max(0,ys.min()-12);b=min(h,ys.max()+13)
  else:l,r,t,b=0,w,0,h
  ax.imshow(target[t:b,l:r],interpolation='nearest');ax.add_patch(Rectangle((x0-l,y0-t),x1-x0,y1-y0,fill=False,ec='#c44740',lw=.7));ax.axis('off')
  nameshort={'add':'Addition','delete':'Removal','move':'Move','multi':'Mixed', 'pair_06':'Sofa','pair_23':'Chair','pair_39':'Room'}[name]
  ax.set_title(nameshort,fontsize=8,pad=2)
  for j in range(1,5):
   row,col=divmod(j-1,2);ax=fig.add_subplot(gs[row,col+1]);crop=crops[j]
   ax.imshow(crop,interpolation='nearest');ax.axis('off')
   if col==1:ax.add_patch(Rectangle((0,0),1,1,transform=ax.transAxes,fill=False,ec='#B2182B',lw=.65,clip_on=False))
   if i==0 and row==0:ax.text(.5,1.34,['Intra codec','RevisitGS'][col],transform=ax.transAxes,ha='center',va='bottom',fontsize=8,color='#B2182B' if col==1 else '#2166AC',fontweight='bold' if col==1 else 'normal')
   ax.text(.5,1.035,f"{rate[j-1]:.2f} MiB | {met['psnr'][j-1]:.2f} dB",transform=ax.transAxes,ha='center',va='bottom',fontsize=7.5)
   mae=np.mean(np.abs(crop-crops[0]))
   ax.text(.5,-.045,f"MAE {mae:.4f}",transform=ax.transAxes,ha='center',va='top',fontsize=7.5)
 fig.subplots_adjust(left=.005,right=.995,top=.91,bottom=.04)
 for ext in ['pdf','png']:fig.savefig(O/f'{filename}.{ext}',dpi=360,bbox_inches='tight',pad_inches=.025)
 plt.close(fig)
def main():
 global O
 parser=argparse.ArgumentParser(description="Redraw the paper's decoded RGB comparisons from saved render crops.")
 parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args();O=args.output;O.mkdir(parents=True,exist_ok=True)
 plot('clsplats',['add','delete','move','multi'],'clsplats_qualitative')
 plot('3rscan',['pair_06','pair_23','pair_39'],'qualitative_3rscan_evidence')
if __name__=='__main__':main()
