#!/usr/bin/env python3
"""Evaluate saved AAM-S in physical seal units and write compact JSON."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent)); from train_aam_s import load,predict,r2
def main():
 a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--dataset',required=True);a.add_argument('--model',required=True);a.add_argument('--out',required=True);z=a.parse_args();root=Path(z.root);ds=Path(z.dataset); q=np.load(Path(z.model)/'aam_s.npz');p={k:q[k] for k in ('w1','b1','w2','b2')}; x,y,g=load(root,ds,'test',80); pn=predict(p,x,q['x_mean'],q['x_std']); norm=json.loads((ds/'normalization.json').read_text());mean,std=norm['mean'][2],norm['std'][2];yt=y[:,0]*std+mean;yp=pn[:,0]*std+mean
 ids=np.unique(g); yr=np.array([yt[g==i][0] for i in ids]); pr=np.array([yp[g==i].mean() for i in ids]); err=pr-yr
 out={'seed':json.loads((Path(z.model)/'metrics.json').read_text())['seed'],'test_r2':r2(yr[:,None],pr[:,None]),'mae':float(np.abs(err).mean()),'rmse':float(np.sqrt((err**2).mean())),'bias':float(err.mean()),'pearson':float(np.corrcoef(yr,pr)[0,1]),'per_run':[{'run_id':int(i),'true_seal':float(t),'estimated_seal':float(p)} for i,t,p in zip(ids,yr,pr)]}
 Path(z.out).write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
