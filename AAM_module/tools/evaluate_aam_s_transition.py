#!/usr/bin/env python3
"""Offline seal-estimator transition test by phase-aligned held-out splicing."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'aam_s')); from train_aam_s import predict
def main():
 a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--dataset',required=True);a.add_argument('--models',nargs='+',required=True);a.add_argument('--out',required=True);z=a.parse_args();root=Path(z.root);ds=Path(z.dataset);ids=json.loads((root/'splits/test.json').read_text()); records=[]; series=[]
 vals=[]
 for rid in ids:
  with np.load(ds/f'{rid:05d}/history_L20.npz') as d: vals.append((float(d['e_target'][0,2]),rid,np.asarray(d['window_input']).reshape(-1,20,18)[:,0]))
 vals.sort(); pairs=list(zip(vals[:8],vals[-8:]))
 for lo,hi in pairs:
  for A,B in ((lo,hi),(hi,lo)):
   sa,ra,xa=A; sb,rb,xb=B; c=100; raw=np.concatenate((xa[1800:1900],xb[1900:2220])); pred=[]
   for m in z.models:
    q=np.load(Path(m)/'aam_s.npz'); pred.append(np.array([predict({k:q[k] for k in ('w1','b1','w2','b2')},raw[max(0,i-19):i+1].reshape(1,-1) if i>=19 else np.pad(raw[:i+1],((19-i,0),(0,0))).reshape(1,-1),q['x_mean'],q['x_std'])[0,0] for i in range(len(raw))]))
   norm=json.loads((ds/'normalization.json').read_text()); pe=np.mean(pred,0)*norm['std'][2]+norm['mean'][2]; true=np.r_[np.full(c,sa),np.full(len(raw)-c,sb)]; delta=sb-sa; frac=(pe-sa)/delta; after=frac[c:]; t50=next((i*.05 for i,v in enumerate(after) if v>=.5),None);t90=next((i*.05 for i,v in enumerate(after) if v>=.9),None); tol=.1*abs(delta); hit=next((i*.05 for i,v in enumerate(pe[c:]) if abs(v-sb)<=tol),None); records.append({'from_run':ra,'to_run':rb,'from_seal':sa,'to_seal':sb,'t50_s':t50,'t90_s':t90,'tolerance_s':hit,'direction_correct':bool(np.sign(pe[-1]-pe[c-1])==np.sign(delta)),'converged':hit is not None});series.append({'true':true.tolist(),'estimated':pe.tolist()})
 def med(k):
  x=[r[k] for r in records if r[k] is not None]; return float(np.median(x)) if x else None
 Path(z.out).write_text(json.dumps({'definition':'offline phase-aligned held-out splice; not a physical or closed-loop test','tolerance':'10% of seal-step magnitude','dt_s':.05,'transitions':records,'median_t50_s':med('t50_s'),'median_t90_s':med('t90_s'),'median_tolerance_s':med('tolerance_s'),'success_fraction':float(np.mean([r['converged'] for r in records])),'series':series},indent=2)+'\n')
if __name__=='__main__':main()
