#!/usr/bin/env python3
"""Versioned scalar-seal bounded-adapter oracle and estimated-context test."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[2]; sys.path[:0]=[str(ROOT),str(ROOT/'AAM_module'),str(ROOT/'AAM_module/aam_s')]
from aam_data import iter_raw_runs,load_runs_info,read_raw_run
from mulag_plant.runtime.cylinder_models_runtime import CylinderIDRuntime,CylinderFDRuntime
from train_aam_s import predict
from tools.train_stage29_latent_adapter import runtime_mean,make_run_samples,train_one,ResidualAdapter,fd_delta_with_grad

def maps(ds, models, rid, kind, donor=None):
 with np.load(ds/f'{rid:05d}/history_L20.npz') as d:
  ticks=np.asarray(d['window_end_tick'],int); true=float(d['e_target_normalized'][0,2]); x=np.asarray(d['window_input'],np.float32)
 if kind=='true': v=np.full(len(ticks),true)
 elif kind=='mean': v=np.zeros(len(ticks))
 elif kind=='shuffled':
  with np.load(ds/f'{donor:05d}/history_L20.npz') as d: v=np.full(len(ticks),float(d['e_target_normalized'][0,2]))
 else:
  out=[]
  for m in models:
   q=np.load(Path(m)/'aam_s.npz'); p={k:q[k] for k in ('w1','b1','w2','b2')};out.append(predict(p,x,q['x_mean'],q['x_std'])[:,0])
  v=np.mean(out,0)
 return {int(t):np.array([float(a)]) for t,a in zip(ticks,v)}

def collect(root,ds,models,info,idrt,dp,force,kind,split):
 xs=[];ys=[];us=[]; ids=[int(r) for r in info if info[r]['frozen_split']==split]; donor={r:ids[(i+1)%len(ids)] for i,r in enumerate(sorted(ids))}
 for rid in ids:
  a=read_raw_run(root,rid,verify=True)['arrays']; x,y,u=make_run_samples(a,id_runtime=idrt,dp_mean=dp,force_mean=force,latent_by_tick=maps(ds,models,rid,kind,donor.get(rid)),run_id=rid,lag=20,smoothing_window=1)
  # Equal, deterministic support from each plant/run: the independent unit is
  # the run, and full trajectories would otherwise make this CPU gate needlessly
  # expensive without adding independent identities.
  ix=np.linspace(0,len(x)-1,min(128,len(x)),dtype=int);xs.append(x[ix].astype(np.float32));ys.append(y[ix].astype(np.float32));us.append(u[ix].astype(np.float32))
 return np.concatenate(xs),np.concatenate(ys),np.concatenate(us),ids

def score(path,x,y,u,fd):
 p=torch.load(path,map_location='cpu',weights_only=False);m=ResidualAdapter(**p['model_config']).eval();m.load_state_dict(p['model_state']);xm=p['feature_mean'];xs=p['feature_std'];bound=p['residual_bound'];lim=p['command_limits'];
 with torch.no_grad():
  xt=torch.as_tensor(x); ut=torch.as_tensor(u); pred=fd_delta_with_grad(fd,xt,torch.clamp(ut+bound*torch.tanh(m((xt-xm)/xs)),-lim,lim));base=fd_delta_with_grad(fd,xt,ut);t=torch.as_tensor(y);return float(((pred-t)**2).mean()),float(((base-t)**2).mean())
def main():
 a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--dataset',required=True);a.add_argument('--models',nargs='+',required=True);a.add_argument('--out',required=True);z=a.parse_args();root=Path(z.root);ds=Path(z.dataset);out=Path(z.out);out.mkdir(parents=True,exist_ok=True);info=load_runs_info(root);idc=ROOT/'id_fd_model/best_id_cylinder_control.pt';fdc=ROOT/'id_fd_model/best_fd_cylinder_strict.pt';idrt=CylinderIDRuntime(models_root=ROOT/'id_fd_model/vendor_models',checkpoint=idc,num_envs=1,device='cpu');fd=CylinderFDRuntime(models_root=ROOT/'id_fd_model/vendor_models',checkpoint=fdc,device='cpu');idrt.model.requires_grad_(False);fd.model.requires_grad_(False);fd.stage29_id_checkpoint=str(idc);dp=runtime_mean(idrt,[f'dP_{i}' for i in range(1,5)]);force=runtime_mean(idrt,[f'Force_{i}' for i in range(1,5)]);result={}
 for i,k in enumerate(('true','mean','shuffled','estimated')):
  tr=collect(root,ds,z.models,info,idrt,dp,force,k,'train');va=collect(root,ds,z.models,info,idrt,dp,force,k,'val');r=train_one(k,*tr[:3],*va[:3],fd=fd,device=torch.device('cpu'),out=out,hidden=32,depth=2,epochs=60,batch_size=8192,lr=1e-3,patience=10,residual_bound=.15,seed=20260900+i,latent_lag=20,smoothing_window=1);te=collect(root,ds,z.models,info,idrt,dp,force,k,'test');adapt,base=score(out/f'{k}_adapter.pt',*te[:3],fd);result[k]={'test_mse':adapt,'base_mse':base,'checkpoint':r['checkpoint']}
 (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
