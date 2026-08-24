#!/usr/bin/env python3
"""CPU-only privileged control-latent ceiling, without any history AAM."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]; sys.path[:0]=[str(ROOT),str(ROOT/'AAM_module')]
from aam_data import iter_raw_runs, load_runs_info, read_raw_run
from aam_data.timebase import rom_tick_indices
from mulag_plant.runtime.cylinder_models_runtime import CylinderIDRuntime,CylinderFDRuntime
from tools.train_stage29_latent_adapter import runtime_mean,make_run_samples,fd_delta_with_grad
K=('flow','compliance','seal','pressure','mass_scale','com_z')

class PrivilegedAdaptationEncoder(nn.Module):
 def __init__(self,d): super().__init__(); self.net=nn.Sequential(nn.Linear(6,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,d))
 def forward(self,x): return self.net(x)
class Residual(nn.Module):
 def __init__(self,d): super().__init__(); self.net=nn.Sequential(nn.Linear(28+d,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU(),nn.Linear(32,4));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,x): return self.net(x)
def f(meta): return np.array([meta[x] for x in K],np.float32)
def samples(root,info,idr,dp,force,split,n=128):
 X=[];Y=[];U=[];F=[]
 for rid,_ in iter_raw_runs(root):
  rid=int(rid)
  if info[f'{rid:05d}']['frozen_split']!=split: continue
  q=read_raw_run(root,rid,verify=True); ticks=rom_tick_indices(q['arrays']); dummy={int(t):np.zeros(1) for t in ticks[19:]}
  x,y,u=make_run_samples(q['arrays'],id_runtime=idr,dp_mean=dp,force_mean=force,latent_by_tick=dummy,run_id=rid,lag=20,smoothing_window=1); ix=np.linspace(0,len(x)-1,min(n,len(x)),dtype=int)
  X.append(x[ix,:28]);Y.append(y[ix]);U.append(u[ix]);F.append(np.repeat(f(q['metadata'])[None],len(ix),0))
 return tuple(np.concatenate(a) for a in (X,Y,U,F))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--dim',type=int,default=4);p.add_argument('--seed',type=int,default=20260910);a=p.parse_args();root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);torch.manual_seed(a.seed)
 info=load_runs_info(root);idr=CylinderIDRuntime(models_root=ROOT/'id_fd_model/vendor_models',checkpoint=ROOT/'id_fd_model/best_id_cylinder_control.pt',num_envs=1,device='cpu');fd=CylinderFDRuntime(models_root=ROOT/'id_fd_model/vendor_models',checkpoint=ROOT/'id_fd_model/best_fd_cylinder_strict.pt',device='cpu');dp=runtime_mean(idr,[f'dP_{i}' for i in range(1,5)]);force=runtime_mean(idr,[f'Force_{i}' for i in range(1,5)]);tr=samples(root,info,idr,dp,force,'train');va=samples(root,info,idr,dp,force,'val');fm,fs=tr[3].mean(0),tr[3].std(0).clip(1e-6);xm,xs=tr[0].mean(0),tr[0].std(0).clip(1e-6);ys=tr[1].std(0).clip(1e-5)
 enc=PrivilegedAdaptationEncoder(a.dim);ad=Residual(a.dim);opt=torch.optim.AdamW(list(enc.parameters())+list(ad.parameters()),lr=1e-3,weight_decay=1e-4);bound=torch.full((1,4),.15);lim=torch.tensor([[.95,.95,.8,.8]]);fm,fs,xm,xs,ys=(torch.tensor(z,dtype=torch.float32) for z in (fm,fs,xm,xs,ys));T=[torch.tensor(z,dtype=torch.float32) for z in tr];V=[torch.tensor(z,dtype=torch.float32) for z in va];best=(1e30,None);hist=[]
 for ep in range(50):
  order=torch.randperm(len(T[0])); tot=0.
  for s in range(0,len(order),512):
   ix=order[s:s+512];z=enc((T[3][ix]-fm)/fs);r=bound*torch.tanh(ad(torch.cat(((T[0][ix]-xm)/xs,z),1)));pred=fd_delta_with_grad(fd,T[0][ix],torch.clamp(T[2][ix]+r,-lim,lim));loss=(((pred-T[1][ix])/ys)**2).mean()+.01*(r/bound).square().mean();opt.zero_grad();loss.backward();opt.step();tot+=float(loss)*len(ix)
  with torch.no_grad():
   z=enc((V[3]-fm)/fs);r=bound*torch.tanh(ad(torch.cat(((V[0]-xm)/xs,z),1)));pred=fd_delta_with_grad(fd,V[0],torch.clamp(V[2]+r,-lim,lim));vl=float((((pred-V[1])/ys)**2).mean())
  hist.append({'epoch':ep,'train':tot/len(T[0]),'val':vl});print(ep,vl,flush=True)
  if vl<best[0]:best=(vl,{'enc':enc.state_dict(),'ad':ad.state_dict(),'epoch':ep})
  elif ep-best[1]['epoch']>=10:break
 torch.save({'dim':a.dim,'state':best[1],'factor_mean':fm,'factor_std':fs,'feature_mean':xm,'feature_std':xs,'bound':.15,'history':hist},out/'model.pt');(out/'metrics.json').write_text(json.dumps({'dim':a.dim,'seed':a.seed,'best_val':best[0],'history':hist},indent=2)+'\n')
if __name__=='__main__':main()
