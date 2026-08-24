#!/usr/bin/env python3
"""Evaluate privileged/mean/shuffled/base contexts on held-out ROM runs."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'AAM_module')]
from aam_data import iter_raw_runs,load_runs_info,read_raw_run
from aam_data.timebase import rom_tick_indices
from mulag_plant.runtime.cylinder_models_runtime import CylinderIDRuntime,CylinderFDRuntime
from tools.train_stage29_latent_adapter import runtime_mean,fd_delta_with_grad
from train_privileged_adapter import PrivilegedAdaptationEncoder,Residual,f
K=('flow','compliance','seal','pressure','mass_scale','com_z')
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--model',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.root);out=Path(a.out);ck=torch.load(Path(a.model),map_location='cpu',weights_only=False);d=ck['dim'];enc=PrivilegedAdaptationEncoder(d);ad=Residual(d);enc.load_state_dict(ck['state']['enc']);ad.load_state_dict(ck['state']['ad']);enc.eval();ad.eval();fm=torch.tensor(ck['factor_mean']);fs=torch.tensor(ck['factor_std']);xm=torch.tensor(ck['feature_mean']);xs=torch.tensor(ck['feature_std']);bound=float(ck['bound']);lim=torch.tensor([[.95,.95,.8,.8]])
 info=load_runs_info(root);ids=[int(r) for r,_ in iter_raw_runs(root) if info[f'{int(r):05d}']['frozen_split']=='test'];raw={r:read_raw_run(root,r,verify=True) for r in ids};factors={r:torch.tensor(f(raw[r]['metadata'])) for r in ids};mean=torch.tensor([float(raw[r]['metadata'][k]) for k in info if False]) if False else torch.tensor([np.mean([f(raw[x]['metadata'])[j] for x in ids]) for j in range(6)]);donor={r:ids[(i+1)%len(ids)] for i,r in enumerate(ids)};idr=CylinderIDRuntime(models_root=ROOT/'id_fd_model/vendor_models',checkpoint=ROOT/'id_fd_model/best_id_cylinder_control.pt',num_envs=1,device='cpu');fd=CylinderFDRuntime(models_root=ROOT/'id_fd_model/vendor_models',checkpoint=ROOT/'id_fd_model/best_fd_cylinder_strict.pt',device='cpu');dp=runtime_mean(idr,[f'dP_{i}' for i in range(1,5)]);force=runtime_mean(idr,[f'Force_{i}' for i in range(1,5)]);result={k:[] for k in ('base','privileged','mean','shuffled')}
 for rid in ids:
  a0=raw[rid]['arrays'];ticks=rom_tick_indices(a0);x0=np.asarray(a0['cylinder_extension'][ticks],np.float32);v0=np.asarray(a0['cylinder_velocity'][ticks],np.float32);tx=np.asarray(a0['desired_cylinder_position'][ticks],np.float32);tv=np.asarray(a0['desired_cylinder_velocity'][ticks],np.float32)
  for name in result:
   x=x0[0].copy();v=v0[0].copy();prev=np.zeros(4,np.float32);errs=[];ctx={'privileged':factors[rid],'mean':mean,'shuffled':factors[donor[rid]]}.get(name)
   for k in range(min(len(ticks),400)):
    rawin=np.r_[x,v,dp,force,tx[k]-x,tv[k]-v,prev].astype(np.float32);base=idr.predict(torch.tensor(rawin[None]),prev_valve_cmd=torch.tensor(prev[None]));u=base
    if ctx is not None and k>=19:
     z=enc(((ctx-fm)/fs).reshape(1,-1));feat=torch.cat(((torch.tensor(rawin[:28]).reshape(1,-1)-xm)/xs,z),1);u=torch.clamp(base+bound*torch.tanh(ad(feat)),-lim,lim)
    nxt=fd.step(valve_cmd=u,extension=torch.tensor(x[None]),extension_velocity=torch.tensor(v[None]),dP=torch.tensor(dp[None]),force=torch.tensor(force[None])).raw_output[0].detach().numpy();x=x+nxt[:4];v=v+nxt[4:8];prev=u[0].detach().numpy();errs.append(x-tx[k])
   e=np.asarray(errs);result[name].append({'run_id':rid,**{h:float(np.sqrt((e[:n]**2).mean())) for h,n in (('H20_1s',20),('H100_5s',100),('H400_20s',400))}})
 summary={k:{h:float(np.mean([r[h] for r in v])) for h in ('H20_1s','H100_5s','H400_20s')} for k,v in result.items()};out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'format':'privileged_control_latent_ceiling_v1','test_runs':ids,'summary':summary,'per_run':result},indent=2)+'\n')
if __name__=='__main__':main()
