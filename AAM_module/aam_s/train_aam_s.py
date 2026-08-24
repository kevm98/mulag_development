#!/usr/bin/env python3
"""Dependency-free CPU training for AAM-S: 20-row causal H0 -> normalized seal."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

SEAL_INDEX = 2

def load(root, ds, split, samples):
    ids=json.loads((root/'splits'/f'{split}.json').read_text()); xs=[]; ys=[]; groups=[]
    for rid in ids:
        with np.load(ds/f'{rid:05d}/history_L20.npz',allow_pickle=False) as d:
            x=np.asarray(d['window_input'],np.float32); y=float(d['e_target_normalized'][0,SEAL_INDEX])
        ix=np.linspace(0,len(x)-1,min(samples,len(x)),dtype=int); xs.append(x[ix]); ys.extend([y]*len(ix)); groups.extend([rid]*len(ix))
    return np.concatenate(xs),np.asarray(ys,np.float32)[:,None],np.asarray(groups)

def predict(params,x,xm,xs):
    a=(x-xm)/xs @ params['w1']+params['b1']; return np.maximum(a,0)@params['w2']+params['b2']

def r2(y,p): return float(1-((y-p)**2).sum()/max(((y-y.mean())**2).sum(),1e-12))

def run(seed, root, ds, out):
    rng=np.random.default_rng(seed); tx,ty,_=load(root,ds,'train',24); vx,vy,vg=load(root,ds,'val',80)
    xm=tx.mean(0); xs=tx.std(0).clip(1e-6); d=tx.shape[1]; h=32
    p={'w1':rng.normal(0,1/np.sqrt(d),(d,h)).astype(np.float32),'b1':np.zeros(h,np.float32),'w2':rng.normal(0,1/np.sqrt(h),(h,1)).astype(np.float32),'b2':np.zeros(1,np.float32)}
    m={k:np.zeros_like(v) for k,v in p.items()}; v={k:np.zeros_like(z) for k,z in p.items()}; best=None; history=[]; step=0; stale=0
    for epoch in range(100):
        order=rng.permutation(len(tx))
        for st in range(0,len(order),128):
            ix=order[st:st+128]; xn=(tx[ix]-xm)/xs; a=xn@p['w1']+p['b1']; z=np.maximum(a,0); q=z@p['w2']+p['b2']; grad=2*(q-ty[ix])/len(ix); back=(grad@p['w2'].T)*(a>0)
            g={'w2':z.T@grad,'b2':grad.sum(0),'w1':xn.T@back,'b1':back.sum(0)}; step+=1
            for k in p:
                m[k]=.9*m[k]+.1*g[k]; v[k]=.999*v[k]+.001*g[k]**2; p[k]-=2e-3*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
        vp=predict(p,vx,xm,xs); loss=float(((vp-vy)**2).mean()); history.append({'epoch':epoch+1,'val_mse':loss,'val_r2_window':r2(vy,vp)})
        if best is None or loss < best[0]-1e-6: best=(loss,{k:z.copy() for k,z in p.items()},epoch+1); stale=0
        else: stale+=1
        if stale>=15: break
    p=best[1]; out.mkdir(parents=True,exist_ok=True); np.savez(out/'aam_s.npz',**p,x_mean=xm,x_std=xs)
    (out/'metrics.json').write_text(json.dumps({'seed':seed,'architecture':'causal-history MLP (20 x 18 flattened newest-first)','history_rows':20,'feature_order':'joint_q(5), joint_qdot(5), dP(4), previous_valve(4)','target':'normalized seal','best_epoch':best[2],'history':history,'val_mse':best[0]},indent=2)+'\n')

def main():
 a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--dataset',required=True);a.add_argument('--seed',type=int,required=True);a.add_argument('--out',required=True);z=a.parse_args();run(z.seed,Path(z.root),Path(z.dataset),Path(z.out))
if __name__=='__main__':main()
