"""R3 uniform checkpoint rule: the saved checkpoint nearest the training-reward peak."""
import re,sys
IT=re.compile(r"Learning iteration (\d+)/")
MR=re.compile(r"Mean reward:\s*(-?[\d.]+)")
SAVED=list(range(0,400,25))+[399]
def peak(path):
    it=None; best=(None,-1e18); n=0
    for line in open(path,errors="ignore"):
        m=IT.search(line)
        if m: it=int(m.group(1)); continue
        m=MR.search(line)
        if m and it is not None:
            r=float(m.group(1)); n+=1
            if r>best[1]: best=(it,r)
    ck=min(SAVED,key=lambda c:(abs(c-best[0]),c))
    return best[0],best[1],ck,n
for p in sys.argv[1:]:
    i,r,ck,n=peak(p)
    print(f"{p:42s} peak iter {i:3d}  reward {r:9.1f}  -> model_{ck}.pt   ({n} pts)")
