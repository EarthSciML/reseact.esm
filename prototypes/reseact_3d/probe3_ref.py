"""Independent reference for the Mx/My/Mz forcing, computed from the raw netCDF with
ncdump -- deliberately NOT sharing code with the .esm expressions."""
import subprocess, json, re

def dump(f, v):
    """Return the variable's flat float list (C order = the file's dim order)."""
    out = subprocess.run(["ncdump","-p","9,17","-v",v,f], capture_output=True, text=True).stdout
    body = out.split(f" {v} =",1)[1]
    body = body.split(";",1)[0]
    return [float(x) for x in body.replace("\n"," ").split(",")]

T0 = 64800.0 + 0.5   # midpoint: the probe integrates over [T0, T0+1]
# Bracket records + weights (0-based record indices), matching the runner.
I3_r0, I3_w = 6, ((T0 -    0.0)/10800.0) % 1.0
A3_r0, A3_w = 5, ((T0 - 5400.0)/10800.0) % 1.0
A1_r0, A1_w = 17, ((T0 - 1800.0)/ 3600.0) % 1.0
NLAT_F, NLON_F, NLEV_F = 46, 72, 72
A = 6371000.0

PS = dump("gf_I3.nc","PS")       # (8,46,72)
U  = dump("gf_A3dyn.nc","U")     # (8,72,46,72)
V_ = dump("gf_A3dyn.nc","V")
OM = dump("gf_A3dyn.nc","OMEGA")
BL = dump("gf_A1.nc","PBLH")     # (24,46,72)

def ps(rec, j1, i1):   # 1-based native lat/lon
    return PS[rec*NLAT_F*NLON_F + (j1-1)*NLON_F + (i1-1)]
def w3(arr, rec, k1, j1, i1):
    return arr[rec*NLEV_F*NLAT_F*NLON_F + (k1-1)*NLAT_F*NLON_F + (j1-1)*NLON_F + (i1-1)]
def bl(rec, j1, i1):
    return BL[rec*NLAT_F*NLON_F + (j1-1)*NLON_F + (i1-1)]

co = json.load(open("hybrid_coefs.json")); dA, dB = co["dA"], co["dB"]
PSb  = lambda j1,i1: 100.0*((1-I3_w)*ps(I3_r0,j1,i1) + I3_w*ps(I3_r0+1,j1,i1))
dpb  = lambda k,j1,i1: dA[k-1] + dB[k-1]*PSb(j1,i1)
Ub   = lambda k,j1,i1: (1-A3_w)*w3(U,A3_r0,k,j1,i1) + A3_w*w3(U,A3_r0+1,k,j1,i1)
Vb   = lambda k,j1,i1: (1-A3_w)*w3(V_,A3_r0,k,j1,i1)+ A3_w*w3(V_,A3_r0+1,k,j1,i1)
OMb  = lambda k,j1,i1: (1-A3_w)*w3(OM,A3_r0,k,j1,i1)+ A3_w*w3(OM,A3_r0+1,k,j1,i1)

def Mx(ie,j,k):
    nL,nR,nj = 13+ie, 14+ie, 29+j
    return 0.5*(dpb(k,nj,nL)+dpb(k,nj,nR)) * 0.5*(Ub(k,nj,nL)+Ub(k,nj,nR)) / A
def My(i,je,k):
    jL,jR,ni = 28+max(je,2), 29+min(je,7), 14+i
    return 0.5*(dpb(k,jL,ni)+dpb(k,jR,ni)) * 0.5*(Vb(k,jL,ni)+Vb(k,jR,ni)) / A
def Mz(i,j,ke):
    kL,kR,nj,ni = max(ke-1,1), min(ke,7), 29+j, 14+i
    return -0.5*(OMb(kL,nj,ni)+OMb(kR,nj,ni))

ref = {
 "pMx":  Mx(3,4,1),  "pMy": My(3,4,1),  "pMz": Mz(3,4,2),
 "pPS":  PSb(29+4, 14+3), "pdp": dpb(1, 29+4, 14+3),
 "pBL":  (1-A1_w)*bl(A1_r0,29+4,14+3) + A1_w*bl(A1_r0+1,29+4,14+3),
 "pMxw": Mx(1,4,1),  "pMys": My(3,1,1),
}
json.dump(ref, open("probe3_ref.json","w"), indent=1)
for k,v in ref.items(): print(f"  {k:6s} = {v!r}")
