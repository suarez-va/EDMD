import numpy as np
from model_systems.models import GICD
from time_independent.fcidvr import validate_nele_spin, fci_mapping, fci_operator, fci_wfn, fci_1rdm, fci_dyson, FCIDVR#, compute_cap, compute_1rdm, compute_rho, compute_dyson
import matplotlib.pyplot as plt
import time


params = {
    'ZA': 0.0,
    'ZB': 0.0,
    'aR': 0.0,
    'bR': 0.0001,
    'DA' : 1.0,
    'bA': 0.25,
    'DB' : 0.8,
    'bB': 1.0,
    'aee': 0.0,
    'bee': 0.0001,
}

R_sub = 8.0
nbo_sub1 = 25
nbo_sub2 = 7
nbo_sub3 = 3

model = GICD(params)
fcidvr = FCIDVR(model = model, xa = -196.7/8.0, xb = 196.7/8.0, xN = 25, xbounds = "(-inf,inf)")

ndvr = fcidvr.nxdvr
hij = fcidvr.hij(R_sub)
gik = fcidvr.gik()
nfci_1eleD, map_norm_1eleD, map_idx_1eleD = fci_mapping(ndvr = ndvr, nele = 1, spin = 'doublet')
nfci_2eleS, map_norm_2eleS, map_idx_2eleS = fci_mapping(ndvr = ndvr, nele = 2, spin = 'singlet')
nfci_2eleT, map_norm_2eleT, map_idx_2eleT = fci_mapping(ndvr = ndvr, nele = 2, spin = 'triplet')
nfci_3eleD, map_norm_3eleD, map_idx_3eleD = fci_mapping(ndvr = ndvr, nele = 3, spin = 'doublet')
print(nfci_1eleD)
print(nfci_2eleS)
print(nfci_2eleT)
print(nfci_3eleD)

#time1=time.time()
#fcidvr.kernel(R = R_sub, nele = 1, spin = 'doublet', nbo = nbo_sub1, derivative_order = 0)
#time2=time.time()
#print(f'time: {(time2 - time1)/60.0} min')
#
#time1=time.time()
#fcidvr.kernel(R = R_sub, nele = 2, spin = 'singlet', nbo = nbo_sub2, derivative_order = 0)
#time2=time.time()
#print(f'time: {(time2 - time1)/60.0} min')
#
#time1=time.time()
#fcidvr.kernel(R = R_sub, nele = 2, spin = 'triplet', nbo = nbo_sub2, derivative_order = 0)
#time2=time.time()
#print(f'time: {(time2 - time1)/60.0} min')
##print(f'predicted time: {(time2 - time1)/60.0*(2*(nfci_3ele)**3/(nfci_2ele)**3)} min')
#
#time1=time.time()
#fcidvr.kernel(R = R_sub, nele = 3, spin = 'doublet', nbo = nbo_sub3, derivative_order = 0)
#time2=time.time()
#print(f'time: {(time2 - time1)/60.0} min')

fcidvr_data1D = np.load('fcidvr_1ele_doublet.npz')
fcidvr_data2S = np.load('fcidvr_2ele_singlet.npz')
fcidvr_data2T = np.load('fcidvr_2ele_triplet.npz')
fcidvr_data3D = np.load('fcidvr_3ele_doublet.npz')

H1D = fci_operator(ndvr, 1, 'doublet', hij, gik)
H2S = fci_operator(ndvr, 2, 'singlet', hij, gik)
H2T = fci_operator(ndvr, 2, 'triplet', hij, gik)
H3D = fci_operator(ndvr, 3, 'doublet', hij, gik)

n = 1
m = 1
E1D = fcidvr_data1D["En"]; C1D = fcidvr_data1D["Cn"]; wfn1Dn = fci_wfn(ndvr, 1, 'doublet', 0.5, C1D[:,n]); wfn1Dm = fci_wfn(ndvr, 1, 'doublet', 0.5, C1D[:,m])
E2S = fcidvr_data2S["En"]; C2S = fcidvr_data2S["Cn"]; wfn2Sn = fci_wfn(ndvr, 2, 'singlet', 0.0, C2S[:,n]); wfn2Sm = fci_wfn(ndvr, 2, 'singlet', 0.0, C2S[:,m])
E2T = fcidvr_data2T["En"]; C2T = fcidvr_data2T["Cn"]; wfn2Tn = fci_wfn(ndvr, 2, 'triplet', 1.0, C2T[:,n]); wfn2Tm = fci_wfn(ndvr, 2, 'triplet', 1.0, C2T[:,m])
E3D = fcidvr_data3D["En"]; C3D = fcidvr_data3D["Cn"]; wfn3Dn = fci_wfn(ndvr, 3, 'doublet', 0.5, C3D[:,n]); wfn3Dm = fci_wfn(ndvr, 3, 'doublet', 0.5, C3D[:,m])

HC1Dm = H1D.matvec(C1D[:,m]); Hwfn1Dm = fci_wfn(ndvr, 1, 'doublet', 0.5, HC1Dm)
HC2Sm = H2S.matvec(C2S[:,m]); Hwfn2Sm = fci_wfn(ndvr, 2, 'singlet', 0.0, HC2Sm)
HC2Tm = H2T.matvec(C2T[:,m]); Hwfn2Tm = fci_wfn(ndvr, 2, 'triplet', 1.0, HC2Tm)
HC3Dm = H3D.matvec(C3D[:,m]); Hwfn3Dm = fci_wfn(ndvr, 3, 'doublet', 0.5, HC3Dm)

S1Dnm = np.sum(wfn1Dn.conj()*wfn1Dm); H1Dnm = np.sum(wfn1Dn.conj()*Hwfn1Dm)
S2Snm = np.sum(wfn2Sn.conj()*wfn2Sm); H2Snm = np.sum(wfn2Sn.conj()*Hwfn2Sm)
S2Tnm = np.sum(wfn2Tn.conj()*wfn2Tm); H2Tnm = np.sum(wfn2Tn.conj()*Hwfn2Tm)
S3Dnm = np.sum(wfn3Dn.conj()*wfn3Dm); H3Dnm = np.sum(wfn3Dn.conj()*Hwfn3Dm)

print(S1Dnm, H1Dnm, E1D[m])
print(S2Snm, H2Snm, E2S[m])
print(S2Tnm, H2Tnm, E2T[m])
print(S3Dnm, H3Dnm, E3D[m])



gam3D = fci_1rdm(ndvr = ndvr, nele = 3, spinbra = 'doublet', Szbra = 0.5, Cbra = C3D[:,n], spinket = 'doublet', Szket = 0.5, Cket = C3D[:,m])
#gam3Dtest = 3*np.sum(wfn3Dn[None,None,:,:,:,:,:,:].conj() * wfn3Dm[:,:,None,None,:,:,:,:], axis=(4,5,6,7))
#print(np.max(np.abs(gam3D)))
#print(np.sum(np.einsum('iaia->ia',gam3D)))
#print(np.sum(np.einsum('iaia->ia',gam3Dtest)))
#print(gam3D)
#print(np.max(np.abs(gam3Dtest - gam3D)))
#print(np.max(np.abs(gam3Dtest - gam3D.conj())))


D1Da2S = fci_dyson(ndvr = ndvr, nele = 2, spinbra = 'doublet', Szbra = 0.5, Cbra = C1D[:,n], spinket = 'singlet', Szket = 0.0, Cket = C2S[:,m])
D1Db2S = fci_dyson(ndvr = ndvr, nele = 2, spinbra = 'doublet', Szbra = -0.5, Cbra = C1D[:,n], spinket = 'singlet', Szket = 0.0, Cket = C2S[:,m])
D1Da2Taa = fci_dyson(ndvr = ndvr, nele = 2, spinbra = 'doublet', Szbra = 0.5, Cbra = C1D[:,n], spinket = 'triplet', Szket = 1.0, Cket = C2T[:,m])
D1Db2Taa = fci_dyson(ndvr = ndvr, nele = 2, spinbra = 'doublet', Szbra = -0.5, Cbra = C1D[:,n], spinket = 'triplet', Szket = 1.0, Cket = C2T[:,m])
D1Da2Tab = fci_dyson(ndvr = ndvr, nele = 2, spinbra = 'doublet', Szbra = 0.5, Cbra = C1D[:,n], spinket = 'triplet', Szket = 0.0, Cket = C2T[:,m])
D1Db2Tab = fci_dyson(ndvr = ndvr, nele = 2, spinbra = 'doublet', Szbra = -0.5, Cbra = C1D[:,n], spinket = 'triplet', Szket = 0.0, Cket = C2T[:,m])
D1Da2Tbb = fci_dyson(ndvr = ndvr, nele = 2, spinbra = 'doublet', Szbra = 0.5, Cbra = C1D[:,n], spinket = 'triplet', Szket = -1.0, Cket = C2T[:,m])
D1Db2Tbb = fci_dyson(ndvr = ndvr, nele = 2, spinbra = 'doublet', Szbra = -0.5, Cbra = C1D[:,n], spinket = 'triplet', Szket = -1.0, Cket = C2T[:,m])
#print(np.max(np.abs(D1Db2S*D1Da2S.conj())))
#print(np.max(np.abs(D1Da2Tab*D1Db2Tab.conj())))
print(D1Da2S[:,1])
print(D1Db2S[:,0])
print(D1Da2Tab[:,1])
print(D1Db2Tab[:,0])
#exit()

D2S3Da = fci_dyson(ndvr = ndvr, nele = 3, spinbra = 'singlet', Szbra = 0.0, Cbra = C2S[:,n], spinket = 'doublet', Szket = 0.5, Cket = C3D[:,m])
D2Taa3Da  = fci_dyson(ndvr = ndvr, nele = 3, spinbra = 'triplet', Szbra = 1.0, Cbra = C2T[:,n], spinket = 'doublet', Szket = 0.5, Cket = C3D[:,m])
D2Tab3Da  = fci_dyson(ndvr = ndvr, nele = 3, spinbra = 'triplet', Szbra = 0.0, Cbra = C2T[:,n], spinket = 'doublet', Szket = 0.5, Cket = C3D[:,m])
D2Tbb3Da  = fci_dyson(ndvr = ndvr, nele = 3, spinbra = 'triplet', Szbra = -1.0, Cbra = C2T[:,n], spinket = 'doublet', Szket = 0.5, Cket = C3D[:,m])
D2S3Db = fci_dyson(ndvr = ndvr, nele = 3, spinbra = 'singlet', Szbra = 0.0, Cbra = C2S[:,n], spinket = 'doublet', Szket = -0.5, Cket = C3D[:,m])
D2Taa3Db  = fci_dyson(ndvr = ndvr, nele = 3, spinbra = 'triplet', Szbra = 1.0, Cbra = C2T[:,n], spinket = 'doublet', Szket = -0.5, Cket = C3D[:,m])
D2Tab3Db  = fci_dyson(ndvr = ndvr, nele = 3, spinbra = 'triplet', Szbra = 0.0, Cbra = C2T[:,n], spinket = 'doublet', Szket = -0.5, Cket = C3D[:,m])
D2Tbb3Db  = fci_dyson(ndvr = ndvr, nele = 3, spinbra = 'triplet', Szbra = -1.0, Cbra = C2T[:,n], spinket = 'doublet', Szket = -0.5, Cket = C3D[:,m])
#print(D2Taa3Da)
#print(D2S3Da)
#print(D2Tab3Da)
#print(np.max(np.abs(D2Taa3Da)))
#print(np.max(np.abs(D2S3Da)))
#print(np.max(np.abs(D2Tab3Da)))
#print(np.max(np.abs(D2Tbb3Da)))
exit()


#D2Taa3Da
#
#D2S3Da
#D2Tab3Da

for p in range(3):
    for q in range(3):
        for n in range(3):
            for m in range(3):
                #D1Da2S_pn = fci_dyson(ndvr = ndvr, nele = 2, spinbra = 'doublet', Szbra = 0.5, Cbra = C1D[:,p], spinket = 'singlet', Szket = 0.0, Cket = C2S[:,n])
                #D1Db2S_qm = fci_dyson(ndvr = ndvr, nele = 2, spinbra = 'doublet', Szbra = -0.5, Cbra = C1D[:,q], spinket = 'singlet', Szket = 0.0, Cket = C2S[:,m])
                #print(np.max(np.abs(D1Da2S_pn*D1Db2S_qm.conj())))

                #D1Da2Tab_pn = fci_dyson(ndvr = ndvr, nele = 2, spinbra = 'doublet', Szbra = 0.5, Cbra = C1D[:,p], spinket = 'triplet', Szket = 0.0, Cket = C2T[:,n])
                #D1Db2Tab_qm = fci_dyson(ndvr = ndvr, nele = 2, spinbra = 'doublet', Szbra = -0.5, Cbra = C1D[:,q], spinket = 'triplet', Szket = 0.0, Cket = C2T[:,m])
                #print(np.max(np.abs(D1Da2Tab_pn*D1Db2Tab_qm.conj())))

                #D2S3Da_pn = fci_dyson(ndvr = ndvr, nele = 3, spinbra = 'singlet', Szbra = 0.0, Cbra = C2S[:,p], spinket = 'doublet', Szket = 0.5, Cket = C3D[:,n])
                #D2Taa3Da_qm  = fci_dyson(ndvr = ndvr, nele = 3, spinbra = 'triplet', Szbra = 1.0, Cbra = C2T[:,q], spinket = 'doublet', Szket = 0.5, Cket = C3D[:,m])
                #print(np.max(np.abs(D2S3Da_pn*D2Taa3Da_qm.conj())))

                D2S3Da_pn = fci_dyson(ndvr = ndvr, nele = 3, spinbra = 'singlet', Szbra = 0.0, Cbra = C2S[:,p], spinket = 'doublet', Szket = 0.5, Cket = C3D[:,n])
                D2Tab3Da_qm  = fci_dyson(ndvr = ndvr, nele = 3, spinbra = 'triplet', Szbra = 0.0, Cbra = C2T[:,q], spinket = 'doublet', Szket = 0.5, Cket = C3D[:,m])
                #print(np.max(np.abs(D2S3Da_pn*D2Tab3Da_qm.conj())))
                gam2S2T_pq = fci_1rdm(ndvr = ndvr, nele = 2, spinbra = 'singlet', Szbra = 0.0, Cbra = C2S[:,p], spinket = 'triplet', Szket = 0.0, Cket = C2T[:,q])
                gam2S2T_qp = fci_1rdm(ndvr = ndvr, nele = 2, spinbra = 'triplet', Szbra = 0.0, Cbra = C2T[:,q], spinket = 'singlet', Szket = 0.0, Cket = C2S[:,p])
                rho2S2T_pq = np.einsum('iaia->ia', gam2S2T_pq)
                #print(np.max(np.abs(gam2S2T_pq - np.einsum('iajb->jbia', gam2S2T_qp.conj()))))
                #print(np.max(np.abs(rho2S2T_pq)))
                #print(np.sum(rho2S2T_pq))



                #rho2D_pq = np.einsum('iaia->ia', fci_1rdm(ndvr = ndvr, nele = 2, spinbra = 'singlet', Szbra = 0.0, Cbra = C2S[:,p], spinket = 'triplet', Szket = 0.0, Cket = C2T[:,q]))
                #rho2D_qp = np.einsum('iaia->ia', fci_1rdm(ndvr = ndvr, nele = 2, spinbra = 'triplet', Szbra = 0.0, Cbra = C2T[:,q], spinket = 'singlet', Szket = 0.0, Cket = C2S[:,p]))
                #rho2T2T_pq = np.einsum('iaia->ia', fci_1rdm(ndvr = ndvr, nele = 2, spinbra = 'triplet', Szbra = 0.0, Cbra = C2T[:,p], spinket = 'triplet', Szket = 0.0, Cket = C2T[:,q]))
                #rho2T2T_qp = np.einsum('iaia->ia', fci_1rdm(ndvr = ndvr, nele = 2, spinbra = 'triplet', Szbra = 0.0, Cbra = C2T[:,q], spinket = 'triplet', Szket = 0.0, Cket = C2T[:,p]))

                #gam2T2T_pq = fci_1rdm(ndvr = ndvr, nele = 2, spinbra = 'triplet', Szbra = 0.0, Cbra = C2T[:,p], spinket = 'triplet', Szket = 0.0, Cket = C2T[:,q])
                #gam2T2T_qp = fci_1rdm(ndvr = ndvr, nele = 2, spinbra = 'triplet', Szbra = 0.0, Cbra = C2T[:,q], spinket = 'triplet', Szket = 0.0, Cket = C2T[:,p])
                #print(np.max(np.abs(gam2T2T_pq - np.einsum('iajb->jbia', gam2T2T_qp.conj()))))
                #wfn_p = fci_wfn(ndvr, 2, 'triplet', 0.0, C2T[:,p]); wfn_q = fci_wfn(ndvr, 2, 'triplet', 0.0, C2T[:,q])
                #gam_pq = 2*np.sum(wfn_p.conj()[None,None,:,:,:,:]*wfn_q[:,:,None,None,:,:], axis=(4,5))
                #gam_qp = 2*np.sum(wfn_q.conj()[None,None,:,:,:,:]*wfn_p[:,:,None,None,:,:], axis=(4,5))
                #print(np.max(np.abs(gam2T2T_pq - gam_pq)))
                #gam2S2T_pq = fci_1rdm(ndvr = ndvr, nele = 2, spinbra = 'singlet', Szbra = 0.0, Cbra = C2S[:,p], spinket = 'triplet', Szket = 0.0, Cket = C2T[:,q])
                #gam2S2T_qp = fci_1rdm(ndvr = ndvr, nele = 2, spinbra = 'triplet', Szbra = 0.0, Cbra = C2T[:,q], spinket = 'singlet', Szket = 0.0, Cket = C2S[:,p])
                #print(np.max(np.abs(gam2S2T_pq - np.einsum('iajb->jbia', gam2S2T_qp.conj()))))
                #wfn_p = fci_wfn(ndvr, 2, 'singlet', 0.0, C2S[:,p]); wfn_q = fci_wfn(ndvr, 2, 'triplet', 0.0, C2T[:,q])
                #gam_pq = 2*np.sum(wfn_p.conj()[None,None,:,:,:,:]*wfn_q[:,:,None,None,:,:], axis=(4,5))
                #gam_qp = 2*np.sum(wfn_q.conj()[None,None,:,:,:,:]*wfn_p[:,:,None,None,:,:], axis=(4,5))
                #print(np.max(np.abs(gam_pq - np.einsum('iajb->jbia', gam_qp.conj()))))
                #print(np.max(np.abs(gam2S2T_pq - gam_pq)))

                #print(np.max(np.abs(gam_pq - np.einsum('iajb->jbia', gam_qp.conj()))))
                #rho_pq = 2*np.sum(wfn_p.conj()*wfn_q, axis=(2,3)); rho_qp = 2*np.sum(wfn_q.conj()*wfn_p, axis=(2,3))
                #rho_pq = 2*np.sum(wfn_p.conj()*wfn_q, axis=(2,3)); rho_qp = 2*np.sum(wfn_p*wfn_q.conj(), axis=(2,3))
                #print(rho_pq-rho_qp.conj())
                #print(np.sum(rho_pq).real, p, q)
                #print(np.max(np.abs(rho_pq - rho_qp.conj())), p, q)
                #print(np.max(np.abs(rho2T2T_pq - rho2T2T_qp.conj())), p, q)


                #print(rho2D_pq)
                #print(np.argmax(np.abs(rho2D_pq - rho2D_qp.conj())), p, q)
                #print(np.sum(rho2D_pq))
                #print(np.sum(rho2D_qp))


                #D2Taa3Da_pn  = fci_dyson(ndvr = ndvr, nele = 3, spinbra = 'triplet', Szbra = 1.0, Cbra = C2T[:,p], spinket = 'doublet', Szket = 0.5, Cket = C3D[:,n])
                #D2Tab3Da_qm  = fci_dyson(ndvr = ndvr, nele = 3, spinbra = 'triplet', Szbra = 0.0, Cbra = C2T[:,q], spinket = 'doublet', Szket = 0.5, Cket = C3D[:,m])
                #print(np.max(np.abs(D2Taa3Da_pn*D2Tab3Da_qm.conj())))




exit()

plt.rcParams.update({
    'figure.figsize': (6.0, 4.0),
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'lines.linewidth': 2,
    'axes.linewidth': 3,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.direction': 'in',
    'xtick.top': True,
    'ytick.direction': 'in',
    'ytick.right': True,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'legend.frameon': False,
})

p = 0
n = 0
N = 0
xi = fcidvr.xi()
Dia12 = fci_dyson(ndvr, 1, 'doublet', -0.5, fcidvr_data1D['Cn'][:,p], 2, 'singlet', 0.0, fcidvr_data2S['Cn'][:,n])
Dia23 = fci_dyson(ndvr, 2, 'singlet', 0.0, fcidvr_data2S['Cn'][:,n], 3, 'doublet', -0.5, fcidvr_data3D['Cn'][:,N])
gam2 = fci_1rdm(ndvr, 2, 'singlet', 0.0, fcidvr_data2S['Cn'][:,n])
gam3 = fci_1rdm(ndvr, 3, 'doublet', -0.5, fcidvr_data3D['Cn'][:,N])
rho2 = np.einsum('iaia->ia', gam2)
rho3 = np.einsum('iaia->ia', gam3)
#print(Dia23)
#print(Dia23.shape)
#print(rho3.shape)


fig, (ax1) = plt.subplots(1, 1, figsize=(6, 4))
ax1.set_xlabel('D_pn(x)')
ax1.set_ylabel('x')
ax1.set_title(f"p = {p}; n = {n}")
#ax1.set_xlim([-35.0, 35.0])
#ax1.set_ylim([-0.01, 0.2])
ax1.plot(xi, np.abs(Dia23[:,0]), color='r')
ax1.plot(xi, np.abs(Dia23[:,1]), color='b')
#ax1.plot(xi, np.angle(Di), color='g')
ax1.plot(xi, rho2, color='g')
ax1.plot(xi, rho3, color='k', linestyle='--')
plt.show()


exit()

C3test = np.zeros((nfci_3ele), dtype=np.complex128)
#C3test[0] = 1.0
C3test[-1] = 1.0
wfn3test = fci_wfn(ndvr, 3, 'doublet', -0.5, C3test)
print(np.sum(wfn3test.conj()*wfn3test))

#print(wfn3test[0,0,1,0,1,1])
#print(wfn3test[0,0,1,1,1,0])
#
#print(wfn3test[1,0,0,0,1,1])
#print(wfn3test[1,1,0,0,1,0])
#
#print(wfn3test[1,1,1,0,0,0])
#print(wfn3test[1,0,1,1,0,0])
#
#print(wfn3test[1,0,1,1,0,0])
#print(wfn3test[1,1,1,0,0,0])

#print(wfn3test[-2,0,-2,0,-1,1])
#print(wfn3test[-2,0,-2,1,-1,0])
#print(wfn3test[-2,1,-2,0,-1,0])
#
#print(wfn3test[-2,0,-1,1,-2,0])
#print(wfn3test[-2,0,-1,0,-2,1])
#print(wfn3test[-2,1,-1,0,-2,0])
#
#print(wfn3test[-2,0,-2,0,-1,1])
#print(wfn3test[-2,1,-2,0,-1,0])
#print(wfn3test[-2,0,-2,1,-1,0])
#
#print(wfn3test[-1,1,-2,0,-2,0])
#print(wfn3test[-1,0,-2,1,-2,0])
#print(wfn3test[-1,0,-2,0,-2,1])
#
#print(wfn3test[-2,0,-1,1,-2,0])
#print(wfn3test[-2,1,-1,0,-2,0])
#print(wfn3test[-2,0,-1,0,-2,1])

#print(np.argwhere(np.abs(wfn3test) != 0))
#
#print(wfn3test[-2,0,-2,1,-1,1])
#print(wfn3test[-2,1,-2,0,-1,1])
#print(wfn3test[-2,1,-2,1,-1,0])
#
#print(wfn3test[-2,0,-1,1,-2,1])
#print(wfn3test[-2,1,-1,1,-2,0])
#print(wfn3test[-2,1,-1,0,-2,1])
#
#print(wfn3test[-1,1,-2,0,-2,1])
#print(wfn3test[-1,1,-2,1,-2,0])
#print(wfn3test[-1,0,-2,1,-2,1])

C2test = np.random.random(nfci_2ele) + 1j*np.random.random(nfci_2ele)
C2test *= 1/np.sqrt(np.sum(C2test.conj()*C2test))

C3test = np.random.random(nfci_3ele) + 1j*np.random.random(nfci_3ele)
C3test *= 1/np.sqrt(np.sum(C3test.conj()*C3test))

import cProfile

profiler = cProfile.Profile()
profiler.enable()

for _ in range(1000):
    #H2S.matvec(C2test)
    H3D.matvec(C3test)

profiler.disable()
profiler.print_stats(sort="cumtime")



exit()

from line_profiler import LineProfiler

lp = LineProfiler()
lp.add_function(H3D.matvec)

lp.enable()

for _ in range(100):
    H3D.matvec(C3test)

lp.disable()
lp.print_stats()

