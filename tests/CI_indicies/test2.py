import numpy as np

filename = "fcidvr_3ele_doublet.npz"

#base = filename.removesuffix(".npz")   # Python ≥3.9
parts = filename.split("_")
nele = int(parts[1][0])
spin = parts[2][:-4]
print(parts)
print(nele)
print(spin)
exit()


ndvr = 7
nbo = 3
a = np.random.random((nbo, nbo, ndvr))
b = np.random.random((nbo, ndvr)) 
A = a.transpose(2, 0, 1)
B = b.T

print(A.shape) # (ndvr, nbo, nbo)
print(B.shape) # (ndvr, nbo)

#C = (A @ B[:,:,None]).squeeze(-1).T
C = (A @ b.T[:,:,None]).squeeze(-1).T
D = np.einsum('nmJ,mJ->nJ', a, b)

print(C)
print(D)
print(C-D)

exit()
nmap1 = np.triu((1 - (1 - 1 / np.sqrt(2)) * np.eye(ndvr)), k = 0)
nmap2 = np.triu(np.ones((ndvr, ndvr)), k = 1)
print(nmap1)
print(nmap2)
