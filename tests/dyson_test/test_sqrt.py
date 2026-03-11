import numpy as np

a = 0.4
b = 1.5

z = (a+1j*b)
print(z)
print(z**2)

print(a**2-b**2)
print(2*a*b)

#ztest = (np.abs(a)+1j*a*b/np.abs(a))
ztest = np.sign(a)*(a+1j*b)

print(np.sqrt(z**2))
print(ztest)
