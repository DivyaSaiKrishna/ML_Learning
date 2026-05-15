import numpy as np

#array
npArr = np.array([1,2,3])
npZero = np.zeros((3,4))
npOne = np.ones((2,3))
nplin = np.linspace(0,1,100)

print(npArr)
print(npZero)
print(npOne)
print(nplin)

#Indexing & Slicing
#first row
print(npZero[0, :])
#third
print(npZero[:, 2])

npLinCopy = nplin.copy()
print(npLinCopy)
npLinFlat = nplin.flatten()
print(npLinFlat)

#Math & Stats Operations

print(npLinFlat.sum())


