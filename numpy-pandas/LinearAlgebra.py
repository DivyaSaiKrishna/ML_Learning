import numpy as np

#matrix multiply
a = np.array([1,2,3])
b = np.array([4,5,6])
ab = np.matmul(a,b)
a_b = np.dot(a,b)
print(ab)
print(a_b)
print(a * b)

#Linear System
A = np.arange(3,20,2)
b = np.arange(1,10,2)

Aa = np.array([[2, 1, 3],
              [1, 3, 2],
              [3, 1, 1]]) 
bB = np.array([9, 8, 6])

x = np.linalg.solve(Aa, bB)
aT = np.linalg.det(Aa)
aI = np.linalg.inv(Aa)

print(A)
print(b)
print(x)
print(aT)
print(aI)

scores = np.array([[85, 90, 78],
                   [70, 65, 80],
                   [90, 95, 88],
                   [60, 55, 70]])

# Covariance matrix — how features vary together
cov = np.cov(scores.T)         # shape (3, 3)
print("Covariance:\n", cov)

# Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov)
print("Eigenvalues:", eigenvalues)   # magnitude of variance in each direction
print("Eigenvectors:\n", eigenvectors)  # directions