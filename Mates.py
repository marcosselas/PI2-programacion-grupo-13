import numpy as np
def my_gauss(A, B):
    N = len(B) # 3
    x = np.zeros(N)
    for i in range(0, N): #0, 1, 2
        for k in range(i + 1, N): # 1, 2 ; 2
            if A[i, i] != 0.0:
                lam = A[k, i] / A[i, i]
                A[k, : ] = A[k, :] - lam * A[i, :]
                B[k] = B[k] - lam * B[i]
    for i in range(N - 1, -1, -1): # 2, 1, 0 
        x[i] = (B[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    print(x)
    return x

# matriz_estatica = np.array ([
# [1.0 , 1.0 , 1.0] , # Suma de fuerzas Z
# [0.0 , 1.5 , -1.5] , # Momentos en X
# [ -2.0 , 1.0 , 1.0] # Momentos en Y
# ])
# vector_cargas = np.array ([25000.0 , 1500.0 , -800.0]) # [N, Nm , Nm]
# my_gauss(matriz_estatica, vector_cargas) #[8600. 8700. 7700.]
# print(np.linalg.solve(matriz_estatica, vector_cargas)) #[8600. 8700. 7700.]