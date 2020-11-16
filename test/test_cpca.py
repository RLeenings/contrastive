import numpy as np


if __name__ == '__main__':
    N = 401; D = 1001; gap=3
    # In B, all the data pts are from the same distribution, which has different variances in three subspaces.
    B = np.zeros((N, D))
    B[:,0:10] = np.random.normal(0,10,(N,10))
    B[:,10:20] = np.random.normal(0,3,(N,10))
    B[:,20:30] = np.random.normal(0,1,(N,10))


    # In A there are four clusters.
    A = np.zeros((N, D))
    A[:,0:10] = np.random.normal(0,10,(N,10))
    # group 1
    A[0:100, 10:20] = np.random.normal(0,1,(100,10))
    A[0:100, 20:30] = np.random.normal(0,1,(100,10))
    # group 2
    A[100:200, 10:20] = np.random.normal(0,1,(100,10))
    A[100:200, 20:30] = np.random.normal(gap,1,(100,10))
    # group 3
    A[200:300, 10:20] = np.random.normal(2*gap,1,(100,10))
    A[200:300, 20:30] = np.random.normal(0,1,(100,10))
    # group 4
    A[300:400, 10:20] = np.random.normal(2*gap,1,(100,10))
    A[300:400, 20:30] = np.random.normal(gap,1,(100,10))
    A_labels = [0]*100+[1]*100+[2]*100+[3]*100

    cpca = CPCA(standardize=False)
    cpca.fit_transform(A, B, plot=True, active_labels=A_labels)

    print(A.shape)
    print(B.shape)