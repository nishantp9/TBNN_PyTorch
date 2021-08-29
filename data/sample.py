import numpy as np
import os
if os.path.basename(os.getcwd()) == "data":
    os.chdir(os.path.join(".."))

def fn(x):
    n_dim = x.shape[-1]
    y = np.matmul(x, np.matmul((x+x.T), (x-x.T)))
    y = np.matmul((y-y.T), x)

    # symmeteric & traceless output
    y = 0.5*(y+y.T)
    y -= np.eye(n_dim) * y.trace()/n_dim
    return y

def get_rand_tensors(N, n_dim):
    A = np.random.randn(N,n_dim,n_dim).astype('float32')
    # traceless input
    for i in range(N):
        A[i] -= np.eye(n_dim) * A[i].trace()/n_dim

    B = []
    for X in A:
        B.append(fn(X))
    B = np.stack(B, axis=0)

    return A, B


if __name__ == '__main__':

    A_train, B_train = get_rand_tensors(N=10000, n_dim=3)

    os.makedirs('data/train', exist_ok=True)
    np.save('data/train/traceless_input', A_train)
    np.save('data/train/traceless_sym_output', B_train)

    A_test, B_test = get_rand_tensors(N=1000, n_dim=3)

    os.makedirs('data/test', exist_ok=True)
    np.save('data/test/traceless_input', A_test)
    np.save('data/test/traceless_sym_output', B_test)