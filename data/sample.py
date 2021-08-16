import numpy as np
import os
if os.path.basename(os.getcwd()) == "data":
    os.chdir(os.path.join(".."))

def fn(x):
    y = np.matmul(x, np.matmul((x+x.T), (x-x.T)))
    y = np.matmul((y-y.T), x)

    # symmeteric & traceless output
    y = 0.5*(y+y.T)
    y -= np.eye(3) * y.trace()/3.
    return y

def get_rand_tensors(N):
    A = np.random.randn(N,3,3).astype('float32')
    # traceless input
    for i in range(N):
        A[i] -= np.eye(3) * A[i].trace()/3.

    B = []
    for X in A:
        B.append(fn(X))
    B = np.stack(B, axis=0)

    return A, B


if __name__ == '__main__':

    A_train, B_train = get_rand_tensors(N=10000)

    os.makedirs('data/train', exist_ok=True)
    np.save('data/train/traceless_input', A_train)
    np.save('data/train/traceless_sym_output', B_train)

    A_test, B_test = get_rand_tensors(N=1000)

    os.makedirs('data/test', exist_ok=True)
    np.save('data/test/traceless_input', A_test)
    np.save('data/test/traceless_sym_output', B_test)