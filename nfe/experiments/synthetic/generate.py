import numpy as np
import scipy.signal
import scipy.integrate
from pathlib import Path

import torch.cuda
from scipy.sparse import random
from tqdm import tqdm




np.random.seed(123)

cuda = torch.cuda.is_available()

NUM_SEQUENCES = 1000 if cuda else 30
NUM_POINTS = 1000 if cuda else 500
MAX_TIME = 10
EXTRAPOLATION_TIME = 20

def get_inital_value(extrap_space):
    if extrap_space:
        return np.random.uniform(-4, -2, (1,)) if np.random.rand() > 0.5 else np.random.uniform(2, 4, (1,))
    else:
        return np.random.uniform(-2, 2, (1,))

def get_n_inital_values(n, extrap_space):
    if n == 1:
        return get_inital_value(extrap_space)
    else:
        res = np.stack(list(map(lambda x: get_inital_value(extrap_space), range(n)))).squeeze()
        return res

def get_inital_value2d(extrap_space):
    if extrap_space:
        return np.random.uniform(1, 2, (2,))
    else:
        return np.random.uniform(0, 1, (2,))

def get_data(data_dir, func, time_min, time_max, extrap_space=False, name=None):
    initial_values = []
    times = []
    sequences = []

    for _ in tqdm(range(NUM_SEQUENCES), desc=f"{name}"):
        t = np.sort(np.random.uniform(time_min, time_max, NUM_POINTS))
        y0, y = func(t, extrap_space)
        times.append(t)
        initial_values.append(y0)
        sequences.append(y)

    initial_values, times, sequences = np.array(initial_values), np.array(times), np.array(sequences)
    if name is None:
        return initial_values, times, sequences
    else:
        np.savez(data_dir / f'{name}.npz', init=initial_values, seq=sequences, time=times)

def make_A(n, density=0.2, seed=1):
    """
    make a DAG nxn
    density: density of non-zero elements
    n: num of rows and cols
    """
    # init as sparse matrix ER graph, see notears
    np.random.seed(seed=seed)
    A = random(n, n, density=density, format='csr')
    A = np.triu(A.todense())  # convert to upper-triangular matrix
    np.fill_diagonal(A, 0)
    return A

def generate(args, DATA_DIR):
    n = args['n_ts']
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args['dag_data'] == 0:
        A = None
    else:
        const_ = 3.0
        _density = const_ / n
        if not DATA_DIR.exists() or not (DATA_DIR / 'dag.pt').exists():
            if n in [3]:
                _density = 0.5
            elif n in [5, 10]:
                _density = 0.35
            elif n >= 15:
                _density = 0.13
            elif n >= 25:
                _density = 0.12
            else:
                raise ValueError('invalid n_ts')
            A = make_A(n=n, density=_density, seed=1)
            torch.save(A, f'{DATA_DIR}/dag.pt')
            np.savetxt(f'{DATA_DIR}/dag.csv', A, delimiter=',')


    # SINE
    def sine_func(t, extrap_space=False):
        y = get_inital_value(extrap_space)
        return y, np.sin(t[:,None]) + y
    if not (DATA_DIR / 'sine.npz').exists():
        get_data(DATA_DIR, sine_func, 0, MAX_TIME, name='sine')
        # get_data(DATA_DIR, sine_func, MAX_TIME, MAX_TIME + EXTRAPOLATION_TIME, name='sine_extrap_time')
        # get_data(DATA_DIR, sine_func, 0, MAX_TIME, extrap_space=True, name='sine_extrap_space')

    # SQUARE
    def square_func(t, extrap_space=False):
        y = get_n_inital_values(n, extrap_space)
        if A is not None:
            sol = (np.sign(np.sin(t[:, None])) + y) @ (np.eye(n) - A.T)
        else:
            sol = np.sign(np.sin(t[:, None])) + y
        return y, sol
    if not (DATA_DIR / 'square.npz').exists():
        get_data(DATA_DIR, square_func, 0, MAX_TIME, name='square')
        # get_data(DATA_DIR, square_func, MAX_TIME, MAX_TIME + EXTRAPOLATION_TIME, name='square_extrap_time')
        # get_data(DATA_DIR, square_func, 0, MAX_TIME, extrap_space=True, name='square_extrap_space')

    # SAWTOOTH
    def sawtooth_func(t, extrap_space=False):
        y = get_n_inital_values(n, extrap_space)
        if A is not None:
            sol = (scipy.signal.sawtooth(t[:None, None]) + y) @ (np.eye(n) - A.T)
        else:
            sol = scipy.signal.sawtooth(t[:, None]) + y
        return y, sol
    if not (DATA_DIR / 'sawtooth.npz').exists():
        get_data(DATA_DIR, sawtooth_func, 0, MAX_TIME, name='sawtooth')
        # get_data(DATA_DIR, sawtooth_func, MAX_TIME, MAX_TIME + EXTRAPOLATION_TIME, name='sawtooth_extrap_time')
        # get_data(DATA_DIR, sawtooth_func, 0, MAX_TIME, extrap_space=True, name='sawtooth_extrap_space')

    # TRIANGLE
    def triangle_func(t, extrap_space=False):
        y = get_n_inital_values(n, extrap_space)
        if A is not None:
            sol = (np.abs(scipy.signal.sawtooth(t[:None, None])) + y) @ (np.eye(n) - A.T)
        else:
            sol = np.abs(scipy.signal.sawtooth(t[:, None])) + y
        return y, sol
    if not (DATA_DIR / 'triangle.npz').exists():
        get_data(DATA_DIR, triangle_func, 0, MAX_TIME, name='triangle')
        # get_data(DATA_DIR, triangle_func, MAX_TIME, MAX_TIME + EXTRAPOLATION_TIME, name='triangle_extrap_time')
        # get_data(DATA_DIR, triangle_func, 0, MAX_TIME, extrap_space=True, name='triangle_extrap_space')

    B = np.array([[-4, 5], [-3, 1]])

    def sink_func(t, extrap_space=False):
        """sink conditioned on known adjacency matrix"""
        iv = [get_inital_value2d(extrap_space) for _ in range(n)]
        y = np.stack(iv).squeeze().reshape(-1, n * 2).squeeze()
        if A is not None:
            sol = scipy.integrate.odeint(lambda y, t: np.kron((np.eye(n) - A.T), B) @ y, y, t)
        else:
            sol = scipy.integrate.odeint(lambda y, t:  np.kron((np.eye(n) - np.zeros(n)), B) @ y, y, t)
        return y, sol
    if not (DATA_DIR / 'sink.npz').exists():
        get_data(DATA_DIR, sink_func, 0, MAX_TIME, name='sink')
        # get_data(DATA_DIR, sink_func, MAX_TIME, MAX_TIME + EXTRAPOLATION_TIME, name='sink_extrap_time')
        # get_data(DATA_DIR, sink_func, 0, MAX_TIME, extrap_space=True, name='sink_extrap_space')

    def mat2vec(data):
        return np.reshape(data, data.shape[:-2] + (-1,))

    def vec2mat(data, shape):
        return np.reshape(data, data.shape[:-1] + shape)

    def f(x, t, shape):
        X = vec2mat(x, shape)
        F = np.column_stack((2 / 3 * X[:, 0] - 2 / 3 * X[:, 0] * X[:, 1], X[:, 0] * X[:, 1] - X[:, 1]))
        f = mat2vec((np.eye(shape[0]) - A.T) @ F)
        return f

    def f2(x, t, shape):
        """ellipse with no graph"""
        X = vec2mat(x, shape)
        F = np.column_stack((2 / 3 * X[:, 0] - 2 / 3 * X[:, 0] * X[:, 1], X[:, 0] * X[:, 1] - X[:, 1]))
        f = mat2vec(F)
        return f

    # ELLIPSE (Lotka-Volterra)
    def ellipse_func(t, extrap_space=False):
        iv = [get_inital_value2d(extrap_space) for _ in range(n)]
        y = np.array(iv)
        y0 = mat2vec(y)
        shape = y.shape
        if A is not None:
            sol = scipy.integrate.odeint(f, y0, t, args=(shape,)) - 1
        else:
            sol = scipy.integrate.odeint(f2, y0, t, args=(shape,)) - 1
        return y0, sol


if __name__ == '__main__':
    generate()
