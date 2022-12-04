# import math functions
import math

import numpy as np

from numpy.random import randn

from numpy import dot
from scipy.linalg import inv

np.random.seed(42)


def select_y_R(x, vel, acc, sigma_x, sigma_yv, sigma_ya, meas):
    """
    Creates and returns the measurement y and measurement noise matrix R,
    depending on given meas.
    Args:
        x (float): current position
        vel (): current velocity
        acc (): current acceleration
        sigma_x (): measurement deviation for position
        sigma_yv (): measurement deviation for velocity
        sigma_ya (): measurement deviation for acceleration
        meas (string): Defines which state variables are measured 'p_v', 'p', ...

    Returns:

    """
    if meas == 'p_v':
        y = np.array([[x + randn() * sigma_x, vel + randn() * sigma_yv]]).T
        R = np.diag([sigma_x ** 2, sigma_yv ** 2])
    elif meas == 'p':
        y = np.array([[x + randn() * sigma_x]]).T
        R = np.diag([sigma_x ** 2])
    elif meas == 'a':
        y = np.array([[acc + randn() * sigma_ya]]).T
        R = np.diag([sigma_ya ** 2])
    elif meas == 'p_a':
        y = np.array([[x + randn() * sigma_x, acc + randn() * sigma_ya]]).T
        R = np.diag([sigma_x ** 2, sigma_ya ** 2])
    elif meas == 'v':
        y = np.array([[vel + randn() * sigma_yv]]).T
        R = np.diag([sigma_yv ** 2])
    else: # 'p_v_a'
        y = np.array([[x + randn() * sigma_x, vel + randn() * sigma_yv,
                       acc + randn() * sigma_ya]]).T
        R = np.diag([sigma_x ** 2, sigma_yv ** 2, sigma_ya ** 2])

    return y, R


def select_C(meas):
    """
    Creates and returns measurement matrix C depending on meas
    Args:
        meas (string): Defines which state variables are measured 'p_v', 'p', ...

    Returns:

    """
    if meas == 'p_v':
        C = np.array([[1., 0., 0.],
                      [0., 1., 0.]])
    elif meas == 'p':
        C = np.array([[1., 0., 0.]])
    elif meas == 'a':
        C = np.array([[0., 0., 1.]])
    elif meas == 'p_a':
        C = np.array([[1., 0., 0.],
                      [0., 0., 1.]])
    elif meas == 'v':
        C = np.array([[0., 1., 0.]])
    else:  # 'p_v_a'
        C = np.array([[1., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.]])

    return C


def generate_acc(num_y):
    """
    Creates a vector of accelerations of size num_y

    Args:
        num_y (int): Number of estimations the Kalman filter should create

    Returns:
        acc: the vector of accelerations
    """
    acc = np.zeros((num_y, 1)).ravel()
    acc[:5] = 0.5
    acc[-20:-15] = -0.5

    acc[25:27] = 0.25
    acc[65:67] = -0.25

    return acc


def compute_y(sigma_yp, process_var, num_y=1, dt=1., meas='p'):
    """
    returns track, measurements 1D ndarrays

    Args:
        sigma_yp (float): stamdard deviation of position measurement
        process_var (float): variance of process noise of acceleration
        num_y (int): Number of estimations the Kalman filter should create
        dt (): delta t
        meas (string): Defines which state variables are measured 'p_v', 'p', ...

    Returns:

    """
    x, vel = 0.5, 0.  # initial position and velocity

    # generate vector of accelerations acc
    acc = generate_acc(num_y)

    # vector of sigma's of position measurement device
    sigma_yps = np.ones((num_y, 1)).ravel() * sigma_yp

    sigma_yv = 0.95#55  # sigma for velocity measurement
    sigma_ya = 0.15#5   # sigma for acceleration measurement

    p_std = math.sqrt(process_var)
    xs, ys, Rs = [], [], []

    for myacc, mysigma_y in zip(acc, sigma_yps):
        a = myacc + (randn() * p_std)
        x += vel*dt + a*1/2.0*dt**2
        vel += a * dt

        xs.append(x)

        y, R = select_y_R(x, vel, a, mysigma_y, sigma_yv, sigma_ya, meas)

        ys.append(y)
        Rs.append(R)

    return np.array(xs), np.array(ys), np.array(Rs)


def init_system(num_y=50, meas='p'):
    """
    Initializes system parameters and parameters of the Kalman filter

    Args:
        num_y (int): Number of estimations the Kalman filter should create
        meas (string): String that defines which measurements are available for the Kalman filter

    Returns:
        b_true, y,
        u: values of input vector u for all instances num_y
        x: initial state estimate
        P: initial covariance matrix
        A: system matrix
        C: output matrix
        Rs:
        Q: process noise covariance matrix
        dt: sampling time
    """
    # initial parameters
    #mu_y = 1.0
    sigma_yp = 0.5       # measurement deviation

    mu_b_post = 1.0     # initial estimate of position
    sigma_b_post = 2.0  # deviation of initial position estimate

    var_b_post = sigma_b_post ** 2  # initial variance of position estimate

    var_a = 0.0001  # process noise for acceleration

    dt = 0.2#0.75 sampling time

    # measurements for mu and motions, U
    b_true, y, Rs = compute_y(sigma_yp, var_a, num_y, dt, meas)

    # definition of input
    u = np.zeros((num_y, 1)).ravel()

    # initial estimate of state vector, covariance amtrix
    x = np.array([[mu_b_post, 0.5, 0.3]]).T
    P = np.diag([var_b_post, 2, 1])

    # definition of system matrix
    A = np.array([[1, dt, 1/2.*dt**2],
                  [0, 1, dt],
                  [0, 0, 1]])

    # definition of output matrix
    C = select_C(meas)

    # definition of process noise covariance matrix
    Q = np.diag([var_a*1/2.*dt**2, var_a*dt, var_a])

    return b_true, y, u, x, P, A, C, Rs, Q, dt


def calc_estim_err(GT, estim, avg=True):
    """
    Calculates estimation error

    Args:
        GT (): groundtruth of position
        estim (): estimated position
        avg (): if True, calculates RSME, else the elementwise root of squared error

    Returns:
        estimation error
    """
    GT = GT.ravel()
    estim = estim.ravel()

    if avg:
        err = np.sqrt(np.mean((GT-estim)**2))
    else:
        err = np.sqrt((GT - estim) ** 2)

    return err


def run_kalman(num_y=30, meas='p_v'):
    """
    Main method to call: Runs the Kalman filter algorithm for a simple localization problem.

    Args:
        num_y (int): Number of estimations the Kalman filter should create
        meas (string): String that defines which measurements are available for the Kalman filter

    Returns:
        xs, b_true, cov, y, Rs, num_y
    """
    # initialize the system
    b_true, y, u, x, P, A, C, Rs, Q, dt = init_system(num_y=num_y, meas=meas)

    xs, cov = [], []
    for myy, R in zip(y, Rs):
        # predict
        P = dot(A, P).dot(A.T) + Q
        x = dot(A, x)

        # update
        S = dot(C, P).dot(C.T) + R
        K = dot(P, C.T).dot(inv(S))
        z = myy - dot(C, x)
        x += dot(K, z)
        P = P - dot(K, C).dot(P)

        xs.append(x)
        cov.append(P)

    xs, cov = np.array(xs), np.array(cov)

    print('estimation error:', calc_estim_err(b_true, xs[:, 0]))

    print('b_estim:', xs[:, 0])
    print('v_estim:', xs[:, 1])
    print('a_estim:', xs[:, 2])
    print('var_b:', cov[:, 0, 0])
    print('var_v:', cov[:, 1, 1])
    print('var_a:', cov[:, 2, 2])

    return xs, b_true, cov, y, Rs, num_y


if __name__ == '__main__':
    run_kalman(num_y=80, meas='p_a')

