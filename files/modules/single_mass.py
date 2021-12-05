# %% Import

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import tensorflow as tf

# %% Solution routine


def solve_parametric(
    m: float,
    d: float,
    c: float,
    x_0: list = [0, 1],
    f_a: float = 1,
    f_f: float = 1,
    t_end: float = 24,
) -> tuple:
    """Solve parametrized single mass oscillator equation.

    The solution is computed in the phase space x = [displacement, velocity].

    Parameters
    ----------
    m : float
        Mass
    d : float
        Damping
    c : float
        Stiffness
    x_0 : list, optional
        Initial conditions, by default [0, 1]
    f_a: float, optional
        Amplitude of excitation force
    f_f: float, optional
        Frequency of excitation force
    t_end : float, optional
        End simulation time, by default 24

    Returns
    -------
    tuple
        Time points t_a and solution points x_a (in phase space)
    """
    def f(x, t):
        x_dot = [
            x[1],
            f_a*np.sin(f_f*t) - d/m*x[1] - c/m*x[0]
        ]
        return x_dot
    t_a = np.linspace(0, t_end, 101)
    x_a = odeint(f, x_0, t_a)
    return t_a, x_a


# %% Gather data


def design_of_experiments(
    d_range: list = [0, 2],
    c_range: list = [0.1, 1.5],
    f_f_range: list = [0.1, 1.5],
    n: int = 4
):
    p_a = np.array([
        [p_1, p_2, p_3]
        for p_1 in np.linspace(*d_range, n)
        for p_2 in np.linspace(*c_range, n)
        for p_3 in np.linspace(*f_f_range, n)
    ])
    return p_a


def compute_objective(p: list, plot=False, info=False):
    d, c, f_f = p
    t_a, x_a = solve_parametric(
        m=1,
        d=d,
        c=c,
        x_0=[0, 0],
        f_a=2,
        f_f=f_f,
        t_end=24
    )
    if plot:
        plot_sol(t_a, x_a, [0])
    i = np.argmax(np.abs(x_a[:, 0]))
    q = t_a[i], x_a[i, 0]
    if info:
        print(f"Time: {q[0]}")
        print(f"Displacement: {q[1]}")
    return q


def generate_objective_data(p_a: np.ndarray):
    return np.array([compute_objective(p) for p in p_a])


# %% ANN model

def generate_tf_model(
    in_d: int = 3,
    out_d: int = 2,
    n_hl: list = [4, 4],
    a_f: str = "relu",
    l_r: float = 0.01,
):
    layers = [tf.keras.layers.Input(shape=[in_d])]
    for n in n_hl:
        layers += [tf.keras.layers.Dense(units=n, activation=a_f)]
    layers += [tf.keras.layers.Dense(units=out_d)]
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=l_r),
        loss="mse"
    )
    return model

# %% Utilities


def i_closest(node, nodes):
    distances = np.sum((nodes - node)**2, axis=1)
    return np.argmin(distances)


# %% Plots


def plot_sol(t_a: list, x_a: np.ndarray, plot_i: list = None) -> None:
    """Plot solution.

    Parameters
    ----------
    t_a : list
        Time points
    x_a : np.ndarray
        Solution points (in phase space)
    plot_i : list, optional
        Index set for solution components to plot, by default None (plots
        all components). Use [0] to plot only the first component. 
    """
    if plot_i == None:
        plot_i = list(range(x_a.shape[1]))
    plt.figure()
    for i in plot_i:
        plt.plot(t_a, x_a[:, i], label=f"x_{i+1}")
    plt.legend()
    plt.xlabel("$t$")
    plt.ylabel("$x(t)$")
    plt.show()


def plot_train_history(h: tf.keras.callbacks.History):
    plt.figure()
    plt.semilogy(h.history["loss"])
    plt.grid(which="both")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
