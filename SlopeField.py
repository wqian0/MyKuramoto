from scipy.integrate import ode, odeint
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-np.pi, np.pi, 10000)
y = np.linspace(-2, 2, 10000)
X, Y = np.meshgrid(x, y)

def first_order_plot(fig_num, w, a, phi):
    plt.figure(fig_num)
    plt.rcParams.update({'font.size': 14})
    plt.title("first order plot")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\dot{\theta}$")
    plt.rcParams.update({'font.size': 14})
    # dx = Y
    # dy = -a*np.cos(phi-X)*(w + a*np.sin(phi - X))
    dx = w + a*np.sin(phi - X)
    dy = -a*np.cos(phi-X)*dx
    plt.streamplot(X, Y, dx, dy, color=dy, density=2, cmap='viridis', arrowsize=1, linewidth=.5)
    plt.colorbar()
    plt.tight_layout()

def second_order_plot(fig_num, m, w, a, phi):
    plt.figure(fig_num)
    plt.rcParams.update({'font.size': 14})
    plt.title("second order plot")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\dot{\theta}$")
    plt.rcParams.update({'font.size': 14})
    dx = Y
    dy = w/m -Y/m + (a/m)*np.sin(phi-X)
    plt.streamplot(X, Y, dx, dy, color=dy, density=2, cmap='viridis', arrowsize=1, linewidth=.5)
    plt.colorbar()
    plt.tight_layout()

first_order_plot(0, .5, 1, 0)
second_order_plot(1, 5, 0.5, 1, 0)
plt.show()