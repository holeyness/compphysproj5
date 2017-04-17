"""
Tufts Comp Physics 2017
Project 5: Time Dependent Schrodinger Equation
By: Yueming Luo, Michael Rosen, Chase Conley

Preface: ...?

    
"""
import scipy as sp
import math
from math import pi, e, sqrt
import numpy as np
from numpy.linalg import inv, solve
import matplotlib.pyplot as plt


# Set up our constants
dx = 80 / 1000
alpha = 1 / (2 * (dx ** 2))
dt = 80 / 1000
x_domain = (-40, 40)

x0 = 0
k0 = 2.5
sigma = 1


def main():
    """Our main execution function"""
    size = int(abs(x_domain[1] - x_domain[0]) / dx)
    hamiltonian = np.matrix(np.full([size, size], 0.0))    # Construct our matrix

    # Add diagnoal elements and fill in nearby elements
    for (x, y), _ in np.ndenumerate(hamiltonian):
        if x == y:
            # Set diagnoal elements
            hamiltonian[x, y] = 2 * alpha + v_free(y)
        elif x < size and y - x == 1:
            # The element to the right of each diag element
            hamiltonian[x, y] = -1 * alpha
        elif x > 0 and x - y == 1:
            # To the left of the diag
            hamiltonian[x, y] = -1 * alpha

    # Set top right and bottom left corners to negative alpha
    hamiltonian[-1, 0] = -1 * alpha
    hamiltonian[0, -1] = -1 * alpha


    # Lets turn this imaginary, and add 1
    hamiltonian = 1j * dt * np.array(hamiltonian)
    for (x, y), val in np.ndenumerate(hamiltonian):
        if val != 0:
            hamiltonian[x, y] += 1
    # hamiltonian = 1 + np.array(hamiltonian)
    hamiltonian = np.matrix(hamiltonian)
    print(hamiltonian)

    # Take the inverse
    ham_inverse = inv(hamiltonian)

    # print('Hamiltonian', hamiltonian)
    print('Hamiltonian Inverse', ham_inverse)

    # Prepare a wavepacket, show the graph
    x = np.arange(x_domain[0], x_domain[1], dx)
    y = [wavepacket(i, x0, k0, sigma) for i in x]
    plt.plot(x, y)
    plt.plot(x, np.real(y), color='b')
    plt.plot(x, np.imag(y), color='g')
    plt.xlabel('X')
    plt.ylim([-1.5, 1.55])
    plt.ylabel('Psi')
    plt.title('Wavepacket')
    plt.show()

    # Dot multiplication of the hamiltonian matrix, for 100 iterations
    psi_result = []
    psi_multiplier = np.matrix(y).transpose()


    for _ in range(100):
        print(ham_inverse.shape)
        print(psi_multiplier.shape)
        psi_iteration = np.dot(ham_inverse, psi_multiplier)
        psi_multiplier = psi_iteration

        # plt.plot(x, (np.absolute(psi_iteration) ** 2), color='r')
        plt.plot(x, np.real(psi_iteration), color='b')
        plt.plot(x, np.imag(psi_iteration), color='g')
        plt.xlim([-20, 20])
        plt.ylim([-1.5, 1.55])
        plt.show()


def wavepacket(x, x0, k0, sigma):
    return (e ** (0.25 * (x - x0) * (4j * k0 + (-1 * x + x0) * (sigma ** 2))) * (pi ** 0.5) * sigma) / (sqrt(sqrt(2) * (pi ** (3/2) * sigma)))


def v_free(x):
    """Empty potential function"""
    return 0


def v_step(x):
    if x < 0:
        return 0
    else:
        return 1

if __name__ == "__main__":
    main()
