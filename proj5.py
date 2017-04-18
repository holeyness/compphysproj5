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
from matplotlib import animation


# Set up our constants
dx = 40.0 / 1000
alpha = 1 / (2 * (dx ** 2))
dt = 80.0 / 1000
x_domain = (-40, 40)

x0 = 0
k0 = 8
sigma = 4


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
    H1 = 1j * dt * np.array(hamiltonian)
    for (x, y), val in np.ndenumerate(H1):
        if val != 0:
            H1[x, y] += 1
    H1 = np.matrix(H1)

    print(H1)

    # Take the inverse
    ham_inverse = inv(H1)

    # print('Hamiltonian', hamiltonian)
    print('Hamiltonian Inverse', ham_inverse)

    # Prepare a wavepacket, show the graph
    x = np.arange(x_domain[0], x_domain[1], dx)
    y = [wavepacket(i, x0, k0, sigma) for i in x]

    # Dot multiplication of the hamiltonian matrix, for 100 iterations
    psi_result = []
    psi_multiplier = np.matrix(y).transpose()

    for _ in range(100):
        psi_iteration = np.dot(ham_inverse, psi_multiplier)
        psi_result.append(psi_iteration)
        psi_multiplier = psi_iteration
        
    # Animation

    fig = plt.figure()
    ax = plt.axes(xlim=(-5, 5), ylim=(-1, 1))
    real_line, = ax.plot([], [], 'b', lw=1)
    im_line, = ax.plot([],[], 'g', lw=1)
    prob_line, = ax.plot([],[], 'r', lw=1)
    
    def init():
        real_line.set_data([], [])
        im_line.set_data([],[])
        prob_line.set_data([],[])
        return real_line, im_line, prob_line

    # Animation function
    def animate(i, *fargs):
        x_list = fargs[0]
        psi = fargs[1]
        real_line.set_data(x_list, np.real(psi[i]))
        im_line.set_data(x_list, np.imag(psi[i]))
        prob_line.set_data(x_list, np.absolute(psi[i]))
        return real_line, im_line, prob_line
    
    # Call the animator
    anim = animation.FuncAnimation(fig, animate, init_func=init, fargs=[x, psi_result], interval=500,
                                   frames=10)
   
    #Save the animation                               
    #anim.save('animation.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
    
    #Show the animation
    plt.show()

    """THE NON-NAIVE Method
    cuz let's be real, we are all adults here"""
    rhs_multiplier = 1j * (0.5) * dt * np.array(hamiltonian)
    rhs_multiplier = np.matrix(rhs_multiplier)
    for (x_index, y_index), val in np.ndenumerate(rhs_multiplier):
        if val != 0.0:
            rhs_multiplier[x_index, y_index] = 1 - val
    rhs_multiplier = np.matrix(rhs_multiplier)
    rhs = np.dot(rhs_multiplier, np.matrix(y).transpose())

    lhs_multiplier = (0.5) * (1j) * dt * np.array(hamiltonian)
    lhs_multiplier = np.matrix(lhs_multiplier)
    for (x_index, y_index), val in np.ndenumerate(lhs_multiplier):
        if val != 0.0:
            lhs_multiplier[x_index, y_index] = 1 + val
    lhs_multiplier = np.matrix(lhs_multiplier)

    print(rhs_multiplier)
    print(lhs_multiplier)

    results = []
    for _ in range(100):
        next_step = solve(lhs_multiplier, rhs)
        results.append(next_step)
        rhs = np.dot(rhs_multiplier, next_step)

    fig = plt.figure()
    ax = plt.axes(xlim=(-20, 20), ylim=(-3, 3))
    real_line, = ax.plot([], [], 'b', lw=1)
    im_line, = ax.plot([],[], 'g', lw=1)
    prob_line, = ax.plot([],[], 'r', lw=1)

    anim = animation.FuncAnimation(fig, animate, init_func=init, fargs=[x, results], interval=100, frames=50)

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
