import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag
from SLSFinite import *
from Polytope import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scienceplots

def optimize(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w):
    """
    Parameters
    ----------
    A_list: list of matrices [A_0, ...A_T]
    B_list: list of tensors [B_0, ...B_T]
    C_list: list of tensors [C_0, ...C_T]
        where A_t, B_t, C_t are the matrices in the dynamics of the system at time t.
    Poly_x, Poly_u, Poly_w: Polytope
        Polytopes cartesian products of polytopes
        for x, u and w over all times t = 0,...,T.

    Returns
    -------
    result: float
        Optimal cost (np.inf if problem is not feasible).
    SLS_data: SLSFinite object
        Instance of the class SLSFinite containing the variables corresponding
        to the optimal cost.
    Lambda: cvxpy.Variable, shape (H_x[0], H_w[0])
        Polytope containment variable corresponding to the optimal cost.
    """
    # load SLS constraints
    SLS_data = SLSFinite(A_list, B_list, C_list)
    constraints = SLS_data.SLP_constraints()
    # add polytope containment constraints
    Lambda_x = cp.Variable((np.shape(Poly_x.H)[0], np.shape(Poly_w.H)[0]))
    Lambda_u = cp.Variable((np.shape(Poly_u.H)[0], np.shape(Poly_w.H)[0]))
    constraints += [Lambda_x >= 0,
                    Lambda_x @ Poly_w.H == Poly_x.H @ cp.hstack([SLS_data.Phi_xx, SLS_data.Phi_xy]),
                    Lambda_x @ Poly_w.h <= Poly_x.h,
                    Lambda_u >= 0,
                    Lambda_u @ Poly_w.H == Poly_u.H @ cp.hstack([SLS_data.Phi_ux, SLS_data.Phi_uy]),
                    Lambda_u @ Poly_w.h <= Poly_u.h]
    # objective function
    objective = cp.Minimize(cp.norm(SLS_data.Phi_uy, 'nuc'))
    #objective = cp.Minimize(1)
    problem = cp.Problem(objective, constraints)
    result = problem.solve()

    return result, SLS_data, Lambda_x, Lambda_u

def run(key, save):
    T = 20; nx = 2; ny = 1; nu = 2
    #A = np.diag([2, 0.9]) first plots using this
    A = np.diag([1.1, 0.9]); B = np.eye(2); C = np.array([[1, 1]])
    A_list = (T+1)*[A]; B_list = (T+1)*[B]; C_list = (T+1)*[C]

    # define x polytope
    center_0 = [0, -0.8]
    half_0 = 0.1
    center_T = [0.7, 0.7]
    half_T = 0.3
    Poly_x = H_cube(center_0, half_0).cart( H_cube([0,0], 1).cartpower(T-1) ).cart(H_cube(center_T, half_T))

    # define u polytope
    u_scale = 1
    Poly_u = H_cube([0,0], u_scale).cartpower(T+1)

    # define w polytope
    w_scale = 0.01
    Poly_w = H_cube(center_0, half_0).cart( H_cube([0,0], w_scale).cartpower(T) ).cart( H_cube([0], w_scale).cartpower(T+1) ) 

    # solve problem
    [result, SLS_data, Lambda, Lambda_u] = optimize(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w)

    # check if feasible
    assert result != np.inf

    # plot trajectories
    def plot_trajectory(w, SLS_data, ax):
        x_traj = np.block([SLS_data.Phi_xx.value, SLS_data.Phi_xy.value]) @ w
        x_traj = x_traj.reshape((-1, 2))
        color = next(ax._get_lines.prop_cycler)['color']
        ax.plot(x_traj[0:,0], x_traj[0:,1], color = color, linewidth=0.6)
        ax.plot(*x_traj[0,:], '.', color = color)
        ax.plot(*x_traj[-1,:], '.', color = color)
        return

    #with plt.style.context(['science', 'ieee']):
    fig1, ax1 = plt.subplots()
    title1 = 'Trajectory ' + key
    ax1.set_title(title1)
    rect_0 = patches.Rectangle((center_0[0] - half_0, center_0[1] - half_0), 2*half_0, 2*half_0, linewidth=1, edgecolor='k', facecolor='lightcoral')
    rect_T = patches.Rectangle((center_T[0] - half_T, center_T[1] - half_T), 2*half_T, 2*half_T, linewidth=1, edgecolor='k', facecolor='lightskyblue')
    # corners trjactories
    sign = [1, -1]
    for i in sign:
        for j in sign:
            w = w_scale*np.random.uniform(-1, 1, (T+1)*(nx+ny))
            w[0:2] = center_0 + np.array([i*half_0,j*half_0])
            plot_trajectory(w, SLS_data, ax1)
    # random sample trajectories
    N = 90
    for i in range(N):
        w = w_scale*np.random.uniform(-1, 1, (T+1)*(nx+ny))
        w[0:2] = center_0 + np.random.uniform(-half_0, half_0, 2)
        plot_trajectory(w, SLS_data, ax1)
    ax1.add_patch(rect_0)
    ax1.add_patch(rect_T)
    ax1.set_xlim([-1,1])
    ax1.set_ylim([-1,1])
    #plt.grid()
    plt.show()

    fig2, ax2 = plt.subplots()
    # find F
    F = SLS_data.optimal_controller()
    print(F)
    epsilon = 10**-4
    #sparse = np.bitwise_or((F > epsilon), (F < -epsilon))
    #print(sparse)
    ax2.spy(F, precision=epsilon)
    title2 = 'Sparsiy of F ' + key
    ax2.set_title(title2)
    plt.show()

    fig3, ax3 = plt.subplots()
    # find singular values
    [U, S, Vh] = np.linalg.svd(F)
    ax3.plot(S,'.')
    print(S)
    title3 = 'Singular Values of F ' + key
    ax3.set_ylim(bottom=0)
    ax3.set_title(title3)
    plt.show()

    if save:
        fig1.savefig('figures/' + ''.join(title1.split()) + '.jpg', dpi=300)
        fig2.savefig('figures/' + ''.join(title2.split()) + '.jpg', dpi=300)
        fig3.savefig('figures/' + ''.join(title3.split()) + '.jpg', dpi=300)


    return

if __name__ == '__main__':
    key1 = 'Nuclear Norm Minimization for a Promoting System'
    key2 = 'Feasibility'
    save = False
    run(key1, save)





#def unit_test():
#    epsilon = 10 ** -4
#    assert np.all(Lambda.value >=  -epsilon)
#    assert np.all(-epsilon <= (Lambda.value @ Poly_w.H - Poly_x.H @ cp.hstack([SLS_data.Phi_xx, SLS_data.Phi_xy]).value))
#    assert np.all(epsilon >= (Lambda.value @ Poly_w.H - Poly_x.H @ cp.hstack([SLS_data.Phi_xx, SLS_data.Phi_xy]).value))
#    assert np.all(Lambda.value @ Poly_w.h <= Poly_x.h + epsilon)
#    return
