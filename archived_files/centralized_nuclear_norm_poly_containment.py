import numpy as np
import scipy as sp
import cvxpy as cp
from scipy.linalg import block_diag
from SLSFinite import *
from Polytope import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scienceplots
import math
import copy
import time
import sys

def polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w):
    # load SLS constraints
    constraints = SLS_data.SLP_constraints()
    # add polytope containment constraints
    Poly_xu = Poly_x.cart(Poly_u)
    Lambda = cp.Variable((np.shape(Poly_xu.H)[0], np.shape(Poly_w.H)[0]), nonneg=True)
    constraints += [Lambda @ Poly_w.H == Poly_xu.H @ SLS_data.Phi_matrix,
                    Lambda @ Poly_w.h <= Poly_xu.h]
    return constraints, Lambda

def optimize(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, key, norm=None):
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
    # constraints
    SLS_data = SLSFinite(A_list, B_list, C_list, norm)
    [constraints, Lambda] = polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w)
    # objective function
    if key == 'Feasibility':
        objective = cp.Minimize(0)
    elif key == 'Nuclear Norm':
        objective = cp.Minimize(cp.norm(SLS_data.Phi_uy, 'nuc'))
    elif key == 'Sensor Norm':
        objective = cp.Minimize(cp.sum(cp.norm(SLS_data.Phi_uy, 2, 0)))
    elif key == 'Actuator Norm':
        objective = cp.Minimize(cp.sum(cp.norm(SLS_data.Phi_uy, 2, 1)))
    else:
        raise Exception('Choose key: \'Feasibility\' or \'Nuclear Norm\'.')
    problem = cp.Problem(objective, constraints)
    #print(problem.is_qp())
    result = problem.solve(solver=cp.MOSEK, verbose=True)
    return result, SLS_data, Lambda, problem.status

def optimize_RTH(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, N, key, delta):
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
    N: int
        Number of iterations of the reweighted nuclear norm iteration.
    key: str
        Set the cost function.

    Returns
    -------
    result_list: float
        Optimal cost (np.inf if problem is not feasible).
    SLS_data_list: SLSFinite object
        List of Instance of the class SLSFinite containing the variables corresponding
        to the optimal cost.
    Lambda: cvxpy.Variable, shape (H_x[0] + H_u[0], H_w[0])
        Polytope containment variable corresponding to the optimal cost.
    """
    SLS_data = SLSFinite(A_list, B_list, C_list)
    [constraints, Lambda] = polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w)
    # Initialize Paramters
    W_1 = cp.Parameter(2*[SLS_data.nu*(SLS_data.T+1)])
    W_2 = cp.Parameter(2*[SLS_data.ny*(SLS_data.T+1)])
    W_1.value = delta**(-1/2)*np.eye(SLS_data.nu*(SLS_data.T+1))
    W_2.value = delta**(-1/2)*np.eye(SLS_data.ny*(SLS_data.T+1))
    result_list = N*[None]
    SLS_data_list = N*[None]
    objective = cp.Minimize(cp.norm(W_1 @ SLS_data.Phi_uy @ W_2, 'nuc'))
    #define problem
    problem = cp.Problem(objective, constraints)
    for k in range(N):
        result = problem.solve(solver=cp.MOSEK, warm_start=True, verbose=True)
        if problem.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")
        #print('Iteration ' + str(k) + ' solve time:', problem.solver_stats.solve_time)
        result_list[k] = result
        SLS_data_list[k] = copy.deepcopy(SLS_data)
        #update params
        [U, S, Vh] = np.linalg.svd((W_1 @ SLS_data.Phi_uy @ W_2).value)
        Y = np.linalg.inv(W_1.value).dot((W_1 @ SLS_data.Phi_uy @ W_2).value).dot(Vh.T).dot(U.T).dot(np.linalg.inv(W_1.value))
        Z = np.linalg.inv(W_2.value).dot(Vh.T).dot(U.T).dot((W_1 @ SLS_data.Phi_uy @ W_2).value).dot(np.linalg.inv(W_2.value))
        # help function
        def update_W(Q, dim, delta):
            W = (Q + delta*np.eye(dim))
            [eig, eigv] = np.linalg.eigh(W)
            assert np.all(eig > 0)
            W = eigv.dot(np.diag(eig**(-1/2))).dot(np.linalg.inv(eigv))
            return W
        W_1.value = update_W(Y, SLS_data.nu*(SLS_data.T+1), delta)
        W_2.value = update_W(Z, SLS_data.ny*(SLS_data.T+1), delta)
    print(np.linalg.matrix_rank(SLS_data.Phi_uy.value, 1e-4))
    return result_list, SLS_data_list, Lambda, problem.status

def optimize_reweighted_atomic(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, N, key, delta):
    SLS_data = SLSFinite(A_list, B_list, C_list)
    [constraints, Lambda] = polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w)
    # Initialize Paramters
    if key == 'Reweighted Sensor Norm':
        W = cp.Parameter(SLS_data.ny*(SLS_data.T+1))
        objective = cp.Minimize(cp.sum(cp.norm(SLS_data.Phi_uy @ cp.diag(W), 2, 0)))
        W.value = delta**-1 * np.ones(SLS_data.ny*(SLS_data.T+1))
    elif key == 'Reweighted Actuator Norm':
        W = cp.Parameter(SLS_data.nu*(SLS_data.T+1))
        objective = cp.Minimize(cp.sum(cp.norm(cp.diag(W) @ SLS_data.Phi_uy, 2, 1)))
        W.value = delta**-1 * np.ones(SLS_data.nu*(SLS_data.T+1))
    
    result_list = N*[None]
    SLS_data_list = N*[None]
    norm_list = N*[None]
    #define problem
    problem = cp.Problem(objective, constraints)
    for k in range(N):
        result = problem.solve(solver=cp.MOSEK, verbose=True)
        if problem.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")
        #print('Iteration ' + str(k) + ' solve time:', problem.solver_stats.solve_time)
        result_list[k] = result
        SLS_data_list[k] = copy.deepcopy(SLS_data)
        
        #update params
        if key == 'Reweighted Sensor Norm':
            norm_list[k] = np.linalg.norm(SLS_data.Phi_uy.value, 2, 0)
            W.value = (np.linalg.norm(SLS_data.Phi_uy.value, 2, 0) + delta)**-1
        if key == 'Reweighted Actuator Norm':
            norm_list[k] = np.linalg.norm(SLS_data.Phi_uy.value, 2, 1)
            W.value = (np.linalg.norm(SLS_data.Phi_uy.value, 2, 1) + delta)**-1
        
    return result_list, SLS_data_list, norm_list, Lambda, problem.status

def optimize_sparsity(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, key, N, epsilon, delta):
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
    N: int
        Number of iterations of the reweighted nuclear norm iteration.
    key: str
        Set the cost function.

    Returns
    -------
    result_list: float
        Optimal cost (np.inf if problem is not feasible).
    SLS_data_list: SLSFinite object
        List of Instance of the class SLSFinite containing the variables corresponding
        to the optimal cost.
    Lambda: cvxpy.Variable, shape (H_x[0] + H_u[0], H_w[0])
        Polytope containment variable corresponding to the optimal cost.
    """
    result_list, SLS_list, norm_list, Lambda, status = optimize_reweighted_atomic(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, N, key, delta)
    SLS = SLS_list[-1]
    if status != cp.OPTIMAL:
        raise Exception('Atomic norm minimization did not converge.')
    #print('Found minimum ' + key + '. Trying to remove columns (if sensor norm) or rows (if actuator norm).')
    argmin = SLS.Phi_uy.value
    if key == 'Reweighted Sensor Norm':
        norm_argmin = np.linalg.norm(argmin, 2, 0)
    elif key == 'Reweighted Actuator Norm':
        norm_argmin = np.linalg.norm(argmin, 2, 1)
    else:
        raise Exception('Choose either the reweighted sensor or actuator norm to minimize!')
    #norm_argmin_sorted = np.sort(norm_argmin)
    #for i in range(len(norm_argmin_sorted)):
    #ind = np.where(norm_argmin<=norm_argmin_sorted[i])[0]
    ind = np.where(norm_argmin<=epsilon)[0]
    #assert len(ind) == i+1
    # define new variable
    test_result, test_SLS, _, status = optimize(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, 'Feasibility', norm=[key, ind])
    #if status != cp.OPTIMAL:
    #    break
    #compute complement to ind
    reopt_result, reopt_SLS = test_result, test_SLS
    re_opt_ind = [i for i in np.arange(len(norm_argmin)) if i not in ind]
    print(len(re_opt_ind))
    return [reopt_result], [reopt_SLS], norm_list, re_opt_ind, Lambda

#def reoptimize_atomic(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, remove_indices, T):
#    SLS_data = SLSFinite(A_list, B_list, C_list)
#    [constraints, Lambda] = polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w)
#    for i in range(len())

def causal_rank_decomposition(F, nu, ny, T, rank_eps):
    assert F.shape[0] == nu*(T+1)
    assert F.shape[1] == ny*(T+1)
    E = np.array([]).reshape((0,F.shape[1]))
    D = np.array([]).reshape((0,0)) # trick
    rank_counter = 0
    for t in range(T+1):
        for s in range(nu):
            row = t*nu + s
            n_cols = t*ny + ny
            submat_new = F[0:row+1, :] # rows up to row (note: last step row = (T+1)*nu)
            rank_new = np.linalg.matrix_rank(submat_new, tol=rank_eps)
            if rank_new - rank_counter == 1:
                rank_counter += 1
                # add vector to E matrix
                E = np.vstack([E, submat_new[row:row+1, :]])
                # modify D matrix
                unit = np.zeros([1, rank_counter]) # E has shape (row+1,rank_counter)
                unit[0, -1] = 1. # add a 1 at the last column of unit
                D = np.hstack([D, np.zeros([row, 1])]) # E is (row, rank_counter)
                D = np.vstack([D, unit]) # E has (row+1, rank_counter)
                assert E.shape == (rank_counter, F.shape[1])
                assert D.shape == (row+1, rank_counter)
            elif rank_new == rank_counter:
                # solve equation
                #c = np.linalg.solve(E[:, 0:rank_counter].T, submat_new[-1, 0:rank_counter])
                c = np.linalg.lstsq(E[:, 0:rank_counter].T, submat_new[-1, 0:rank_counter])[0]
                c = c.reshape(([1, rank_counter]))
                D = np.vstack([D, c])
                assert E.shape == (rank_counter, F.shape[1])
                assert D.shape == (row+1, rank_counter)
            else:
                raise Exception('Rank increased more than 1.')
    assert E.shape == (rank_counter, F.shape[1])
    assert D.shape == (F.shape[0], rank_counter)
    assert rank_counter == np.linalg.matrix_rank(F, tol=rank_eps)
    return D, E, rank_counter

if __name__ == '__main__':
    # A = np.diag([2, 0.9]); B = np.eye(2); C = np.array([[1, 1]])
    # A = np.diag([1.1, 0.9]); B = np.eye(2); C = np.array([[1, 1]])
    np.random.seed(1)
    T = 10
    dt = 1
    A_0 = np.block([[np.zeros([2,2]), np.eye(2)], [np.zeros([2,2]), np.zeros([2,2])]])
    B_0 = np.block([[np.zeros([2,2])],[np.eye(2)]])
    A = sp.linalg.expm(A_0*dt)
    B = np.sum([np.linalg.matrix_power(A_0*dt,i)/math.factorial(i+1) for i in np.arange(100)], axis=0).dot(B_0)
    C = np.block([[np.eye(2), np.zeros([2,2])]])
    A_list = (T+1)*[A]; B_list = (T+1)*[B]; C_list = (T+1)*[C]

    # define x polytope


    #center_0 = [0, -0.8]
    #half_0 = 2*[0.1]
    #center_T = [0.7, 0.7]
    #half_T = 2*[0.3]
    #Poly_x = H_cube(center_0, half_0).cart( H_cube([0,0], 1).cartpower(T-1) ).cart(H_cube(center_T, half_T))

    np.random.seed(1)
    dt = 1
    A_0 = np.block([[np.zeros([2,2]), np.eye(2)], [np.zeros([2,2]), np.zeros([2,2])]])
    B_0 = np.block([[np.zeros([2,2])],[np.eye(2)]])
    A = sp.linalg.expm(A_0*dt)
    B = np.sum([np.linalg.matrix_power(A_0*dt,i)/math.factorial(i+1) for i in np.arange(100)], axis=0).dot(B_0)
    C = np.block([[np.eye(2), np.zeros([2,2])]])
    A_list = (T+1)*[A]; B_list = (T+1)*[B]; C_list = (T+1)*[C]
    max_v = 4
    max_x0 = 1
    max_v0 = 0
    box_x = 10
    center_times = (T+1)*[[0,0,0,0]]
    radius_times = (T+1)*[[box_x, box_x, max_v, max_v]]
    box_check = [0,5,10]
    center_times[box_check[0]] = [-7,-7,0,0]
    radius_times[box_check[0]] = [max_x0,max_x0,max_v0,max_v0]
    center_times[box_check[1]] = [7,-7,0,0]
    radius_times[box_check[1]] = [2,2,max_v,max_v]
    center_times[box_check[2]] = [7,7,0,0]
    radius_times[box_check[2]] = [2,2,1,1]
    Poly_x = cart_H_cube(center_times, radius_times)
    u_scale = 4
    Poly_u = H_cube([0,0], u_scale).cartpower(T+1)
    wx_scale = 0.05
    wxdot_scale = 0.05
    v_scale = 0.05
    Poly_w = H_cube(center_times[0], radius_times[0]).cart( H_cube([0,0,0,0], [wx_scale, wx_scale, wxdot_scale, wxdot_scale]).cartpower(T) ).cart( H_cube([0,0], v_scale).cartpower(T+1) ) 
    delta = 0.01
    N=10
    test_feas = True
    # solve problem
    ### simulation #########################################################################################################

    #key = 'Feasibility'
    #key = 'Reweighted Sensor Norm'
    #key = 'Reweighted Actuator Norm'
    key = 'Reweighted Nuclear Norm'
    save = False
    direc = 'T20_1/'
    #result, SLS, Lambda, status = optimize(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, key, norm = None); result_list = [result]; SLS_list = [SLS]
    result_list, SLS_list, Lambda, status = optimize_RTH(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, N, key, delta)
    #for key in ['Reweighted Sensor Norm', 'Reweighted Actuator Norm']:
    #result_list, SLS_list, norm_list, reopt_ind_list, Lambda = optimize_sparsity(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, key, N, 1e-4, delta)
    #result_list, SLS_list, norm_list, Lambda, status = optimize_reweighted_atomic(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, N, key)
    result = result_list[-1]
    SLS = SLS_list[-1]

    ###########################################################################################################

    # update dependent variables
    SLS.calculate_dependent_variables()

    # plot trajectories
    def plot_trajectory(w, Phi_row, n_dim, ax):
        traj = Phi_row @ w
        traj = traj.reshape((-1, n_dim))
        #print(x_traj)
        color = next(ax._get_lines.prop_cycler)['color']
        ax.plot(traj[:,0], traj[:,1], color = color, linewidth=0.6)
        ax.plot(traj[0,0], traj[0,1], '.', color = color)
        ax.plot(traj[-1,0], traj[-1,1], '.', color = color)
        return

    def plot_control(w, SLS, axs):
        u_traj = np.block([SLS.Phi_ux.value, SLS.Phi_uy.value]) @ w
        u_traj = u_traj.reshape((-1,SLS.nu))
        color = next(axs[0]._get_lines.prop_cycler)['color']
        axs[0].plot(u_traj[:,0], color = color, linewidth=0.6)
        axs[1].plot(u_traj[:,1], color = color, linewidth=0.6)
        return

    #with plt.style.context(['science', 'ieee']):
    fig1, ax1 = plt.subplots()
    title1 = 'Trajectory - ' + key
    ax1.set_title(title1)
    colors = [(255/255,114/255,118/255,0.5), (153/255,186/255,221/255,0.5), (204/255,255/255,204/255,0.5), (204/255,255/255,204/255,0.5), (204/255,255/255,204/255,0.5), (204/255,255/255,204/255,0.5)]
    counter = 0
    for i in box_check:
        rect = patches.Rectangle((center_times[i][0] - radius_times[i][0], center_times[i][1] - radius_times[i][1]), 2*radius_times[i][0], 2*radius_times[i][1], linewidth=1, edgecolor='k', facecolor= colors[counter], label='$\mathcal{X}_{' + str(i) + '}$')
        ax1.add_patch(rect)
        counter += 1
    sign = [1, -1]

    #for i in sign:
    #    for j in sign:
    #        #w = w_scale*np.random.uniform(-1, 1, (T+1)*(SLS.nx+SLS.ny))
    #        w_h = w_scale*np.ones((T+1)*(SLS.nx+SLS.ny))
    #        w_l = -w_scale*np.ones((T+1)*(SLS.nx+SLS.ny))
    #        w_h[0:SLS.nx] = center_times[0]
    #        w_h[0:2] += np.array([i*radius_times[0][0],j*radius_times[0][1]])
    #        w_l[0:SLS.nx] = w_h[0:SLS.nx]; w_l[0:2] = w_h[0:2]
    #        plot_trajectory(w_h, np.block([SLS.Phi_xx.value, SLS.Phi_xy.value]), SLS.nx, ax1)
    #        plot_trajectory(w_l, np.block([SLS.Phi_xx.value, SLS.Phi_xy.value]), SLS.nx, ax1)


    N = 90
    u_trajs = np.zeros([N, SLS.nu*(SLS.T+1)])
    for i in range(N):
        w = np.array([])
        for _ in range(T+1):
            w = np.hstack([w, wx_scale*np.random.uniform(-1, 1, 2), wxdot_scale*np.random.uniform(-1, 1, 2)])
        w = np.hstack([w, v_scale*np.random.uniform(-1, 1, (T+1)*(SLS.ny))])
        #print(w)
        #w = np.concatenate([wx_scale*np.random.uniform(-1, 1, (T+1)*SLS.nx), v_scale*np.random.uniform(-1, 1, (T+1)*SLS.ny)])
        w[0:SLS.nx] = center_times[0]
        w[0] += np.random.uniform(-radius_times[0][0], radius_times[0][0])
        w[1] += np.random.uniform(-radius_times[0][1], radius_times[0][1])
        w[2] += np.random.uniform(-radius_times[0][2], radius_times[0][2])
        w[3] += np.random.uniform(-radius_times[0][3], radius_times[0][3])
        #plot traj
        plot_trajectory(w, np.block([SLS.Phi_xx.value, SLS.Phi_xy.value]), SLS.nx, ax1)
        #u statistics
        u_trajs[i,:] = np.block([SLS.Phi_ux.value, SLS.Phi_uy.value]).dot(w)

    ax1.set_xlim([-12,12])
    ax1.set_ylim([-12,12])
    ax1.grid()

    fig2, ax2 = plt.subplots()
    # find F
    #F = SLS.F
    epsilon = 10**-4
    ax2.spy(SLS.F, precision=epsilon)
    title2 = 'Sparsiy of $\mathbf{K}$ - ' + key
    save2 = 'Sparsity of K - ' + key
    ax2.set_title(title2)

    # write phi uy instead of F, also for rank!!! its not always true rank of F is the same as that of Phi uy when we have approimations!


    if key == 'Reweighted Nuclear Norm' or key == 'Feasibility' or key == 'Nuclear Norm':
        fig30, ax30 = plt.subplots()
        for k in range(len(SLS_list)):
            SLS_k = SLS_list[k]
            SLS_k.calculate_dependent_variables()
            [U, S, Vh] = np.linalg.svd(SLS_k.Phi_uy.value)
            #print(S)
            ax30.plot(np.arange(1,S.size+1), np.log10(S),'.-', label='k = ' + str(k + 1), linewidth=0.5, markersize=5)
            title30 = 'Singular Values of $\mathbf{\Phi}_{uy}$ - ' + key
            save30 = 'Singular Values of Phi_uy - ' + key
            ax30.set_ylim(bottom=-12)
            ax30.set_ylabel('$\log_{10}(\sigma_i(\mathbf{\Phi}_{uy}))$')
            ax30.set_xlabel('singular value index $i$')
            #ax30.set_title(title30)
            ax30.grid()
            ax30.legend()
            fig31, ax31 = plt.subplots()
            for k in range(len(SLS_list)):
                SLS_k = SLS_list[k]
                SLS_k.calculate_dependent_variables()
                [U, S, Vh] = np.linalg.svd(SLS_k.F)
                #print(S)
                ax31.plot(np.arange(1,S.size+1), np.log10(S),'.-', label='k = ' + str(k + 1), linewidth=0.5, markersize=6)
            title31 = 'Singular Values of $\mathbf{K}$ - ' + key
            save31 = 'Singular Values of K - ' + key
            ax31.set_ylim(bottom=-12)
            ax31.set_ylabel('$\log_{10}(\sigma_i(\mathbf{K}))$')
            ax31.set_xlabel('singular value index $i$')
            #ax31.set_title(title31)
            ax31.grid()
            ax31.legend()
            if save:
                fig30.savefig('figures/' + direc + ''.join(save30.split()) + '.jpg', dpi=300)
                fig31.savefig('figures/' + direc + ''.join(save31.split()) + '.jpg', dpi=300)
    if key == 'Sensor Norm' or key == 'Actuator Norm' or key == 'Reweighted Sensor Norm' or key == 'Reweighted Actuator Norm':
        fig6, ax6 = plt.subplots()
        for k in range(len(norm_list)):
            norm_k = norm_list[k]
            #print(S)
            ax6.plot(np.arange(1,norm_k.size+1), np.log10(norm_k),'.-', label='k = ' + str(k + 1), linewidth=0.5, markersize=6)
        title6 = 'Atomic Norm of $\mathbf{\Phi}_{uy}$ - ' + key
        save6 = 'Atomic Norm of Phi_uy - ' + key
        ax6.set_ylim(bottom=-12)
        if key == 'Actuator Norm' or key == 'Reweighted Actuator Norm':
            ax6.set_ylabel('$\log_{10}(||{\mathbf{\Phi}_{uy}}_{(i,:)}||)$')
            ax6.set_xlabel('row $i$')
        if key == 'Sensor Norm' or key == 'Reweighted Sensor Norm':
            ax6.set_ylabel('$\log_{10}(||{\mathbf{\Phi}_{uy}}_{(:,i)}||)$')
            ax6.set_xlabel('column $i$')
        #ax6.set_title(title6)
        ax6.grid()
        ax6.legend()
        fig61, ax61 = plt.subplots()
        for k in range(len(norm_list)):
            norm_k = norm_list[k]
            #print(S)
            ax61.plot(np.arange(1,norm_k.size+1), np.log10(norm_k),'.-', label='k = ' + str(k + 1), linewidth=0.5, markersize=6)
        title61 = 'Atomic Norm of $\mathbf{K}$ - ' + key
        save61 = 'Atomic Norm of K - ' + key
        ax61.set_ylim(bottom=-12)
        if key == 'Actuator Norm' or key == 'Reweighted Actuator Norm':
            ax61.set_ylabel('$\log_{10}(||\mathbf{K}_{(i,:)}||)$')
            ax61.set_xlabel('row $i$')
        if key == 'Sensor Norm' or key == 'Reweighted Sensor Norm':
            ax61.set_ylabel('$\log_{10}(||\mathbf{K}_{(:,i)}||)$')
            ax61.set_xlabel('column $i$')
        #ax6.set_title(title6)
        ax61.grid()
        ax61.legend()
        if save:
            fig6.savefig('figures/' + direc + ''.join(save6.split()) + '.jpg', dpi=300)
            fig61.savefig('figures/' + direc + ''.join(save61.split()) + '.jpg', dpi=300)


        fig4, axs = plt.subplots(2)
        u_trajs = u_trajs.reshape((N,-1,SLS.nu))
        u_trajs_mean = np.mean(u_trajs,axis=0)
        u_trajs_max = np.max(u_trajs - u_trajs_mean,axis=0)
        u_trajs_min = np.max(u_trajs_mean - u_trajs,axis=0)
        u_trajs_std = np.std(u_trajs, axis=0)
        times = np.arange(SLS.T+1)
        m = 5
        c = 3
        l = 0.6
        e = l
        axs[0].errorbar(times, u_trajs_mean[:,0],yerr=[u_trajs_max[:,0],u_trajs_min[:,0]],fmt='.', markersize=0, capsize=c, color="k", linewidth=l, capthick=e, label='range')
        axs[0].errorbar(times, u_trajs_mean[:,0],yerr=u_trajs_std[:,0],fmt='.', markersize=0, capsize=c, color="b", linewidth=l, capthick=e, label='standard deviation')
        axs[0].plot(times, u_trajs_mean[:,0], '.-', markersize=m, linewidth=l, color="b")
        title4 = '$u_1, u_2$ - ' + key
        axs[0].set_title(title4)
        axs[0].set_ylabel('$u_1(t)$')
        axs[0].set_xlabel('$t$')
        axs[0].legend()
        axs[0].grid()
        axs[1].errorbar(times, u_trajs_mean[:,1],yerr=[u_trajs_max[:,1],u_trajs_min[:,1]],fmt='.', markersize=0, capsize=c, color="k", linewidth=l, capthick=e, label='range')
        axs[1].errorbar(times, u_trajs_mean[:,1],yerr=u_trajs_std[:,1],fmt='.', markersize=0, capsize=c, color="r", linewidth=l, capthick=e, label='standard deviation')
        axs[1].plot(times, u_trajs_mean[:,1], '.-', markersize=m, linewidth=l, color="r")
        axs[1].set_ylabel('$u_2(t)$')
        axs[1].set_xlabel('$t$')
        axs[1].legend()
        axs[1].grid()

    if key == 'Nuclear Norm' or key == 'Reweighted Nuclear Norm':
        fig5, axs = plt.subplots(1,2)
        Poly_xu = Poly_x.cart(Poly_u)
        D, E, rank_F = causal_rank_decomposition(SLS.F, SLS.nu, SLS.ny, SLS.T, 10**-7)
        Phi_trunc = SLS.F_to_Phi_2(D.dot(E))
        Lambda = cp.Variable((np.shape(Poly_xu.H)[0], np.shape(Poly_w.H)[0]), nonneg=True)
        constraints = []
        constraints += [Lambda @ Poly_w.H == Poly_xu.H @ Phi_trunc,
                        Lambda @ Poly_w.h <= Poly_xu.h]
        obj = cp.Minimize(1)
        prob = cp.Problem(obj, constraints)
        #_ = prob.solve(solver=cp.MOSEK, verbose=True)
        
        title5 = 'Causal Encoder-Decoder Decomposition - ' + key
        #fig5.suptitle(title5)
        axs[0].set_title('Decoder - ' + key)
        axs[0].spy(D)
        axs[1].set_title('Encoder - ' + key)
        axs[1].spy(E)

        cons = []
        print(np.sum(np.abs(SLS.Phi_matrix.value - Phi_trunc)))
        if save:
                fig5.savefig('figures/' + direc + ''.join(title5.split()) + '.jpg', dpi=300)


    # check feasibility
    Poly_xu = Poly_x.cart(Poly_u)
    #original solution
    #assert np.all( np.isclose( (Lambda.value.dot(Poly_w.H)).astype('float'), (Poly_xu.H.dot(SLS.Phi_matrix.value)).astype('float') , atol = 1e-6) )
    #assert np.all( (Lambda.value.dot(Poly_w.h)).astype('float') <= (Poly_xu.h).astype('float') + 1e-6 )

    #truncated solution
    #print(np.isclose( (Lambda.value.dot(Poly_w.H)).astype('float'), (Poly_xu.H.dot(Phi_trunc)).astype('float') , atol = 1e-6) )
    #print(np.isclose( (Lambda.value.dot(Poly_w.H)).astype('float'), (Poly_xu.H.dot(Phi_trunc)).astype('float') , atol = 1e-5) )

    plt.show()

    if save:
        fig1.savefig('figures/' + direc + ''.join(title1.split()) + '.jpg', dpi=300)
        fig2.savefig('figures/' + direc + ''.join(save2.split()) + '.jpg', dpi=300)
        fig4.savefig('figures/' + direc + ''.join(title4.split()) + '.jpg', dpi=300)

#    return

def run():
    t = ""
    with open('centralized_nuclear_norm_poly_containment.py') as f:
        t = f.read()
    return t