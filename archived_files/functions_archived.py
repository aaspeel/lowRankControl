import numpy as np
import cvxpy as cp
from SLSFinite_archived import *
from Polytope import *
import matplotlib.pyplot as plt
import copy
import sympy

def polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w):
    # load SLS constraints
    constraints = SLS_data.SLP_constraints()
    # add polytope containment constraints
    Poly_xu = Poly_x.cart(Poly_u)
    Lambda = cp.Variable((np.shape(Poly_xu.H)[0], np.shape(Poly_w.H)[0]), nonneg=True)
    constraints += [Lambda @ Poly_w.H == Poly_xu.H @ SLS_data.Phi_matrix,
                    Lambda @ Poly_w.h <= Poly_xu.h]
    return constraints, Lambda

def optimize(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, key, opt_eps, norm=None):
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
    else:
        raise Exception('Choose key: \'Feasibility\'.')
    problem = cp.Problem(objective, constraints)
    result = problem.solve( solver=cp.MOSEK,
                            mosek_params = {'MSK_DPAR_INTPNT_CO_TOL_DFEAS':  opt_eps,
                                            },
                            verbose=True)
    if problem.status != cp.OPTIMAL:
        raise Exception("Solver did not converge!")
    return result, SLS_data, Lambda, problem.status, constraints

def optimize_RTH(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, N, delta, rank_eps, opt_eps, feasibility):
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
        result = problem.solve(solver=cp.MOSEK,
                               mosek_params = {'MSK_DPAR_INTPNT_CO_TOL_DFEAS':  opt_eps,
                                            },
                               verbose=True)
        if problem.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")
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
    
    if feasibility == 'truncate_F':
        #print(np.linalg.matrix_rank(SLS_data.Phi_uy.value, 1e-4))
        SLS_data_list[-1].calculate_dependent_variables()
        SLS_data_list[-1].causal_rank_decomposition(rank_eps,'F')
        SLS_data_list[-1].F_trunc_to_Phi_trunc()
        #print(np.max(np.abs( SLS_data_list[-1].Phi_trunc - SLS_data_list[-1].F_to_Phi_2(D.dot(E)) )))
        #assert np.all(SLS_data_list[-1].Phi_trunc == SLS_data_list[-1].F_to_Phi_2(D.dot(E)))
        # check feasibility up to tolerance of optimization problem (1e-4 by default)
        Poly_xu = Poly_x.cart(Poly_u)
        print("rank F:", SLS_data_list[-1].rank_F_trunc)
        print("Error true polytope constraint:", np.max(np.abs( Lambda.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_data_list[-1].Phi_matrix.value))))
        print("Error true F and truncated F:", np.max( np.abs(SLS_data_list[-1].F - SLS_data_list[-1].F_trunc) ) )
        print("Error true Phi and truncated Phi:", np.max( np.abs(SLS_data_list[-1].Phi_matrix.value - SLS_data_list[-1].Phi_trunc) ) )
        print("Error truncated polytope constraint:", np.max( np.abs(Lambda.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_data_list[-1].Phi_trunc)) ) )
        print("Error Phi with SLP inverse:", np.max( np.abs(SLS_data_list[-1].Phi_matrix.value - SLS_data_list[-1].F_to_Phi(SLS_data_list[-1].F)) ) )
        _, S, _ = np.linalg.svd(SLS_data_list[-1].F)
        plt.plot(np.log10(S), '.-')
        plt.show()
        S_sort = np.flip(np.sort(S))
        print(S_sort)
        #assert np.all( np.isclose( Lambda.value.dot(Poly_w.H), Poly_xu.H.dot(SLS_data_list[-1].Phi_matrix.value) , atol = 1e-8) )
        #assert np.all( Lambda.value.dot(Poly_w.h) <= Poly_xu.h + 1e-6 )
        #assert np.all( np.isclose( Lambda.value.dot(Poly_w.H), (Poly_xu.H.dot(SLS_data_list[-1].F_to_Phi_2(SLS_data_list[-1].F))).astype('float') , atol = 1e-8) ) # this is 0, perfect causal factorization seems to work
        
        #_, S, _ = np.linalg.svd(SLS_data_list[-1].F)
        #plt.plot(np.log10(S), '.-')
        #plt.show()
        #assert np.all( np.isclose( Lambda.value.dot(Poly_w.H), (Poly_xu.H.dot(SLS_data_list[-1].Phi_trunc)).astype('float') , atol = np.max( np.abs(Lambda.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_data_list[-1].Phi_trunc))) )) 
        # for i in range(len(S_sort)):
        #     if i == len(S_sort)-1:
        #         rank_eps == S_sort[i]
        #     else:
        #         rank_eps = (S_sort[i] + S_sort[i+1])/2
        #     SLS_data_list[-1].calculate_dependent_variables()
        #     SLS_data_list[-1].causal_rank_decomposition(rank_eps)
        #     SLS_data_list[-1].F_trunc_to_Phi_trunc()
        #     Poly_xu = Poly_x.cart(Poly_u)
        #     Lambda_feas = cp.Variable((np.shape(Poly_xu.H)[0], np.shape(Poly_w.H)[0]), nonneg=True)
        #     constraints_feas = [Lambda_feas @ Poly_w.H == Poly_xu.H @ SLS_data_list[-1].Phi_trunc,
        #                         Lambda_feas @ Poly_w.h <= Poly_xu.h]
        #     objective_feas = cp.Minimize(1)
        #     problem_feas = cp.Problem(objective_feas, constraints_feas)
        #     result_feas = problem_feas.solve(solver=cp.MOSEK, 
        #                                      mosek_params = {'MSK_DPAR_INTPNT_CO_TOL_DFEAS':  1e-8,
        #                                     },
        #                                      verbose=True)
        #     if problem_feas.status == cp.OPTIMAL:
        #         break
        # #add truncated Phi and F that gives a feasible solution.
        
    elif feasibility == 'truncate_Phi_and_F':
        pass
        SLS = SLS_data_list[-1]
        SLS.calculate_dependent_variables()
        # truncate Phi
        SLS.causal_rank_decomposition(rank_eps, 'Phi')
        # solve feasibility with SLP + polytope constraints
        _, SLS_reopt, Lambda_reopt, _, _ = optimize(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, 'Feasibility', opt_eps, norm=['Reweighted Nuclear Norm', SLS.Phi_uy_trunc])
        # calculate reoptimized F
        SLS_reopt.calculate_dependent_variables()
        # truncate reoptimzed F
        SLS_reopt.causal_rank_decomposition(rank_eps, 'F')
        # check feasibility of truncated reoptimzed F
        SLS_reopt.F_trunc_to_Phi_trunc()
        Poly_xu = Poly_x.cart(Poly_u)
        print("rank Phi_uy:", SLS.rank_Phi_uy_trunc)
        print("Error Phi_uy and truncated Phi_uy:", np.max( np.abs(SLS.Phi_uy.value - SLS.Phi_uy_trunc) ) )
        print("Error F and reoptimized F:", np.max( np.abs(SLS.F - SLS_reopt.F) ) )
        print("rank reoptized F:", SLS_reopt.rank_F_trunc)
        print("Error reoptimized F and reoptimzed truncated F:", np.max( np.abs(SLS_reopt.F - SLS_reopt.F_trunc) ) )
        print("Error reoptimzed Phi and reoptimzed truncated Phi:", np.max( np.abs(SLS_reopt.Phi_matrix.value - SLS_reopt.Phi_trunc) ) )
        print("Error reoptimized truncated polytope constraint:", np.max( np.abs(Lambda_reopt.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_reopt.Phi_trunc)) ) )
        _, S_Phi_uy, _ = np.linalg.svd(SLS.Phi_uy.value)
        _, S_Phi_uy_trunc, _ = np.linalg.svd(SLS.Phi_uy_trunc)
        _, S_F_reopt, _ = np.linalg.svd(SLS_reopt.F)
        _, S_F_reopt_trunc, _ = np.linalg.svd(SLS_reopt.F_trunc)
        plt.plot(np.log10(S_Phi_uy), '.-', label='Phi_uy')
        plt.plot(np.log10(S_Phi_uy_trunc), '.-', label='Phi_uy_trunc')
        plt.plot(np.log10(S_F_reopt), '.-', label='F_trans')
        plt.plot(np.log10(S_F_reopt_trunc), '.-', label='F_trans_trunc')
        plt.legend()
        plt.show()
        SLS_data_list[-1] = SLS_reopt
        Lambda = Lambda_reopt
    elif feasibility == "truncate_F_svd":
        SLS_data_list[-1].calculate_dependent_variables()
        U, S, Vh = np.linalg.svd(SLS_data_list[-1].F)
        old_F = SLS_data_list[-1].F
        plt.plot(np.log10(S), '.-', label='singular values of F')
        plt.legend()
        plt.show()
        # truncate svd (self.F)
        keep_inds = np.where(S>1e-7)[0]
        print(keep_inds)
        svd_trunc_F = U[:,keep_inds].dot(np.diag(S[keep_inds])).dot(Vh[keep_inds, :])
        #k=10
        #U_int = sympy.Matrix((10**k*np.round(U[:,keep_inds],k)).astype('int64'))
        #S_int = sympy.Matrix((10**k*np.round(np.diag(S[keep_inds]),k)).astype('int64'))
        #Vh_int = sympy.Matrix((10**k*np.round(Vh[keep_inds, :],k)).astype('int64'))
        #svd_trunc_F = np.array(10**(-3*k)*U_int*S_int*Vh_int)
        # compute false causal factorization of svd trunc F (self.F_trunc)
        SLS_data_list[-1].F = svd_trunc_F
        SLS_data_list[-1].causal_rank_decomposition(1e-7, 'F')
        false_causal_svd_trunc_F = SLS_data_list[-1].F_trunc
        row_indices = SLS_data_list[-1].F_causal_row_basis
        # compute true causal factorization of svd trunc F (self.F)
        true_E = old_F[row_indices,:]
        true_causal_svd_trunc_F = SLS_data_list[-1].D.dot(true_E)
        print('Error F and svd trunc F:', np.max( np.abs(old_F - svd_trunc_F) ))
        print('Error svd trunc F and svd false causal svd trunc F:', np.max( np.abs(svd_trunc_F - false_causal_svd_trunc_F) ))
        print('Error false causal svd trunc F and true causal svd trunc F:', np.max( np.abs(false_causal_svd_trunc_F - true_causal_svd_trunc_F) ))
        print('Error true causal svd trunc F and F:', np.max( np.abs(true_causal_svd_trunc_F - old_F) ))
        Phi_matrix_trunc = SLS_data_list[-1].F_to_Phi(true_causal_svd_trunc_F)
        Poly_xu = Poly_x.cart(Poly_u)
        print("Error Phi and truncated Phi:", np.max( np.abs(SLS_data_list[-1].Phi_matrix.value - Phi_matrix_trunc) ) )
        print("Error polytope constraint:", np.max( np.abs(Lambda.value.dot(Poly_w.H) - Poly_xu.H.dot(Phi_matrix_trunc)) ) )
        plt.figure()
        plt.show()
        # if all is good this should now work
        SLS_data_list[-1].F_trunc = true_causal_svd_trunc_F
        SLS_data_list[-1].F_trunc_to_Phi_trunc()
    return [result_list, SLS_data_list, Lambda, problem.status, constraints]

def optimize_reweighted_atomic(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, N, key, delta, opt_eps):
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
        result = problem.solve(solver=cp.MOSEK,
                               mosek_params = {'MSK_DPAR_INTPNT_CO_TOL_DFEAS':  opt_eps,
                                            },
                               verbose=True)
        if problem.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")
        result_list[k] = result
        SLS_data_list[k] = copy.deepcopy(SLS_data)
        
        #update params
        if key == 'Reweighted Sensor Norm':
            norm_list[k] = np.linalg.norm(SLS_data.Phi_uy.value, 2, 0)
            W.value = (np.linalg.norm(SLS_data.Phi_uy.value, 2, 0) + delta)**-1
        if key == 'Reweighted Actuator Norm':
            norm_list[k] = np.linalg.norm(SLS_data.Phi_uy.value, 2, 1)
            W.value = (np.linalg.norm(SLS_data.Phi_uy.value, 2, 1) + delta)**-1
    return result_list, SLS_data_list, norm_list, Lambda, problem.status, constraints


def optimize_sparsity(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, key, N, delta, rank_eps, opt_eps):
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
    result_list, SLS_list, norm_list, Lambda, status, constraints = optimize_reweighted_atomic(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, N, key, delta, opt_eps)
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
    ind = np.where(norm_argmin<=rank_eps)[0]
    reopt_result, reopt_SLS, Lambda, status, constraints = optimize(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, 'Feasibility', rank_eps, norm=[key, ind])
    re_opt_ind = [i for i in np.arange(len(norm_argmin)) if i not in ind] # take complement
    reopt_SLS.calculate_dependent_variables() # only get F, no other truncation needed.
    return [[reopt_result], [reopt_SLS], norm_list, re_opt_ind, SLS_list, Lambda, constraints]