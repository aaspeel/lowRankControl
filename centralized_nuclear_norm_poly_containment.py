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


key = 'Nuclear Norm Minimization'
#key = 'Feasibility'
save = True
#simulate(key1, save)

#def simulate(key, save):
T = 50
# A = np.diag([2, 0.9]); B = np.eye(2); C = np.array([[1, 1]])
# A = np.diag([1.1, 0.9]); B = np.eye(2); C = np.array([[1, 1]])


dt = 1
A = np.eye(4) + dt*np.block([[np.zeros([2,2]), np.eye(2)], [np.zeros([2,2]), np.zeros([2,2])]])
B = dt*np.block([[np.zeros([2,2])],[np.eye(2)]])

#A_0 = np.block([[np.zeros([2,2]), np.eye(2)], [np.zeros([2,2]), np.zeros([2,2])]])
#B_0 = np.block([[np.zeros([2,2])],[np.eye(2)]])
#A = np.exp(A_0*dt)
#B = np.sum([np.linalg.matrix_power(A_0*dt,i)/math.factorial(i+1) for i in np.arange(100)], axis=0) @ B_0
#print(A)
#print(B)
C = np.block([[np.eye(2), np.zeros([2,2])]])
#print(A)
#print(B)
#print(C)

#B = np.eye(2); C = np.array([[1, 1]])
A_list = (T+1)*[A]; B_list = (T+1)*[B]; C_list = (T+1)*[C]

# define x polytope
#center_0 = [0, -0.8]
#half_0 = 2*[0.1]
#center_T = [0.7, 0.7]
#half_T = 2*[0.3]
#Poly_x = H_cube(center_0, half_0).cart( H_cube([0,0], 1).cartpower(T-1) ).cart(H_cube(center_T, half_T))

max_v = 0.1
center_times = [[-.7,-.7,0,0]] + (T//2-1)*[[0,0,0,0]] + [[.7,-.7,0,0]] + (T//2-1)*[[0,0,0,0]] + [[.7,.7,0,0]]
radius_times = [[.1,.1,0,0]] + (T//2-1)*[[1,1,max_v,max_v]] + [[.3,.3,max_v,max_v]] + (T//2-1)*[[1,1,max_v,max_v]] + [[.3,.3,.1,.1]]
Poly_x = cart_H_cube(center_times, radius_times)
# define u polytope
u_scale = 0.1
Poly_u = H_cube([0,0], u_scale).cartpower(T+1)

# define w polytope
w_scale = 0.005
Poly_w = H_cube(center_times[0], radius_times[0]).cart( H_cube([0,0,0,0], w_scale).cartpower(T) ).cart( H_cube([0,0], w_scale).cartpower(T+1) ) 

# solve problem
[result, SLS, Lambda, Lambda_u] = optimize(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w)
# check if feasible
assert result != np.inf
# update dependent variables
SLS.calculate_dependent_variables()

# plot trajectories
def plot_trajectory(w, SLS, ax):
    x_traj = np.block([SLS.Phi_xx.value, SLS.Phi_xy.value]) @ w
    x_traj = x_traj.reshape((-1, SLS.nx))
    #print(x_traj)
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(x_traj[:,0], x_traj[:,1], color = color, linewidth=0.6)
    ax.plot(x_traj[0,0], x_traj[0,1], '.', color = color)
    ax.plot(x_traj[-1,0], x_traj[-1,1], '.', color = color)
    return

#with plt.style.context(['science', 'ieee']):
fig1, ax1 = plt.subplots()
title1 = 'Trajectory - ' + key
ax1.set_title(title1)

rect_0 = patches.Rectangle((center_times[0][0] - radius_times[0][0], center_times[0][1] - radius_times[0][1]), 2*radius_times[0][0], 2*radius_times[0][1], linewidth=1, edgecolor='k', facecolor=(255/255,114/255,118/255,0.5))
rect_T_half = patches.Rectangle((center_times[T//2][0] - radius_times[T//2][0], center_times[T//2][1] - radius_times[T//2][1]), 2*radius_times[T//2][0], 2*radius_times[T//2][1], linewidth=1, edgecolor='k', facecolor=(153/255,186/255,221/255,0.5))
rect_T = patches.Rectangle((center_times[T][0] - radius_times[T][0], center_times[T][1] - radius_times[T][1]), 2*radius_times[T][0], 2*radius_times[T][1], linewidth=1, edgecolor='k', facecolor=(153/255,186/255,221/255,0.5))
ax1.add_patch(rect_0)
ax1.add_patch(rect_T_half)
ax1.add_patch(rect_T)
# corners trjactories
sign = [1, -1]
for i in sign:
    for j in sign:
        w = w_scale*np.random.uniform(-1, 1, (T+1)*(SLS.nx+SLS.ny))
        w[0:SLS.nx] = center_times[0]
        w[0:2] += np.array([i*radius_times[0][0],j*radius_times[0][1]])
        plot_trajectory(w, SLS, ax1)
# random sample trajectories
N = 90
for i in range(N):
    w = w_scale*np.random.uniform(-1, 1, (T+1)*(SLS.nx+SLS.ny))
    w[0:SLS.nx] = center_times[0]
    w[0] += np.random.uniform(-radius_times[0][0], radius_times[0][0])
    w[1] += np.random.uniform(-radius_times[0][1], radius_times[0][1])
    plot_trajectory(w, SLS, ax1)
ax1.set_xlim([-1,1])
ax1.set_ylim([-1,1])
plt.grid()
plt.show()

fig2, ax2 = plt.subplots()
# find F
F = SLS.F
#print(F)
epsilon = 10**-5
#sparse = np.bitwise_or((F > epsilon), (F < -epsilon))
#print(sparse)
ax2.spy(F, precision=epsilon)
title2 = 'Sparsiy of F - ' + key
ax2.set_title(title2)
plt.show()

fig3, ax3 = plt.subplots()
# find singular values
[U, S, Vh] = np.linalg.svd(F)
print(S)
ax3.plot(S,'.')
title3 = 'Singular Values of F - ' + key
ax3.set_ylim(bottom=0)
ax3.set_title(title3)
plt.show()

u_traj = np.block([SLS.Phi_ux.value, SLS.Phi_uy.value]) @ w
u_traj = u_traj.reshape((-1,2))
print(u_traj)

fig4, ax4 = plt.subplots()
title4 = '$u_1$ - ' + key
ax4.set_title(title4)
ax4.plot(u_traj[:,0])
ax4.set_ylabel('$u_1(t)$')
ax4.set_xlabel('$t$')
ax4.grid()
plt.show()

fig5, ax5 = plt.subplots()
title5 = '$u_2$ - ' + key
ax5.set_title(title5)
ax5.plot(u_traj[:,1])
ax5.set_ylabel('$u_2(t)$')
ax5.set_xlabel('$t$')
ax5.grid()
plt.show()

if save:
    fig1.savefig('figures/' + ''.join(title1.split()) + '.jpg', dpi=300)
    fig2.savefig('figures/' + ''.join(title2.split()) + '.jpg', dpi=300)
    fig3.savefig('figures/' + ''.join(title3.split()) + '.jpg', dpi=300)
    fig4.savefig('figures/' + ''.join(title4.split()) + '.jpg', dpi=300)
    fig5.savefig('figures/' + ''.join(title5.split()) + '.jpg', dpi=300)

#    return

def run():
    t = ""
    with open('centralized_nuclear_norm_poly_containment.py') as f:
        t = f.read()
    return t


#def unit_test():
#    epsilon = 10 ** -4
#    assert np.all(Lambda.value >=  -epsilon)
#    assert np.all(-epsilon <= (Lambda.value @ Poly_w.H - Poly_x.H @ cp.hstack([SLS.Phi_xx, SLS.Phi_xy]).value))
#    assert np.all(epsilon >= (Lambda.value @ Poly_w.H - Poly_x.H @ cp.hstack([SLS.Phi_xx, SLS.Phi_xy]).value))
#    assert np.all(Lambda.value @ Poly_w.h <= Poly_x.h + epsilon)
#    return
