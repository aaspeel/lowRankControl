import numpy as np
import scipy as sp
import cvxpy as cp
from scipy.linalg import block_diag
from Polytope import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scienceplots
import math
import copy

def youla_optimization(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w):

    # define constants
    T = len(A_list) - 1
    Tp1 = len(A_list)
    nx = A_list[0].shape[0]
    nu = B_list[0].shape[1]
    ny = C_list[0].shape[0]
    Z = np.block([  [np.zeros([nx,T*nx]), np.zeros([nx,nx])   ],
                    [np.eye(T*nx),        np.zeros([T*nx, nx])]     ])
        # define block-diagonal matrices
    cal_A = block_diag(*A_list)
    cal_B = block_diag(*B_list)
    cal_C = block_diag(*C_list)
    I = np.eye(nx*(Tp1))
    P_xw = np.linalg.inv((I - Z.dot(cal_A)))
    P_xu = P_xw.dot(Z).dot(cal_B)
    P_xv = 0
    P_yw = cal_C.dot(P_xw)
    P_yu = P_yw.dot(Z).dot(cal_B)
    P_yv = np.eye(ny*(Tp1))

    epsilon = 10**-3
    plt.spy(P_yu, precision=epsilon)
    plt.show()


    # youla paramter
    Q = cp.Variable((nu*(Tp1), ny*(Tp1)))
    Phi_1 = cp.bmat([[P_xw + P_xu @ Q @ P_yw, P_xu @ Q @ P_yv]])
    Phi_2 = cp.bmat([[Q @ P_yw, Q @ P_yv]])
    #Poly_xu = Poly_x.cart(Poly_u)
    Lambda_x = cp.Variable((np.shape(Poly_x.H)[0], np.shape(Poly_w.H)[0]))
    Lambda_u = cp.Variable((np.shape(Poly_u.H)[0], np.shape(Poly_w.H)[0]))


    #objective = cp.Minimize(cp.norm(Q, 'nuc'))
    objective = cp.Minimize(1)

    Tp1_upper_tri_bsupp = np.triu(np.ones([Tp1, Tp1]), 1)
    constraints = [cp.multiply( Q, cp.kron(Tp1_upper_tri_bsupp, np.ones((nu, ny))) ) == 0,
                Lambda_x >= 0,
                Lambda_x @ Poly_w.H == Poly_x.H @ Phi_1,
                Lambda_x @ Poly_w.h <= Poly_x.h,
                Lambda_u >= 0,
                Lambda_u @ Poly_w.H == Poly_u.H @ Phi_2,
                Lambda_u @ Poly_w.h <= Poly_u.h]
    
    problem = cp.Problem(objective, constraints)
    result = problem.solve(solver=cp.SCS, warm_start=True, verbose=True)
    F = np.linalg.inv(np.eye(nu*Tp1) + Q.value @ P_yu) @ Q.value
    return result, Q, Phi_1, Phi_2, F

T = 30
dt = 1
A = np.eye(4) + dt*np.block([[np.zeros([2,2]), np.eye(2)], [np.zeros([2,2]), np.zeros([2,2])]])
B = dt*np.block([[np.zeros([2,2])],[np.eye(2)]])
C = np.block([[np.eye(2), np.zeros([2,2])]])
A_list = (T+1)*[A]; B_list = (T+1)*[B]; C_list = (T+1)*[C]
nx = np.shape(A)[0]
nu = np.shape(B)[1]
ny = np.shape(C)[0]
key = 'Youla Parametrization'


max_v = 0.2
max_x0 = 0.1
center_times = [[-.7,-.7,0,0]] + (T//2-1)*[[0,0,0,0]] + [[.7,-.7,0,0]] + (T//2-1)*[[0,0,0,0]] + [[.7,.7,0,0]]
radius_times = [[max_x0,max_x0,0,0]] + (T//2-1)*[[1,1,max_v,max_v]] + [[.3,.3,max_v,max_v]] + (T//2-1)*[[1,1,max_v,max_v]] + [[.3,.3,.1,.1]]
Poly_x = cart_H_cube(center_times, radius_times)
# define u polytope
u_scale = 0.1
Poly_u = H_cube([0,0], u_scale).cartpower(T+1)

w_scale = 0 #2*T/8000
Poly_w = H_cube(center_times[0], radius_times[0]).cart( H_cube([0,0,0,0], w_scale).cartpower(T) ).cart( H_cube([0,0], w_scale).cartpower(T+1) ) 

result, Q, Phi_1, Phi_2, F = youla_optimization(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w)
assert result != np.inf

def plot_trajectory(w, Phi_1, nx, ax):
    x_traj = Phi_1.value @ w
    x_traj = x_traj.reshape((-1, nx))
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
        w = w_scale*np.random.uniform(-1, 1, (T+1)*(nx+ny))
        w[0:nx] = center_times[0]
        w[0:2] += np.array([i*radius_times[0][0],j*radius_times[0][1]])
        plot_trajectory(w, Phi_1, nx, ax1)
# random sample trajectories
N = 90
for i in range(N):
    w = w_scale*np.random.uniform(-1, 1, (T+1)*(nx+ny))
    w[0:nx] = center_times[0]
    w[0] += np.random.uniform(-radius_times[0][0], radius_times[0][0])
    w[1] += np.random.uniform(-radius_times[0][1], radius_times[0][1])
    w[2] += np.random.uniform(-radius_times[0][2], radius_times[0][2])
    w[3] += np.random.uniform(-radius_times[0][3], radius_times[0][3])
    plot_trajectory(w, Phi_1, nx, ax1)
ax1.set_xlim([-1,1])
ax1.set_ylim([-1,1])
plt.grid()
plt.show()

fig2, ax2 = plt.subplots()
epsilon = 10**-5
ax2.spy(F, precision=epsilon)
title2 = 'Sparsiy of F - ' + key
ax2.set_title(title2)
plt.show()

fig3, ax3 = plt.subplots()
#for k in range(len(SLS_list)):
#    SLS_k = SLS_list[k]
#    SLS_k.calculate_dependent_variables()
#    [U, S, Vh] = np.linalg.svd(SLS_k.F)
#    print(S)
#    ax3.plot(np.log10(S),'.-', label='k = ' + str(k + 1))
#
[_, S, __] = np.linalg.svd(F)
ax3.plot(np.log10(S),'.-')#, label='k = ' + str(k + 1))
title3 = 'Singular Values of F - ' + key
ax3.set_ylim(bottom=0)
ax3.set_ylabel('$\log_{10}(\sigma_i)$')
ax3.set_xlabel('$i$')
ax3.set_title(title3)
ax3.grid()
ax3.legend()
plt.show()

u_traj = Phi_2.value @ w
u_traj = u_traj.reshape((-1,2))

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

def run():
    t = ""
    with open('centralized_nuclear_norm_poly_containment_youla.py') as f:
        t = f.read()
    return t
