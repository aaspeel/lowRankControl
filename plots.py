import pickle
import numpy as np
import scipy as sp
import cvxpy as cp
from scipy.linalg import block_diag
from SLSFinite import *
from Polytope import *
from centralized_nuclear_norm_poly_containment import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import scienceplots
import math
import copy

np.random.seed(1)
T=20
max_v = 2
max_x0 = 1
max_v0 = 0
box_x = 15
center_times = [[-7,-7,0,0]] +                   (T//2-1)*[[0,0,0,0]] +                 [[6,-6,0,0]] +        (T//2-1)*[[0,0,0,0]] +                 [[6,6,0,0]]
radius_times = [[max_x0,max_x0,max_v0,max_v0]] + (T//2-1)*[[box_x,box_x,max_v,max_v]] + [[3,3,max_v,max_v]] + (T//2-1)*[[box_x,box_x,max_v,max_v]] + [[3,3,1,1]]
Poly_x = cart_H_cube(center_times, radius_times)
u_scale = 2
Poly_u = H_cube([0,0], u_scale).cartpower(T+1)
w_scale = 0.07
v_scale = 0.07
Poly_w = H_cube(center_times[0], radius_times[0]).cart( H_cube([0,0,0,0], w_scale).cartpower(T) ).cart( H_cube([0,0], v_scale).cartpower(T+1) ) 

key = 'Reweighted Nuclear Norm'
save = True

def load_file(filename):
    objects = []
    with (open(filename, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    return objects

objects_nuclear = load_file("data_reweighted_nuclear_norm.pickle")
objects_actuator = load_file("data_reweighted_actuator_norm.pickle")

### MAKE PLOTS ###
# 1 Plot of trajectory actuator (square)
# 2 Plot of u for the actuator and sensor but different (2 above each other)
# 3 Plot of singular values and actuator norm reduction 2 rows
# 4 Plot sparsity, maybe sparsity and columns in one picture and represent factorization

# save data
SLS_nuc_list= objects_nuclear[0]
SLS_nuc = SLS_nuc_list[-1]
SLS_act_list = objects_actuator[0]
norm_list = objects_actuator[1]
SLS_act = SLS_act_list[-1]
textsize = 8


###########################################################################################################

# update dependent variables
SLS_nuc.calculate_dependent_variables()
SLS_act.calculate_dependent_variables()

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

def plot_control(w, SLS_nuc, axs):
    u_traj = np.block([SLS_nuc.Phi_ux.value, SLS_nuc.Phi_uy.value]) @ w
    u_traj = u_traj.reshape((-1,SLS_nuc.nu))
    color = next(axs[0]._get_lines.prop_cycler)['color']
    axs[0].plot(u_traj[:,0], color = color, linewidth=0.6)
    axs[1].plot(u_traj[:,1], color = color, linewidth=0.6)
    return

#with plt.style.context(['science', 'ieee']):
fig1, ax1 = plt.subplots()
rect_0 = patches.Rectangle((center_times[0][0] - radius_times[0][0], center_times[0][1] - radius_times[0][1]), 2*radius_times[0][0], 2*radius_times[0][1], linewidth=1, edgecolor='k', facecolor=(255/255,114/255,118/255,0.5), label='$\mathcal{X}_0$')
rect_T_half = patches.Rectangle((center_times[T//2][0] - radius_times[T//2][0], center_times[T//2][1] - radius_times[T//2][1]), 2*radius_times[T//2][0], 2*radius_times[T//2][1], linewidth=1, edgecolor='k', facecolor=(153/255,186/255,221/255,0.5), label='$\mathcal{X}_{10}$')
rect_T = patches.Rectangle((center_times[T][0] - radius_times[T][0], center_times[T][1] - radius_times[T][1]), 2*radius_times[T][0], 2*radius_times[T][1], linewidth=1, edgecolor='k', facecolor=(204/255,255/255,204/255,0.5), label='$\mathcal{X}_{20}$')
ax1.add_patch(rect_0)
ax1.add_patch(rect_T_half)
ax1.add_patch(rect_T)
ax1.legend()
sign = [1, -1]

N = 40
for i in sign:
    for j in sign:
        for k in range(N//4):
            w_corn = w_scale*np.round(np.random.random((T+1)*(SLS_nuc.nx+SLS_nuc.ny)))
            w_corn[0:SLS_nuc.nx] = center_times[0]
            w_corn[0:2] += np.array([i*radius_times[0][0],j*radius_times[0][1]])
            plot_trajectory(w_corn, np.block([SLS_nuc.Phi_xx.value, SLS_nuc.Phi_xy.value]), SLS_nuc.nx, ax1)

u_trajs = np.zeros([N, SLS_nuc.nu*(SLS_nuc.T+1)])
for i in range(N):
    w = np.concatenate([w_scale*np.random.uniform(-1, 1, (T+1)*(SLS_nuc.nx)), v_scale*np.random.uniform(-1, 1, (T+1)*(SLS_nuc.ny))])
    w[0:SLS_nuc.nx] = center_times[0]
    w[0] += np.random.uniform(-radius_times[0][0], radius_times[0][0])
    w[1] += np.random.uniform(-radius_times[0][1], radius_times[0][1])
    w[2] += np.random.uniform(-radius_times[0][2], radius_times[0][2])
    w[3] += np.random.uniform(-radius_times[0][3], radius_times[0][3])
    #plot traj
    plot_trajectory(w, np.block([SLS_nuc.Phi_xx.value, SLS_nuc.Phi_xy.value]), SLS_nuc.nx, ax1)
    #u statistics
    u_trajs[i,:] = np.block([SLS_nuc.Phi_ux.value, SLS_nuc.Phi_uy.value]) @ w

ax1.set_xlim([-10,11])
ax1.set_ylim([-10,10])
ax1.grid()
fig1.savefig("NuclearNormTrajPlot.pdf", bbox_inches="tight")
plt.show()

gs2 = gridspec.GridSpec(1,4, width_ratios=[5,1,5,5])
fig2 = plt.figure()
axs20 = plt.subplot(gs2[0])
axs21 = plt.subplot(gs2[1])
axs22 = plt.subplot(gs2[2])
axs23 = plt.subplot(gs2[3])
# find F
#F = SLS_nuc.F
epsilon = 10**-4
D, E, rank_F = causal_rank_decomposition(SLS_nuc.F, SLS_nuc.nu, SLS_nuc.ny, SLS_nuc.T)
axs20.spy(SLS_nuc.F, epsilon, markersize=0.5, color='b')
#axs20.tick_params(axis='both', labelsize=5)
axs21.spy(D, epsilon, markersize=0.5, color='b')
axs22.spy(E, epsilon, markersize=0.5, color='b')
axs23.spy(SLS_act.F, epsilon, markersize=0.5, color='r')

axs20.tick_params(axis='both', labelsize=textsize)
axs21.tick_params(axis='both', labelsize=textsize)
axs21.locator_params(axis='x', nbins=2)
axs22.tick_params(axis='both', labelsize=textsize)
axs22.locator_params(axis='y', nbins=2)
axs23.tick_params(axis='both', labelsize=textsize)
fig2.tight_layout()
fig2.savefig("SparsityPlot.pdf", bbox_inches="tight")
plt.show()

gs3 = gridspec.GridSpec(4,1, height_ratios=[1,1,1,1])
fig3 = plt.figure()
axs30 = plt.subplot(gs3[0])
for k in range(len(SLS_nuc_list)):
    SLS_nuc_k = SLS_nuc_list[k]
    SLS_nuc_k.calculate_dependent_variables()
    [U, S, Vh] = np.linalg.svd(SLS_nuc_k.F)
    if k == len(norm_list)-1:
        axs30.plot(np.arange(1,S.size+1), np.log10(S),'.-', label='k = ' + str(k + 1), linewidth=0.5, markersize=6, color='k')
    else:
        axs30.plot(np.arange(1,S.size+1), np.log10(S),'.--', label='k = ' + str(k + 1), linewidth=0.5, markersize=6)
axs30.set_ylim(bottom=-12)
axs30.set_ylabel('$\log_{10}(\sigma_i(\mathbf{K}))$', fontsize=textsize)
axs30.set_xlabel('singular value index $i$', fontsize=textsize)
axs30.tick_params(axis='both', labelsize=textsize)
axs30.tick_params(axis='both', labelsize=textsize)
axs30.locator_params(axis='y', nbins=3)
axs30.grid()
axs30.legend(fontsize=5)

axs31 = plt.subplot(gs3[1])
for k in range(len(norm_list)):
    norm_k = norm_list[k]
    if k == len(norm_list)-1:
        axs31.plot(np.arange(1,norm_k.size+1), np.log10(norm_k),'.-', label='k = ' + str(k + 1), linewidth=0.5, markersize=6, color='k')
    else:
        axs31.plot(np.arange(1,norm_k.size+1), np.log10(norm_k),'.--', label='k = ' + str(k + 1), linewidth=0.5, markersize=6)
axs31.set_ylim(bottom=-12)
axs31.set_ylabel('$\log_{10}(||\mathbf{K}_{(i,:)}||_2)$', fontsize=textsize)
axs31.set_xlabel('row index $i$', fontsize=textsize)
axs31.tick_params(axis='both', labelsize=textsize)
axs31.tick_params(axis='both', labelsize=textsize)
axs31.locator_params(axis='y', nbins=3)
axs31.grid()
fig3.tight_layout()
fig3.savefig("ReweightingPlots.pdf", bbox_inches="tight")
#axs31.legend(fontsize=6)
plt.show()


fig4, axs = plt.subplots(2)
u_trajs = u_trajs.reshape((N,-1,SLS_nuc.nu))
u_trajs_mean = np.mean(u_trajs,axis=0)
u_trajs_max = np.max(u_trajs - u_trajs_mean,axis=0)
u_trajs_min = np.max(u_trajs_mean - u_trajs,axis=0)
u_trajs_std = np.std(u_trajs, axis=0)
times = np.arange(SLS_nuc.T+1)
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




# check feasibility
#Poly_xu = Poly_x.cart(Poly_u)
# original solution
#assert np.all( np.isclose( (Lambda.value.dot(Poly_w.H)).astype('float'), (Poly_xu.H.dot(SLS_nuc.Phi_matrix.value)).astype('float') , atol = 1e-4) )
#assert np.all( (Lambda.value.dot(Poly_w.h)).astype('float') <= (Poly_xu.h).astype('float') + 1e-4 )

# truncated solution
#if key == 'Reweighted Nuclear Norm':
#    assert np.all( np.isclose( (Lambda.value.dot(Poly_w.H)).astype('float'), (Poly_xu.H.dot(Phi_trunc)).astype('float') , atol = 1e-2) )

#    return

def run():
    t = ""
    with open('centralized_nuclear_norm_poly_containment.py') as f:
        t = f.read()
    return t