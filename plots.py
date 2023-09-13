import pickle
import numpy as np
import scipy as sp
from SLSFinite import *
from Polytope import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import math

np.random.seed(1)
T=20
dt = 1
A_0 = np.block([[np.zeros([2,2]), np.eye(2)], [np.zeros([2,2]), np.zeros([2,2])]])
B_0 = np.block([[np.zeros([2,2])],[np.eye(2)]])
A = sp.linalg.expm(A_0*dt)
B = np.sum([np.linalg.matrix_power(A_0*dt,i)/math.factorial(i+1) for i in np.arange(100)], axis=0).dot(B_0)
C = np.block([[np.eye(2), np.zeros([2,2])]])
A_list = (T+1)*[A]; B_list = (T+1)*[B]; C_list = (T+1)*[C]
max_v = 2
max_x0 = 1
max_v0 = 0
box_x = 10
center_times = (T+1)*[[0,0,0,0]]
radius_times = (T+1)*[[box_x, box_x, max_v, max_v]]
box_check = [0,10,20]
center_times[box_check[0]] = [-7,-7,0,0]
radius_times[box_check[0]] = [max_x0,max_x0,max_v0,max_v0]
center_times[box_check[1]] = [7,-7,0,0]
radius_times[box_check[1]] = [2,2,max_v,max_v]
center_times[box_check[2]] = [7,7,0,0]
radius_times[box_check[2]] = [2,2,1,1]
Poly_x = cart_H_cube(center_times, radius_times)
u_scale = 2
Poly_u = H_cube([0,0], u_scale).cartpower(T+1)
wx_scale = 0.05
wxdot_scale = 0.05
v_scale = 0.05
delta = 0.01
RTH_opt_eps = 1e-11
sparse_opt_eps = 1e-10
rank_eps = 1e-7
N=8
Poly_w = H_cube(center_times[0], radius_times[0]).cart( H_cube([0,0,0,0], [wx_scale, wx_scale, wxdot_scale, wxdot_scale]).cartpower(T) ).cart( H_cube([0,0], v_scale).cartpower(T+1) ) 
Poly_w = H_cube(center_times[0], radius_times[0]).cart( H_cube([0,0,0,0], [wx_scale, wx_scale, wxdot_scale, wxdot_scale]).cartpower(T) ).cart( H_cube([0,0], v_scale).cartpower(T+1) ) 

test_feas = True
file = "simulation_results/simulationT20.pickle"
save = True
# solve problem

###########################################################################################################

simulation_data = pickle.load(open(file, "rb"))
optimize_RTH_data = simulation_data['Reweighted Nuclear Norm']
optimize_actuator_data = simulation_data['Reweighted Actuator Norm']
optimize_sensor_data = simulation_data['Reweighted Sensor Norm']

SLS_nuc_list= optimize_RTH_data[1]
SLS_nuc = SLS_nuc_list[-1]
Lambda_nuc = optimize_RTH_data[2]
#Lambda_nuc_reopt = optimize_RTH_output[3]

SLS_sen = optimize_sensor_data[1]
norm_sen_list = optimize_sensor_data[3]
Lambda_sen = optimize_sensor_data[2]
kept_sen = optimize_sensor_data[4]
SLS_sen_reweighted = optimize_sensor_data[5][-1]

SLS_act = optimize_actuator_data[1]
norm_act_list = optimize_actuator_data[3]
Lambda_act = optimize_actuator_data[2]
kept_act = optimize_actuator_data[4]
SLS_act_reweighted = optimize_actuator_data[5][-1]

print('Nuc time:', optimize_RTH_data[-1], 'Sen time:', optimize_sensor_data[-1], 'Act time:', optimize_actuator_data[-1])

### MAKE PLOTS ############################################################################################
textsize=10
# Trajectory plots
def plot_trajectory(w, Phi_row, n_dim, checkpoints, ax):
    traj = Phi_row @ w
    traj = traj.reshape((-1, n_dim))
    #print(x_traj)
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(traj[:,0], traj[:,1], color = color, linewidth=0.5)
    for i in checkpoints:
        ax.plot(traj[i,0], traj[i,1], '.', color = color)
    return

fig1, ax1 = plt.subplots()
colors = [(255/255,114/255,118/255,0.5), (153/255,186/255,221/255,0.5), (204/255,255/255,204/255,0.5)]#, (238/255,0,238/255,0.5), (255/255,255/255,0,0.5), (255/255,114/255,118/255,0.5), (255/255,114/255,118/255,0.5), (255/255,114/255,118/255,0.5)]
counter = 0
for i in box_check:
    rect = patches.Rectangle((center_times[i][0] - radius_times[i][0], center_times[i][1] - radius_times[i][1]), 2*radius_times[i][0], 2*radius_times[i][1], linewidth=1, edgecolor='k', facecolor= colors[counter], label='$\mathcal{X}_{' + str(i) + '}$')
    ax1.add_patch(rect)
    counter += 1
ax1.legend()
sign = [1, -1]

N_corner = 10
for i in sign:
    for j in sign:
        for k in range(N_corner):
            w_corner = np.array([])
            for _ in range(T+1):
                w_corner = np.hstack([w_corner, wx_scale*np.random.choice([-1, 1], SLS_nuc.nx//2), wxdot_scale*np.random.choice([-1, 1], SLS_nuc.nx//2)])
            w_corner = np.hstack([w_corner, v_scale*np.random.choice([-1, 1], (T+1)*(SLS_nuc.ny))])
            w_corner[0:SLS_nuc.nx] = center_times[0]
            w_corner[0:2] += np.array([i*radius_times[0][0],j*radius_times[0][1]])
            plot_trajectory(w_corner, SLS_nuc.Phi_trunc[0:(T+1)*SLS_nuc.nx, 0:(T+1)*(SLS_nuc.nx + SLS_nuc.ny)], SLS_nuc.nx, box_check, ax1)

N_samples = 40
for i in range(N_samples):
    #w = np.concatenate([w_scale*np.random.uniform(-1, 1, (T+1)*(SLS_nuc.nx)), v_scale*np.random.uniform(-1, 1, (T+1)*(SLS_nuc.ny))])
    w = np.array([])
    for _ in range(T+1):
        w = np.hstack([w, wx_scale*np.random.uniform(-1, 1, 2), wxdot_scale*np.random.uniform(-1, 1, SLS_nuc.nx//2)])
    w = np.hstack([w, v_scale*np.random.uniform(-1, 1, (T+1)*(SLS_nuc.ny))])
    w[0:SLS_nuc.nx] = center_times[0]
    w[0] += np.random.uniform(-radius_times[0][0], radius_times[0][0])
    w[1] += np.random.uniform(-radius_times[0][1], radius_times[0][1])
    w[2] += np.random.uniform(-radius_times[0][2], radius_times[0][2])
    w[3] += np.random.uniform(-radius_times[0][3], radius_times[0][3])
    plot_trajectory(w, SLS_nuc.Phi_trunc[0:(T+1)*SLS_nuc.nx, 0:(T+1)*(SLS_nuc.nx + SLS_nuc.ny)], SLS_nuc.nx, box_check, ax1)
ax1.set_xlim([-10,10])
ax1.set_ylim([-10,10])
ax1.locator_params(axis='both', nbins=5)
ax1.tick_params(axis='both', labelsize=textsize)
ax1.grid()
if save:
    fig1.savefig("simulation_results/NuclearNormTrajPlot.pdf", bbox_inches="tight")
#plt.show()


# sparsity plots causal
epsilon = 0
assert np.isclose(np.max(np.abs(SLS_nuc.F_trunc[-2:, :])), 0)
assert np.isclose(np.max(np.abs(SLS_nuc.F_trunc[:, -2:])), 0)
assert np.isclose(np.max(np.abs(SLS_nuc.D[-2:, :])), 0)
assert np.isclose(np.max(np.abs(SLS_nuc.E[:, -2:])), 0)
F = SLS_nuc.F_trunc[0:-2, 0:-2]
D = SLS_nuc.D[0:-2, :]
E = SLS_nuc.E[:, 0:-2]
gs2 = gridspec.GridSpec(2,3, width_ratios=[T*SLS_nuc.nu/E.shape[0],1,T*SLS_nuc.nu/E.shape[0]])
fig2 = plt.figure()
axs20 = plt.subplot(gs2[0,0])
axs21 = plt.subplot(gs2[0,1])
axs22 = plt.subplot(gs2[0,2])
axs20.spy(F, epsilon, markersize=1, color='b', label='$\mathbf{K}$')
axs21.spy(D, epsilon, markersize=1, color='b', label='$\mathbf{D}$')
axs22.spy(E, epsilon, markersize=1, color='b', label='$\mathbf{E}$')
axs20.set_xticks(np.arange(4,T*SLS_nuc.ny,5), np.arange(5,T*SLS_nuc.ny+1,5))
axs20.set_yticks(np.arange(4,T*SLS_nuc.nu,5), np.arange(5,T*SLS_nuc.nu+1,5))
axs20.tick_params(axis='both', labelsize=textsize)
axs21.set_xticks(np.arange(4,D.shape[1],5), np.arange(5,D.shape[1]+1,5))
axs21.set_yticks(np.arange(4,T*SLS_nuc.nu,5), np.arange(5,T*SLS_nuc.nu+1,5))
axs21.tick_params(axis='both', labelsize=textsize)
axs22.set_yticks(np.arange(4,E.shape[0],5), np.arange(5,E.shape[0]+1,5))
axs22.set_xticks(np.arange(4,T*SLS_nuc.ny,5), np.arange(5,T*SLS_nuc.ny+1,5))
axs22.tick_params(axis='both', labelsize=textsize)
axs20.grid()
axs21.grid()
axs22.grid()
axs20.legend(markerscale=0, handlelength=-0.8)
axs21.legend(markerscale=0, handlelength=-0.8)
axs22.legend(markerscale=0, handlelength=-0.8)
#axs22.locator_params(axis='y', nbins=3)
fig2.tight_layout()
if save:
    fig2.savefig("simulation_results/SparsityPlotCausal.pdf", bbox_inches="tight")

# sparsity plots sensor and actuator
assert np.isclose(np.max(np.abs(SLS_sen.F[-2:, :])), 0)
assert np.isclose(np.max(np.abs(SLS_sen.F[:, -2:])), 0)
assert np.isclose(np.max(np.abs(SLS_act.F[-2:, :])), 0)
assert np.isclose(np.max(np.abs(SLS_act.F[:, -2:])), 0)
fig21 = plt.figure()
gs21 = gridspec.GridSpec(2,2)
axs23 = plt.subplot(gs21[0,0])
axs24 = plt.subplot(gs21[0,1])
axs23.spy(SLS_sen.F[0:-2,0:-2], epsilon, markersize=1, color='tab:green', label="$\mathbf{K}$")
axs24.spy(SLS_act.F[0:-2,0:-2], epsilon, markersize=1, color='r', label="$\mathbf{K}$")
axs23.set_xticks(np.arange(4,T*SLS_nuc.ny,5), np.arange(5,T*SLS_nuc.ny+1,5))
axs23.set_yticks(np.arange(4,T*SLS_nuc.nu,5), np.arange(5,T*SLS_nuc.nu+1,5))
axs24.set_xticks(np.arange(4,T*SLS_nuc.ny,5), np.arange(5,T*SLS_nuc.ny+1,5))
axs24.set_yticks(np.arange(4,T*SLS_nuc.nu,5), np.arange(5,T*SLS_nuc.nu+1,5))
axs23.tick_params(axis='both', labelsize=textsize)
axs24.tick_params(axis='both', labelsize=textsize)
axs23.grid()
axs24.grid()
axs23.legend(markerscale=0, handlelength=-0.8)
axs24.legend(markerscale=0, handlelength=-0.8)
fig21.tight_layout()
if save:
    fig21.savefig("simulation_results/SparsityPlotSenAct.pdf", bbox_inches="tight")


# reweighting iteration plots
gs3 = gridspec.GridSpec(4,1, height_ratios=[1,1,1,1])
fig3 = plt.figure()
axs30 = plt.subplot(gs3[0])
for k in range(len(SLS_nuc_list)):
    SLS_nuc_k = SLS_nuc_list[k]
    SLS_nuc_k.calculate_dependent_variables("Reweighted Nuclear Norm")
    [U, S, Vh] = np.linalg.svd(SLS_nuc_k.F)
    if k == len(SLS_nuc_list)-1:
        axs30.plot(np.arange(1,S.size+1), np.log10(S),'.-', label='k = ' + str(k + 1), linewidth=0.5, markersize=6, color='k')
    else:
        axs30.plot(np.arange(1,S.size+1), np.log10(S),'.--', label='k = ' + str(k + 1), linewidth=0.5, markersize=6)
axs30.set_ylim(bottom=-16)
axs30.set_ylabel('$\log_{10}(\sigma_i(\mathbf{K}))$', fontsize=textsize)
axs30.set_xlabel('singular value index $i$', fontsize=textsize)
axs30.tick_params(axis='both', labelsize=textsize)
axs30.tick_params(axis='both', labelsize=textsize)
axs30.locator_params(axis='y', nbins=3)
axs30.grid()
axs30.legend(fontsize=5)

axs31 = plt.subplot(gs3[1])
for k in range(len(norm_act_list)):
    norm_k = norm_act_list[k]
    if k == len(norm_act_list)-1:
        axs31.plot(np.arange(1,norm_k.size+1), np.log10(norm_k),'.-', label='k = ' + str(k + 1), linewidth=0.5, markersize=6, color='k')
    else:
        axs31.plot(np.arange(1,norm_k.size+1), np.log10(norm_k),'.--', label='k = ' + str(k + 1), linewidth=0.5, markersize=6)
axs31.set_ylim(bottom=-16)
axs31.set_ylabel('$\log_{10}(||\mathbf{\Phi}_{uy, (i,:)}||_2)$', fontsize=textsize)
axs31.set_xlabel('row index $i$', fontsize=textsize)
axs31.tick_params(axis='both', labelsize=textsize)
axs31.tick_params(axis='both', labelsize=textsize)
axs31.locator_params(axis='y', nbins=3)
axs31.grid()

axs32 = plt.subplot(gs3[2])
for k in range(len(norm_sen_list)):
    norm_k = norm_sen_list[k]
    if k == len(norm_sen_list)-1:
        axs32.plot(np.arange(1,norm_k.size+1), np.log10(norm_k),'.-', label='k = ' + str(k + 1), linewidth=0.5, markersize=6, color='k')
    else:
        axs32.plot(np.arange(1,norm_k.size+1), np.log10(norm_k),'.--', label='k = ' + str(k + 1), linewidth=0.5, markersize=6)
axs32.set_ylim(bottom=-16)
axs32.set_ylabel('$\log_{10}(||\mathbf{\Phi}_{uy, (:,i)}||_2)$', fontsize=textsize)
axs32.set_xlabel('column index $i$', fontsize=textsize)
axs32.tick_params(axis='both', labelsize=textsize)
axs32.tick_params(axis='both', labelsize=textsize)
axs32.locator_params(axis='y', nbins=3)
axs32.grid()
fig3.tight_layout()
if save:
    fig3.savefig("simulation_results/ReweightingPlots.pdf", bbox_inches="tight")

# final reweighting iteratation plots
gs4 = gridspec.GridSpec(3,1, height_ratios=[1,1,1])
fig4 = plt.figure()
axs40 = plt.subplot(gs4[0])
[U, S, Vh] = np.linalg.svd(SLS_nuc.F)
# calculate the last iteration of norms of K
SLS_sen_reweighted.calculate_dependent_variables("Reweighted Sensor Norm")
SLS_act_reweighted.calculate_dependent_variables("Reweighted Actuator Norm")
sen_norm_F = np.linalg.norm(SLS_sen_reweighted.F, 2, 0)
act_norm_F = np.linalg.norm(SLS_act_reweighted.F, 2, 1)
axs40.semilogy(np.arange(1,S.size+1), S,'s-', label='$\sigma_i(\mathbf{K})$', linewidth=0.5, markersize=5, color='b')
axs40.semilogy(np.arange(1,act_norm_F.size+1), act_norm_F,'o-', label='$||\mathbf{K}_{(i,:)}||_2$', linewidth=0.5, markersize=5, color='r')
axs40.semilogy(np.arange(1,sen_norm_F.size+1), sen_norm_F,'^-', label='$||\mathbf{K}_{(:,i)}||_2$', linewidth=0.5, markersize=5, color='tab:green')
axs40.set_ylim([1e-18,1000])
axs40.set_xlim([0,43])
axs40.set_xlabel('index $i$', fontsize=10)
axs40.tick_params(axis='both', labelsize=textsize)
axs40.tick_params(axis='both', labelsize=textsize)
axs40.grid()
axs40.legend(fontsize=8, loc='lower left')
fig4.tight_layout()
if save:
    fig4.savefig("simulation_results/ReweightingPlotsFinalIteration.pdf", bbox_inches="tight")


Poly_xu = Poly_x.cart(Poly_u)
print("causal_time:", SLS_nuc.causal_time)
print()
print("--- Nuclear Norm -----------------------------------------------------")
print("rank K:", SLS_nuc.rank_F_trunc)
print("band (D,E) = messages:", SLS_nuc.E.shape[0])
print("message times:", SLS_nuc.F_causal_row_basis)

print("max |K - K_trunc|:", np.max( np.abs(SLS_nuc.F - SLS_nuc.F_trunc) ) )
print("max |Phi - Phi_trunc|:", np.max( np.abs(SLS_nuc.Phi_matrix.value - SLS_nuc.Phi_trunc) ) )
print("Error true polytope constraint:", np.max(np.abs( Lambda_nuc.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_nuc.Phi_matrix.value))))
print("Error truncated polytope constraint:", np.max( np.abs(Lambda_nuc.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_nuc.Phi_trunc)) ) )
#print("Error reoptimized polytope constraint:", np.max( np.abs(Lambda_nuc_reopt.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_nuc.Phi_trunc)) ) )
print()
print("--- Sensor Norm ------------------------------------------------------")
print("number of nonzero cols = messages:", len(kept_sen))
print("Error true F and truncated F:", np.max( np.abs(SLS_sen.F - SLS_sen.F_trunc) ) )
print("Error true Phi and truncated Phi:", np.max( np.abs(SLS_sen.Phi_matrix.value - SLS_sen.Phi_trunc) ) )
print("Error truncated polytope constraint:", np.max( np.abs(Lambda_sen.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_sen.Phi_trunc)) ) )
print()
print("--- Actuator Norm ----------------------------------------------------")
print("number of nonzero rows = messages:", len(kept_act))
print("Error true F and truncated F:", np.max( np.abs(SLS_act.F - SLS_act.F_trunc) ) )
print("Error true Phi and truncated Phi:", np.max( np.abs(SLS_act.Phi_matrix.value - SLS_act.Phi_trunc) ) )
print("Error truncated polytope constraint:", np.max( np.abs(Lambda_act.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_act.Phi_trunc)) ) )

plt.show()

def run():
    t = ""
    with open('simulation.py') as f:
        t = f.read()
    return t