import pickle
import numpy as np
import scipy as sp
from SLSFinite_archived import *
from Polytope import *
from functions_archived import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import math
import time

### PARAMETER SELECTION #############################################################################################################
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
opt_eps = 1e-11
rank_eps = 1e-7
N=10
Poly_w = H_cube(center_times[0], radius_times[0]).cart( H_cube([0,0,0,0], [wx_scale, wx_scale, wxdot_scale, wxdot_scale]).cartpower(T) ).cart( H_cube([0,0], v_scale).cartpower(T+1) ) 
Poly_w = H_cube(center_times[0], radius_times[0]).cart( H_cube([0,0,0,0], [wx_scale, wx_scale, wxdot_scale, wxdot_scale]).cartpower(T) ).cart( H_cube([0,0], v_scale).cartpower(T+1) ) 

test_feas = True
file = "simulationT20_0.pickle"
# solve problem

### SIMULATION #########################################################################################################
if test_feas:
    _ = optimize(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, 'Feasibility', opt_eps)

key = 'Reweighted Nuclear Norm'
start = time.time()
# solve optimization problem
optimize_RTH_output = optimize_RTH(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, N, delta, rank_eps, opt_eps, 'truncate_F')
#optimize_RTH_output = optimize_RTH(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, N, delta, rank_eps, opt_eps, 'truncate_F_svd')
t1 = time.time() - start
optimize_RTH_output.append(t1)

key = 'Reweighted Actuator Norm'
start = time.time()
optmize_sparsity_actuator_output = optimize_sparsity(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, key, N, delta, rank_eps, 1e-8)
t2 = time.time() - start
optmize_sparsity_actuator_output.append(t2)

key = 'Reweighted Sensor Norm'
start = time.time()
optmize_sparsity_sensor_output = optimize_sparsity(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, key, N, delta, rank_eps, 1e-8)
t3 = time.time() - start
optmize_sparsity_sensor_output.append(t3)

data = {
'Reweighted Nuclear Norm': optimize_RTH_output,
'Reweighted Actuator Norm': optmize_sparsity_actuator_output,
'Reweighted Sensor Norm': optmize_sparsity_sensor_output
}

with open(file,"wb") as f:
    pickle.dump(data, f)

def load_file(filename):
    objects = []
    with (open(filename, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    return objects
    
simulation_data = pickle.load(open(file, "rb"))
optimze_RTH_data = simulation_data['Reweighted Nuclear Norm']
optimze_actuator_data = simulation_data['Reweighted Actuator Norm']
optimze_sensor_data = simulation_data['Reweighted Sensor Norm']
#[result_list, SLS_data_list, Lambda, problem.status, constraints]
#[[reopt_result], [reopt_SLS], norm_list, re_opt_ind, Lambda, constraints]
SLS_nuc_list= optimze_RTH_data[1]
SLS_nuc = SLS_nuc_list[-1]
SLS_act_list = optimze_actuator_data[1]
norm_act_list = optimze_actuator_data[2]
SLS_act = SLS_act_list[-1]
SLS_sen_list = optimze_sensor_data[1]
norm_sen_list = optimze_sensor_data[2]
SLS_sen = SLS_sen_list[-1]
print('Nuc time:', optimze_RTH_data[-1], 'Sen time:', optimze_sensor_data[-1], 'Act time:', optimze_actuator_data[-1])

###########################################################################################################

# plot trajectories
textsize=8
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
ax1.grid()
fig1.savefig("NuclearNormTrajPlot.pdf", bbox_inches="tight")
#plt.show()


# find F
#F = SLS_nuc.F
epsilon = 10**-7
F_remove = SLS_nuc.F_trunc#[0:-2, 0:-2]
D = SLS_nuc.D
E = SLS_nuc.E
gs2 = gridspec.GridSpec(2,3, width_ratios=[T*SLS_nuc.nu/E.shape[0],1,T*SLS_nuc.nu/E.shape[0]])
fig2 = plt.figure()
axs20 = plt.subplot(gs2[0,0])
axs21 = plt.subplot(gs2[0,1])
axs22 = plt.subplot(gs2[0,2])
axs20.spy(F_remove, epsilon, markersize=1, color='b')
axs21.spy(D, epsilon, markersize=1, color='b')
axs22.spy(E, epsilon, markersize=1, color='b')
axs20.set_xticks(np.arange(4,T*SLS_nuc.ny,5), np.arange(5,T*SLS_nuc.ny+1,5))
axs20.set_yticks(np.arange(4,T*SLS_nuc.nu,5), np.arange(5,T*SLS_nuc.nu+1,5))
axs20.tick_params(axis='both', labelsize=textsize)
#axs20.locator_params(axis='x', nbins=9)
axs21.set_xticks(np.arange(4,D.shape[1],5), np.arange(5,D.shape[1]+1,5))
axs21.set_yticks(np.arange(4,T*SLS_nuc.nu,5), np.arange(5,T*SLS_nuc.nu+1,5))
axs21.tick_params(axis='both', labelsize=textsize)
axs22.set_yticks(np.arange(4,E.shape[0],5), np.arange(5,E.shape[0]+1,5))
axs22.set_xticks(np.arange(4,T*SLS_nuc.ny,5), np.arange(5,T*SLS_nuc.ny+1,5))
axs22.tick_params(axis='both', labelsize=textsize)
#axs22.locator_params(axis='y', nbins=3)
fig2.tight_layout()
fig2.savefig("SparsityPlotCausal.pdf", bbox_inches="tight")


fig21 = plt.figure()
gs21 = gridspec.GridSpec(2,2)
axs24 = plt.subplot(gs21[0,0])
axs23 = plt.subplot(gs21[0,1])
axs23.spy(SLS_sen.Phi_uy.value[0:-2,0:-2], epsilon, markersize=1, color='tab:green')
axs24.spy(SLS_act.Phi_uy.value[0:-2,0:-2], epsilon, markersize=1, color='r')
axs23.set_xticks(np.arange(4,T*SLS_nuc.ny,5), np.arange(5,T*SLS_nuc.ny+1,5))
axs23.set_yticks(np.arange(4,T*SLS_nuc.nu,5), np.arange(5,T*SLS_nuc.nu+1,5))
axs24.set_xticks(np.arange(4,T*SLS_nuc.ny,5), np.arange(5,T*SLS_nuc.ny+1,5))
axs24.set_yticks(np.arange(4,T*SLS_nuc.nu,5), np.arange(5,T*SLS_nuc.nu+1,5))
axs23.tick_params(axis='both', labelsize=textsize)
axs24.tick_params(axis='both', labelsize=textsize)
fig21.tight_layout()
fig21.savefig("SparsityPlotSenAct.pdf", bbox_inches="tight")



gs3 = gridspec.GridSpec(4,1, height_ratios=[1,1,1,1])
fig3 = plt.figure()
axs30 = plt.subplot(gs3[0])
for k in range(len(SLS_nuc_list)):
    SLS_nuc_k = SLS_nuc_list[k]
    SLS_nuc_k.calculate_dependent_variables()
    [U, S, Vh] = np.linalg.svd(SLS_nuc_k.F)
    if k == len(SLS_nuc_list)-1:
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
for k in range(len(norm_act_list)):
    norm_k = norm_act_list[k]
    if k == len(norm_act_list)-1:
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

axs32 = plt.subplot(gs3[2])
for k in range(len(norm_sen_list)):
    norm_k = norm_sen_list[k]
    if k == len(norm_sen_list)-1:
        axs32.plot(np.arange(1,norm_k.size+1), np.log10(norm_k),'.-', label='k = ' + str(k + 1), linewidth=0.5, markersize=6, color='k')
    else:
        axs32.plot(np.arange(1,norm_k.size+1), np.log10(norm_k),'.--', label='k = ' + str(k + 1), linewidth=0.5, markersize=6)
axs32.set_ylim(bottom=-12)
axs32.set_ylabel('$\log_{10}(||\mathbf{K}_{(:,i)}||_2)$', fontsize=textsize)
axs32.set_xlabel('column index $i$', fontsize=textsize)
axs32.tick_params(axis='both', labelsize=textsize)
axs32.tick_params(axis='both', labelsize=textsize)
axs32.locator_params(axis='y', nbins=3)
axs32.grid()
fig3.tight_layout()
fig3.savefig("ReweightingPlots.pdf", bbox_inches="tight")


gs4 = gridspec.GridSpec(2,1, height_ratios=[1,1])
fig4 = plt.figure()
axs40 = plt.subplot(gs4[0])
SLS_nuc_k = SLS_nuc_list[-1]
act_norm_k = norm_act_list[-1]
sen_norm_k = norm_sen_list[-1]
SLS_nuc_k.calculate_dependent_variables()
[U, S, Vh] = np.linalg.svd(SLS_nuc_k.F)
# axs40.plot(np.arange(1,S.size+1), np.log10(S),'s-', label='$\log_{10}\sigma_i(\mathbf{K})$', linewidth=0.5, markersize=5, color='b')
# axs40.plot(np.arange(1,act_norm_k.size+1), np.log10(act_norm_k),'o-', label='$\log_{10}||\mathbf{K}_{(i,:)}||_2$', linewidth=0.5, markersize=5, color='r')
# axs40.plot(np.arange(1,sen_norm_k.size+1), np.log10(sen_norm_k),'^-', label='$\log_{10}||\mathbf{K}_{(:,i)}||_2$', linewidth=0.5, markersize=5, color='tab:green')
axs40.semilogy(np.arange(1,S.size+1), S,'s-', label='$\sigma_i(\mathbf{K})$', linewidth=0.5, markersize=5, color='b')
axs40.semilogy(np.arange(1,act_norm_k.size+1), act_norm_k,'o-', label='$||\mathbf{K}_{(i,:)}||_2$', linewidth=0.5, markersize=5, color='r')
axs40.semilogy(np.arange(1,sen_norm_k.size+1), sen_norm_k,'^-', label='$||\mathbf{K}_{(:,i)}||_2$', linewidth=0.5, markersize=5, color='tab:green')
axs40.set_ylim([1e-32,10000])
axs40.set_xlim([0,43])
axs40.set_xlabel('index $i$', fontsize=10)
axs40.tick_params(axis='both', labelsize=textsize)
axs40.tick_params(axis='both', labelsize=textsize)
axs40.grid()
axs40.legend(fontsize=8, loc='lower center')
fig4.tight_layout()
fig4.savefig("ReweightingPlotsFinalIteration.pdf", bbox_inches="tight")
plt.show()

Lambda = optimize_RTH_output[2]
if True:
    Poly_xu = Poly_x.cart(Poly_u)
    print("Error true F and truncated F:", np.max( np.abs(SLS_nuc.F - SLS_nuc.F_trunc) ) )
    print("Error true Phi and truncated Phi:", np.max( np.abs(SLS_nuc.Phi_matrix.value - SLS_nuc.Phi_trunc) ) )
    print("Error truncated polytope constraint:", np.max( np.abs(Lambda.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_nuc.Phi_trunc)) ) )
    print("Error Phi with SLP inverse:", np.max( np.abs(SLS_nuc.Phi_matrix.value - SLS_nuc.F_to_Phi(SLS_nuc.F)) ) ) 

def run():
    t = ""
    with open('simulation.py') as f:
        t = f.read()
    return t