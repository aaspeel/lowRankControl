import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag
from SLSFinite import *
import matplotlib.pyplot as plt

def optimize(A_list, B_list, C_list, H_w, h_w, H_x, h_x):
    SLS_data = SLSFinite(A_list, B_list, C_list)
    constraints = SLS_data.SLP_constraints()
    # polytope containment
    Lambda = cp.Variable((np.shape(H_x)[0], np.shape(H_w)[0]))
    constraints += [Lambda >= 0,
                    Lambda @ H_w == H_x @ cp.hstack([SLS_data.Phi_xx, SLS_data.Phi_xy]),
                    Lambda @ h_w <= h_x]
    
    # objective function
    objective = cp.Minimize(cp.norm(SLS_data.Phi_uy, 'nuc'))
    #objective = cp.Minimize(1)
    problem = cp.Problem(objective, constraints)
    result = problem.solve()
    return result, SLS_data, Lambda


T = 30; nx = 2; ny = 1; nu = 2
scale = 0.1
A = np.diag([1.1, 0.9]); B = np.eye(2); C = np.random.randn(ny, nx)
A_list = (T+1)*[A]; B_list = (T+1)*[B]; C_list = (T+1)*[C]
H_wx = np.array([[1, 0],
                 [-1, 0],
                 [0, 1],
                 [0, -1]])
H_wy = np.array([[1],
                 [-1]])
H_w_list = (T+1)*[H_wx] + (T+1)*[H_wy]
H_x_list = (T+1)*[H_wx]
H_w = block_diag(*H_w_list)
H_x = block_diag(*H_x_list)

h_w = scale*np.array(np.ones((H_w.shape[0], 1)))
h_w[0:4,0] = 0.5
#h_w[0,0] = -0.5
#h_w[1:4,0] = 1
#print(h_w)

h_x = np.ones((H_x.shape[0], 1))
h_x[0:4,0] = h_w[0:4,0]
#h_x[-3,0] = -0.5
#print(h_x)

[result, SLS_data, Lambda] = optimize(A_list, B_list, C_list, H_w, h_w, H_x, h_x)
assert result != np.inf
#print(result)
F = SLS_data.optimal_controller()
#print(F)
[U, S, Vh] = np.linalg.svd(F)
#print(S)

# sampling
w = scale*np.random.uniform(-1, 1, (T+1)*(nx+ny))
#prinSLS_data.Lambda)
#print(SLS_data.Lambda @ H_w)
#print(SLS_data.Lambda @ H_w)
#print([SLS_data.Phi_xx, SLS_data.Phi_xy])
#print()
#w[0:2] = np.array([-0.95, 0])

w[0:2] = 0.5
#print(w)
#print(np.block([SLS_data.Phi_xx.value, SLS_data.Phi_xy.value]).shape)
x_traj = np.block([SLS_data.Phi_xx.value, SLS_data.Phi_xy.value]) @ w
x_traj = x_traj.reshape((-1, 2))
#times = np.arange(0, T+1)
print(x_traj)
plt.plot(x_traj[0:-1,0], x_traj[0:-1,1])
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.show()

# debug
epsilon = 10 ** -4
assert np.all(Lambda.value >=  -epsilon)
assert np.all(-epsilon <= (Lambda.value @ H_w - H_x @ cp.hstack([SLS_data.Phi_xx, SLS_data.Phi_xy]).value))
assert np.all(epsilon >= (Lambda.value @ H_w - H_x @ cp.hstack([SLS_data.Phi_xx, SLS_data.Phi_xy]).value))
assert np.all(Lambda.value @ h_w <= h_x + epsilon)
#assert SLP_data.Phi_xx == cp.multiply( self.Phi_xx, cp.kron(low_tri_bsupp, np.ones((self.nx, self.nx))) ),
#                self.Phi_xy == cp.multiply( self.Phi_xy, cp.kron(low_tri_bsupp, np.ones((self.nx, self.ny))) ),
#                self.Phi_ux == cp.multiply( self.Phi_ux, cp.kron(low_tri_bsupp, np.ones((self.nu, self.nx))) ),
#                self.Phi_uy == cp.multiply( self.Phi_uy, cp.kron(low_tri_bsupp, np.ones((self.nu, self.ny))) )]


def run():
    t = ""
    with open('centralized_nuclear_norm_poly_containment.py') as f:
        t = f.read()
    return t