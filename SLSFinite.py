import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag

class SLSFinite():
    def __init__(self, A_list, B_list, C_list):
        """
        Store the variables used for convex optimization in finite time system level synthesis framework.
    
        Parameters
        ----------
        A_list: list of matrices [A_0, ...A_T]
        B_list: list of matrices [B_0, ...B_T]
        C_list: list of matrices [C_0, ...C_T]
            where A_t, B_t, C_t are the matrices in the dynamics of the system at time t.
        
        Attributes
        ----------
        Phi_xx: cvxpy.Variable, shape ((T+1)*nx, (T+1)*nx)
        Phi_xy: cvxpy.Variable, shape ((T+1)*nx, (T+1)*ny)
        Phi_ux: cvxpy.Variable, shape ((T+1)*nu, (T+1)*nx)
        Phi_uy: cvxpy.Variable, shape ((T+1)*nu, (T+1)*ny)
        """
        # init variables
        assert len(A_list) == len(B_list) == len(C_list)
        # define dimanesions
        self.T = len(A_list) - 1
        self.nx = A_list[0].shape[0]
        self.nu = B_list[0].shape[1]
        self.ny = C_list[0].shape[0]
        # define optimization variables
        self.Phi_xx = cp.Variable( ((self.T+1)*self.nx, (self.T+1)*self.nx) )
        self.Phi_xy = cp.Variable( ((self.T+1)*self.nx, (self.T+1)*self.ny) )
        self.Phi_ux = cp.Variable( ((self.T+1)*self.nu, (self.T+1)*self.nx) )
        self.Phi_uy = cp.Variable( ((self.T+1)*self.nu, (self.T+1)*self.ny) )
        self.Phi_matrix = cp.bmat( [[self.Phi_xx,   self.Phi_xy], 
                                    [self.Phi_ux,   self.Phi_uy]] )
        # define downshift operator
        self.Z = np.block([
                            [np.zeros([self.nx,self.T*self.nx]),    np.zeros([self.nx,self.nx])        ],
                            [np.eye(self.T*self.nx),                np.zeros([self.T*self.nx, self.nx])]
                            ])
        # define block-diagonal matrices
        self.cal_A = block_diag(*A_list)
        self.cal_B = block_diag(*B_list)
        self.cal_C = block_diag(*C_list)

        assert self.Z.shape == self.cal_A.shape
        assert self.Z.shape[0] == self.cal_B.shape[0]
        assert self.Z.shape[0] == self.cal_C.shape[1]

        # dependent variables
        self.F = None
        self.Phi_yx = None
        self.Phi_yy = None
        

    def SLP_constraints(self):
        """
        Compute the system level parametrization constraints used in finite time system level synthesis.

        Return
        ------
        SLP: list of 6 cvxpy.Constraint objects
            These are constraints on the Phi variables consisting of
            1. 2 affine system level parametrization constraints and
            2. 4 lower block triangular constraints.
        """
        Tp1 = self.T + 1
        I = np.eye(Tp1*self.nx)

        # define affine SLS constraint
        SLP = [cp.bmat([[I - self.Z @ self.cal_A, -self.Z @ self.cal_B]]) @ self.Phi_matrix == cp.bmat([[I, np.zeros( (Tp1*self.nx, Tp1*self.ny) )]]),
                self.Phi_matrix @ cp.bmat([[I - self.Z @ self.cal_A], [-self.cal_C]]) == cp.bmat([[I], [np.zeros( (Tp1*self.nu, Tp1*self.nx) )]])]
        
        # define lower block triangular constraints
        low_tri_bsupp = np.tril(np.ones([Tp1,Tp1]))
        SLP += [self.Phi_xx == cp.multiply( self.Phi_xx, cp.kron(low_tri_bsupp, np.ones((self.nx, self.nx))) ),
                self.Phi_xy == cp.multiply( self.Phi_xy, cp.kron(low_tri_bsupp, np.ones((self.nx, self.ny))) ),
                self.Phi_ux == cp.multiply( self.Phi_ux, cp.kron(low_tri_bsupp, np.ones((self.nu, self.nx))) ),
                self.Phi_uy == cp.multiply( self.Phi_uy, cp.kron(low_tri_bsupp, np.ones((self.nu, self.ny))) )]
        return SLP

    def calculate_dependent_variables(self):
        """
        Compute the controller F, Phi_yx and Phi_yy from the independent varialbe Phi_matrix.
        """
        self.F = self.Phi_uy.value - self.Phi_ux.value @ np.linalg.inv(self.Phi_xx.value) @ self.Phi_xy.value
        self.Phi_yx = self.cal_C @ self.Phi_xx
        self.Phi_yy = self.cal_C @ self.Phi_xy + np.eye(self.cal_C.shape[0])
        return
