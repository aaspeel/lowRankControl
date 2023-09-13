import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag
import time

def low_block_tri_variable(n, m, Tp1):
    var = (n*Tp1)*[None]
    for t in range(Tp1):
        for i in range(n):
            add_var = cp.Variable((1, m*(t+1)))
            if t == 0 and i == 0:
                var[0] = cp.hstack([add_var, np.zeros((1, m*(Tp1-(t+1))))])
            elif t == Tp1-1:
                var[t*n+i] = add_var
            else:
                var[t*n+i] = cp.hstack([add_var, np.zeros((1, m*(Tp1-(t+1))))])
    var = cp.vstack(var)
    assert var.shape == (n*Tp1, m*Tp1)
    return var

def row_sparse_low_block_tri_variable(n, m, Tp1, rem_row):
    var = (n*Tp1)*[None]
    for t in range(Tp1):
        for i in range(n):
            if t*n + i in rem_row:
                add_var = np.zeros((1, m*(t+1)))
            else:
                add_var = cp.Variable((1, m*(t+1)))
            if t == 0 and i == 0:
                var[0] = cp.hstack([add_var, np.zeros((1, m*(Tp1-(t+1))))])
            elif t == Tp1-1:
                var[t*n+i] = add_var
            else:
                var[t*n+i] = cp.hstack([add_var, np.zeros((1, m*(Tp1-(t+1))))])
    var = cp.vstack(var)
    assert var.shape == (n*Tp1, m*Tp1)
    return var

def col_sparse_low_block_tri_variable(n, m, Tp1, rem_col):
    var = (m*Tp1)*[None]
    for t in range(Tp1):
        for i in range(m):
            if t*m + i in rem_col:
                add_var = np.zeros((1, n*(Tp1-t)))
            else:
                add_var = cp.Variable((1, n*(Tp1-t)))
            if t == 0 and i == 0:
                var[0] = add_var
            elif t == 0:
                var[t*m+i] = add_var
            else:
                var[t*m+i] = cp.hstack([np.zeros((1, n*t)), add_var])
    var = cp.vstack(var)
    assert var.T.shape == (n*Tp1, m*Tp1)
    return var.T

class SLSFinite():
    def __init__(self, A_list, B_list, C_list, norm=None):
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
        self.Phi_xx = low_block_tri_variable(self.nx, self.nx, self.T+1)
        if norm == None:
            self.Phi_uy = low_block_tri_variable(self.nu, self.ny, self.T+1)
            self.Phi_ux = low_block_tri_variable(self.nu, self.nx, self.T+1)
            self.Phi_xy = low_block_tri_variable(self.nx, self.ny, self.T+1)
        else:
            if norm[0] == 'Reweighted Actuator Norm':
                self.Phi_uy = row_sparse_low_block_tri_variable(self.nu, self.ny, self.T+1, norm[1])
                self.Phi_ux = row_sparse_low_block_tri_variable(self.nu, self.nx, self.T+1, norm[1])
                self.Phi_xy = low_block_tri_variable(self.nx, self.ny, self.T+1)
            elif norm[0] == 'Reweighted Sensor Norm':
                self.Phi_uy = col_sparse_low_block_tri_variable(self.nu, self.ny, self.T+1, norm[1])
                self.Phi_ux = low_block_tri_variable(self.nu, self.nx, self.T+1)
                self.Phi_xy = col_sparse_low_block_tri_variable(self.nx, self.ny, self.T+1, norm[1])
            else:
                raise Exception("Reweighted Actuator or Sensor.")
            
        self.Phi_matrix = cp.bmat( [[self.Phi_xx,   self.Phi_xy], 
                                    [self.Phi_ux,   self.Phi_uy]] )
        # define downshift operator
        self.Z = np.block([ [np.zeros([self.nx,self.T*self.nx]),    np.zeros([self.nx,self.nx])        ],
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
        self.E = None
        self.D = None
        self.F_trunc = None
        self.F_causal_rows_basis = None
        self.Phi_trunc = None
        self.Phi_uy_trunc = None
        self.causal_time = None

    def SLP_constraints(self, constant_matrix=None):
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
        SLP = [cp.bmat([[I - self.Z @ self.cal_A, -self.Z @ self.cal_B]]) @ self.Phi_matrix == cp.bmat([[I, np.zeros( (Tp1*self.nx, Tp1*self.ny) )]]),
                self.Phi_matrix @ cp.bmat([[I - self.Z @ self.cal_A], [-self.cal_C]]) == cp.bmat([[I], [np.zeros( (Tp1*self.nu, Tp1*self.nx) )]])]
        return SLP

    def calculate_dependent_variables(self, key):
        """
        Compute the controller F
        """
        F_test = self.Phi_uy.value - self.Phi_ux.value @ np.linalg.inv((self.Phi_xx.value).astype('float64')) @ self.Phi_xy.value
        if key=="Reweighted Nuclear Norm" or key=="Reweighted Sensor Norm":
            self.F = np.linalg.inv( (np.eye(self.nu*(self.T+1)) + self.Phi_ux.value @ self.Z @ self.cal_B).astype('float64') ) @ self.Phi_uy.value
        elif key=="Reweighted Actuator Norm":
            self.F = self.Phi_uy.value @ np.linalg.inv( (np.eye(self.ny*(self.T+1)) + self.cal_C @ self.Phi_xy.value).astype('float64') )
        print(key + ':', np.max(np.abs(F_test - self.F)))
        assert np.all(np.isclose( self.F.astype('float64'), F_test.astype('float64')) )
        filter = np.kron( np.tril(np.ones([self.T+1,self.T+1])) , np.ones([self.nu, self.ny]) )
        self.F = filter*self.F
        return
    
    def F_trunc_to_Phi_trunc(self):
        Phi_xx = np.linalg.inv( (np.eye(self.nx*(self.T+1)) - self.Z @ self.cal_A - self.Z @ self.cal_B @ self.F_trunc @ self.cal_C).astype('float64') )
        Phi_xy = Phi_xx.dot(self.Z).dot(self.cal_B).dot(self.F_trunc)
        Phi_ux = self.F_trunc.dot(self.cal_C).dot(Phi_xx)
        Phi_uy = ( np.eye(self.nu*(self.T+1)) + Phi_ux.dot(self.Z).dot(self.cal_B)).dot(self.F_trunc)
        Phi_uy_sum = self.F_trunc + self.F_trunc.dot(self.cal_C).dot(Phi_xx).dot(self.Z).dot(self.cal_B).dot(self.F_trunc)
        assert np.all(np.isclose( Phi_uy.astype('float64'), Phi_uy_sum.astype('float64')) )
        self.Phi_trunc = np.bmat([[Phi_xx, Phi_xy], [Phi_ux, Phi_uy]])
        return
    
    def causal_factorization(self, rank_eps):
        start = time.time()
        low_btri = self.F
        assert low_btri.shape[0] == self.nu*(self.T+1)
        assert low_btri.shape[1] == self.ny*(self.T+1)
        E = np.array([]).reshape((0,self.F.shape[1]))
        D = np.array([]).reshape((0,0)) # trick
        rank_counter = 0
        rank_low_btri = np.linalg.matrix_rank(low_btri, tol=rank_eps)
        added_rows = rank_low_btri*[None]
        for t in range(self.T+1):
            for s in range(self.nu):
                row = t*self.nu + s
                submat_new = low_btri[0:row+1, :] # rows up to row (note: last step row = (T+1)*nu)
                rank_new = np.linalg.matrix_rank(submat_new, tol=rank_eps)
                if rank_new - rank_counter == 1:
                    added_rows[rank_counter] = row
                    rank_counter += 1
                    # add vector to E matrix
                    E = np.vstack([E, submat_new[row:row+1, :]])
                    # modify D matrix
                    unit = np.zeros([1, rank_counter]) # E has shape (row+1,rank_counter)
                    unit[0, -1] = 1. # add a 1 at the last column of unit
                    D = np.hstack([D, np.zeros([row, 1])]) # E is (row, rank_counter)
                    D = np.vstack([D, unit]) # E has (row+1, rank_counter)
                    assert E.shape == (rank_counter, low_btri.shape[1])
                    assert D.shape == (row+1, rank_counter)

                elif rank_new == rank_counter:
                    # solve linear system
                    c = np.linalg.lstsq( E.T , low_btri[row, :])[0]
                    c = c.reshape(([1, rank_counter]))
                    D = np.vstack([D, c])
                    assert E.shape == (rank_counter, low_btri.shape[1])
                    assert D.shape == (row+1, rank_counter)
                else:
                    raise Exception('Rank increased more than 1.')
                
        assert E.shape == (rank_counter, low_btri.shape[1])
        assert D.shape == (low_btri.shape[0], rank_counter)
        assert rank_counter == rank_low_btri
        assert len(added_rows) == rank_low_btri
        # set attributees and compute truncated F
        self.E = E
        self.D = D
        self.F_trunc = D.dot(E)
        self.rank_F_trunc = rank_low_btri
        self.F_causal_row_basis = added_rows
        self.causal_time = time.time() - start
        return