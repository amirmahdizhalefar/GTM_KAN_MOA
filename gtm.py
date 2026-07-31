"""
gtm.py ── Generative Topographic Mapping
=============================================================
A numerically stable implementation of GTM for biological networks.
"""

import numpy as np
from scipy.spatial.distance import cdist

class GTM:
    def __init__(self, M=100, L=25, n_iter=30, tol=1e-6, verbose=False):
        """
        M: Number of prototypes (latent grid size)
        L: Number of RBF centers
        n_iter: Maximum EM iterations
        """
        self.M = M
        self.L = L
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, X):
        X = X.astype(np.float32)
        N, D = X.shape
        
        # 1. Initialize latent grid Z_ (M x 2)
        k = int(np.ceil(np.sqrt(self.M)))
        grid_1d = np.linspace(-1, 1, k)
        X_grid, Y_grid = np.meshgrid(grid_1d, grid_1d)
        self.Z_ = np.column_stack([X_grid.ravel(), Y_grid.ravel()])[:self.M].astype(np.float32)
        
        # 2. Initialize RBF centers mu_ (L x 2)
        k_l = int(np.ceil(np.sqrt(self.L)))
        grid_1d_l = np.linspace(-1, 1, k_l)
        X_grid_l, Y_grid_l = np.meshgrid(grid_1d_l, grid_1d_l)
        self.mu_ = np.column_stack([X_grid_l.ravel(), Y_grid_l.ravel()])[:self.L].astype(np.float32)
        
        # 3. Calculate RBF width
        sigma = 2.0 / (k_l - 1) if k_l > 1 else 1.0
        
        # 4. Construct Phi matrix
        dist2 = cdist(self.Z_, self.mu_, 'sqeuclidean').astype(np.float32)
        Phi = np.exp(-dist2 / (2 * sigma**2)).astype(np.float32)
        self.Phi = np.column_stack([Phi, np.ones(self.M, dtype=np.float32)])
        
        # 5. Initialize weights (W) and precision (beta)
        self.W = np.random.randn(self.L + 1, D).astype(np.float32) * 0.1
        var_x = np.var(X)
        self.beta = 1.0 / (var_x if var_x > 0 else 1.0)
        
        # EM Algorithm
        for iteration in range(self.n_iter):
            # --- E-step ---
            Y_pt = self.Phi @ self.W
            dist_X_Y = cdist(Y_pt, X, 'sqeuclidean').astype(np.float32)
            
            # Log-sum-exp trick for responsibilities to prevent underflow
            log_R = -0.5 * self.beta * dist_X_Y
            max_log_R = np.max(log_R, axis=0, keepdims=True)
            R_exp = np.exp(log_R - max_log_R)
            sum_R_exp = np.sum(R_exp, axis=0, keepdims=True) + 1e-12
            self.R = R_exp / sum_R_exp  # Shape: (M x N)
            
            # --- M-step ---
            G = np.sum(self.R, axis=1)
            Phi_T_G_Phi = self.Phi.T @ (G[:, None] * self.Phi)
            R_X = self.R @ X
            Phi_T_R_X = self.Phi.T @ R_X
            
            # Solve for W using Ridge regression-style stabilization
            try:
                self.W = np.linalg.solve(Phi_T_G_Phi + 1e-6 * np.eye(self.L + 1, dtype=np.float32), Phi_T_R_X)
            except np.linalg.LinAlgError:
                self.W = np.linalg.solve(Phi_T_G_Phi + 1e-2 * np.eye(self.L + 1, dtype=np.float32), Phi_T_R_X)
            
            # Update precision (beta)
            Y_pt_new = self.Phi @ self.W
            dist_X_Y_new = cdist(Y_pt_new, X, 'sqeuclidean').astype(np.float32)
            self.beta = (N * D) / (np.sum(self.R * dist_X_Y_new) + 1e-12)
            
        return self

    def responsibilities(self, X):
        X = X.astype(np.float32)
        Y_pt = self.Phi @ self.W
        dist_X_Y = cdist(Y_pt, X, 'sqeuclidean').astype(np.float32)
        
        log_R = -0.5 * self.beta * dist_X_Y
        max_log_R = np.max(log_R, axis=0, keepdims=True)
        R_exp = np.exp(log_R - max_log_R)
        sum_R_exp = np.sum(R_exp, axis=0, keepdims=True) + 1e-12
        return R_exp / sum_R_exp
        
    def hard_assignments(self, X):
        R = self.responsibilities(X)
        return np.argmax(R, axis=0)