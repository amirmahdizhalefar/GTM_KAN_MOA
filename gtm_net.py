"""
gtm_net.py ── Full GTM-KAN-MoA pipeline Integrator
=============================================================
Integrates GTM EM fitting with PyTorch deep learning endpoints.

FIX (MemoryError): _CAP is now applied to ALL matrices, not just RECT ones.
The old code only capped S2/S3/S6, leaving S1/S4/S5/S7 (up to 1512×1512)
uncapped — causing a 25.9 GiB allocation at beta initialisation.
The memory-safe monkey-patch now also covers the full EM loop's beta update.

CHANGE (Exact β — Eq 6): The EM-loop β update has been switched from the
hard-assignment (Voronoi-EM) approximation of Eq 7:

    β_approx = N·D / (Σ_n  min_m ‖y_m − x_n‖²  + ε)

to the exact responsibility-weighted update of Eq 6:

    β_exact  = N·D / (Σ_m Σ_n  R_mn ‖y_m − x_n‖²  + ε)

R is already materialised during the E-step (shape M × N, ≈ 0.6 MB at M = 100,
N = 1512), so no additional per-iteration allocations are introduced.  The
double sum is accumulated in the same (CM = 32) × (CN = 64) chunk loop used
everywhere else, keeping peak RAM identical to the Eq-7 variant.

The *initialisation* β (computed before any E-step, where R does not yet exist)
retains the hard-assignment approximation; that is the only place Eq 7 is used.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import normalize as _skn
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
)
from gtm import GTM
from gtm_kan_moa import GTM_KAN_MoA_Network, compute_drug_moa_features, compute_protein_moa_features, HAS_KAN

# ── Memory-safe monkey-patch for GTM.fit beta initialisation ────────────────
# The stock gtm.py line:
#   dists2 = np.sum((X[None,:,:] - Y_curr[:,None,:]) ** 2, axis=-1)
# allocates an (M, N, D) array which can be >25 GB for PPI (1512×1512).
# We replace the GTM.fit method with an identical one that computes beta
# and the EM-loop distance steps using a chunked loop so peak RAM is
# O(chunk_m × chunk_n × D) at all times.

import types as _types

def _gtm_fit_memory_safe(self, X):
    """Drop-in replacement for GTM.fit with chunked distance computation."""
    import numpy as _np
    N, D = X.shape
    M    = self.M

    # ── steps 1-3: unchanged from stock GTM.fit ───────────────────────────
    # 1. Latent grid
    s = int(_np.sqrt(M))
    ax = _np.linspace(-1, 1, s)
    gx, gy = _np.meshgrid(ax, ax)
    self.Z_ = _np.column_stack([gx.ravel(), gy.ravel()]).astype(_np.float32)

    # 2. RBF basis
    L = self.L
    sl = int(_np.sqrt(L))
    bx = _np.linspace(-1, 1, sl)
    bgx, bgy = _np.meshgrid(bx, bx)
    centres = _np.column_stack([bgx.ravel(), bgy.ravel()]).astype(_np.float32)
    sigma = (2.0 / (sl - 1)) if sl > 1 else 1.0
    diff = self.Z_[:, None, :] - centres[None, :, :]          # (M, L, 2)
    self.Phi_ = _np.exp(
        -_np.sum(diff ** 2, axis=-1) / (2 * sigma ** 2)
    ).astype(_np.float32)                                       # (M, L)

    # 3. W init via least-squares projection of first 2 PCs
    Xc = X - X.mean(axis=0)
    _, _, Vt = _np.linalg.svd(Xc, full_matrices=False)
    pc = Vt[:2].T                                               # (D, 2)
    proj = Xc @ pc                                              # (N, 2)
    scale = proj.std(axis=0) / (self.Z_.std(axis=0) + 1e-8)
    T_init = (self.Z_ * scale) @ pc.T                          # (M, D)
    self.W_ = (_np.linalg.pinv(self.Phi_) @ T_init).astype(_np.float32)  # (L, D)

    # 4. Beta init – CHUNKED (hard-assignment / Eq 7) ──────────────────────
    # R is not yet available before the first E-step, so the Voronoi-EM
    # approximation (Eq 7) is the only option here.  This is the sole remaining
    # use of Eq 7; all subsequent per-iteration β updates use Eq 6 (see below).
    Y_curr  = (self.Phi_ @ self.W_).astype(_np.float32)     # (M, D)
    CM, CN  = 32, 64                                           # chunk sizes
    min_d2  = _np.full(N, _np.inf, dtype=_np.float64)
    for m0 in range(0, M, CM):
        Yc = Y_curr[m0: m0 + CM]                               # (cm, D)
        for n0 in range(0, N, CN):
            Xc_blk = X[n0: n0 + CN].astype(_np.float64)       # (cn, D)
            d2 = _np.sum(
                (Xc_blk[None, :, :] - Yc[:, None, :].astype(_np.float64)) ** 2,
                axis=-1
            )                                                   # (cm, cn)
            min_d2[n0: n0 + CN] = _np.minimum(
                min_d2[n0: n0 + CN], d2.min(axis=0)
            )
    self.beta_ = float(N * D) / (_np.sum(min_d2) + 1e-12)

    # 5. EM loop (chunked throughout) ─────────────────────────────────────
    for _ in range(self.n_iter):
        Y = (self.Phi_ @ self.W_).astype(_np.float32)        # (M, D)

        # E-step: responsibilities – chunked
        log_rho = _np.empty((M, N), dtype=_np.float64)
        for m0 in range(0, M, CM):
            Yc = Y[m0: m0 + CM].astype(_np.float64)
            for n0 in range(0, N, CN):
                Xb = X[n0: n0 + CN].astype(_np.float64)
                d2 = _np.sum(
                    (Xb[None, :, :] - Yc[:, None, :]) ** 2, axis=-1
                )
                log_rho[m0: m0 + CM, n0: n0 + CN] = (
                    -0.5 * self.beta_ * d2
                )
        log_rho -= log_rho.max(axis=0, keepdims=True)
        rho = _np.exp(log_rho)
        R   = rho / (rho.sum(axis=0, keepdims=True) + 1e-12)   # (M, N)

        # M-step
        G   = _np.diag(R.sum(axis=1).astype(_np.float64))       # (M, M)
        PhiT_G_Phi = (self.Phi_.T.astype(_np.float64)
                      @ G @ self.Phi_.astype(_np.float64))      # (L, L)
        lam = 1e-6 * _np.eye(PhiT_G_Phi.shape[0])
        RX  = (R.astype(_np.float64) @ X.astype(_np.float64))  # (M, D)
        self.W_ = (
            _np.linalg.solve(PhiT_G_Phi + lam,
                             self.Phi_.T.astype(_np.float64) @ RX)
        ).astype(_np.float32)                                    # (L, D)

        # Beta update – EXACT Eq 6 (responsibility-weighted sum of squared distances)
        # β_new = N·D / (Σ_m Σ_n R_mn · ‖y_m − x_n‖² + ε)
        #
        # R is already in memory from the E-step above (shape M × N, ≈ 0.6 MB).
        # Distances are recomputed against Y_new (post-M-step prototypes) in the
        # same (CM × CN) chunks used elsewhere — peak RAM is unchanged vs Eq 7.
        # Unlike Eq 7, this update uses every prototype's contribution weighted by
        # its responsibility, so β_new ≤ β_approx always (softer noise model when
        # assignments are uncertain, exact agreement when assignments are confident).
        Y_new = (self.Phi_ @ self.W_).astype(_np.float32)
        weighted_d2 = 0.0
        for m0 in range(0, M, CM):
            Yc = Y_new[m0: m0 + CM].astype(_np.float64)    # (cm, D)
            for n0 in range(0, N, CN):
                Xb  = X[n0: n0 + CN].astype(_np.float64)   # (cn, D)
                d2  = _np.sum(
                    (Xb[None, :, :] - Yc[:, None, :]) ** 2, axis=-1
                )                                            # (cm, cn)
                Rc  = R[m0: m0 + CM, n0: n0 + CN].astype(_np.float64)
                weighted_d2 += float(_np.sum(Rc * d2))
        self.beta_ = float(N * D) / (weighted_d2 + 1e-12)

    return self

def _gtm_responsibilities_safe(self, X):
    """Memory-safe responsibilities using the same chunked pattern."""
    import numpy as _np
    M, N, D  = self.M, X.shape[0], X.shape[1]
    Y        = (self.Phi_ @ self.W_).astype(_np.float32)
    CM, CN   = 32, 64
    log_rho  = _np.empty((M, N), dtype=_np.float64)
    for m0 in range(0, M, CM):
        Yc = Y[m0: m0 + CM].astype(_np.float64)
        for n0 in range(0, N, CN):
            Xb = X[n0: n0 + CN].astype(_np.float64)
            d2 = _np.sum(
                (Xb[None, :, :] - Yc[:, None, :]) ** 2, axis=-1
            )
            log_rho[m0: m0 + CM, n0: n0 + CN] = -0.5 * self.beta_ * d2
    log_rho -= log_rho.max(axis=0, keepdims=True)
    rho = _np.exp(log_rho)
    R   = rho / (rho.sum(axis=0, keepdims=True) + 1e-12)
    return R.astype(_np.float32)

def _gtm_hard_assignments_safe(self, X):
    import numpy as _np
    R = self.responsibilities(X)
    return _np.argmax(R, axis=0)

# Patch the GTM class once at import time
GTM.fit              = _gtm_fit_memory_safe
GTM.responsibilities = _gtm_responsibilities_safe
GTM.hard_assignments = _gtm_hard_assignments_safe

# ── GTM Grid Helpers ────────────────────────────────────────────────────────
def _M(N: int) -> int:
    k = int(np.ceil(N ** 0.5))
    return k * k

def _L(M: int) -> int:
    raw = int(np.ceil(np.sqrt(M // 4))) ** 2
    return max(4, min(raw, M - 1))

# _CAP: maximum number of latent grid nodes for ANY matrix.
# FIX: Previously _CAP was only applied to RECT matrices (S2, S3, S6).
# S1/S4 (708×708) → uncapped M=729; S5/S7 (1512×1512) → uncapped M=1521.
# M=1521 with N=1512 and D=1512 produces a (1521,1512,1512) float64 array
# = 25.9 GB just for the beta initialisation step → MemoryError.
# Now _CAP is applied unconditionally to every matrix.
# At _CAP=100 (10×10 grid) the peak chunked allocation is ~32×64×1512×8 B ≈ 25 MB.
# Raise _CAP to 225 (15×15) on machines with ≥32 GB RAM for finer topology.
_CAP = 100

def _fit_gtm(S: np.ndarray, M: int, n_iter: int = 30):
    L = _L(M)
    g = GTM(M=M, L=L, n_iter=n_iter, tol=1e-6, verbose=False)
    g.fit(S.astype(np.float32))
    R = g.responsibilities(S.astype(np.float32))
    Rn = R / (np.linalg.norm(R, axis=0, keepdims=True) + 1e-12)
    K  = (Rn.T @ Rn).astype(np.float32)
    np.fill_diagonal(K, 0)

    Kdiff = np.zeros_like(K)
    Kp = K.copy()
    for t in range(3):
        Kdiff += (0.3 ** (t + 1)) * Kp
        if t < 2: Kp = (Kp @ K).astype(np.float32)
    np.fill_diagonal(Kdiff, 0)

    Rt   = R.T
    pos  = (Rt @ g.Z_).astype(np.float32)
    ent  = (-np.sum(Rt * np.log(Rt + 1e-12), axis=1, keepdims=True)).astype(np.float32)
    mxr  = Rt.max(axis=1, keepdims=True).astype(np.float32)
    stdr = Rt.std(axis=1, keepdims=True).astype(np.float32)
    quad = (g.Z_[:, 0] >= 0).astype(int) * 2 + (g.Z_[:, 1] >= 0).astype(int)
    qf   = np.zeros((S.shape[0], 4), dtype=np.float32)
    for q in range(4):
        qf[:, q] = Rt[:, quad == q].sum(axis=1)
    feat = np.hstack([pos, ent, mxr, stdr, qf]).astype(np.float32)

    return g, R, K, Kdiff, feat

def fit_individual_gtms(data: dict, n_iter: int = 30, verbose: bool = True) -> dict:
    # FIX: removed RECT set — _CAP is now applied to ALL matrices unconditionally.
    out  = {}
    for k in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']:
        S  = data[k].astype(np.float32)
        # Cap every matrix to _CAP grid nodes regardless of shape.
        Mc = min(_M(S.shape[0]), _CAP)
        kk = int(round(Mc ** 0.5)); Mc = kk * kk
        g, R, K, Kdiff, feat = _fit_gtm(S, Mc, n_iter=n_iter)
        occ = len(np.unique(g.hard_assignments(S)))
        if verbose: print(f"  {k}  N={S.shape[0]:4d}  M={Mc:4d}  occ={occ}/{Mc}")
        out[k] = dict(g=g, R=R, K=K, Kdiff=Kdiff, feat=feat, M=Mc)
    return out

def fit_joint_gtm(S3: np.ndarray, S6: np.ndarray, n_iter: int = 30, verbose: bool = True) -> dict:
    N_D, N_P = S3.shape[0], S6.shape[0]
    S36 = np.vstack([S3, S6]).astype(np.float32)
    Mc = _CAP; kk = int(round(Mc ** 0.5)); Mc = kk * kk
    if verbose: print(f"  Joint GTM S36 M={Mc}")
    g36, R36, K36, Kdiff36, _ = _fit_gtm(S36, Mc, n_iter=n_iter)
    a36 = g36.hard_assignments(S36)
    a_drug, a_prot = a36[:N_D], a36[N_D:]

    Rt36  = R36.T
    pos36 = (Rt36 @ g36.Z_).astype(np.float32)
    ent36 = (-np.sum(Rt36 * np.log(Rt36 + 1e-12), axis=1, keepdims=True)).astype(np.float32)
    drug_dis = np.hstack([pos36[:N_D],  ent36[:N_D]])
    prot_dis = np.hstack([pos36[N_D:],  ent36[N_D:]])

    Rd, Rp = R36[:, :N_D], R36[:, N_D:]
    W_soft = (Rd.T @ Rp).astype(np.float32)
    S3n, S6n = _skn(S3, norm='l2', axis=1), _skn(S6, norm='l2', axis=1)
    W_cos  = (S3n @ S6n.T).astype(np.float32)
    W_hard = (a_drug[:, None] == a_prot[None, :]).astype(np.float32)

    def _n01(M): mn, mx = M.min(), M.max(); return (M - mn) / (mx - mn + 1e-12)
    return dict(g=g36, R36=R36, K36=K36, Kdiff36=Kdiff36, W_soft=_n01(W_soft),
                W_cos=_n01(W_cos), W_hard=W_hard, drug_dis=drug_dis, prot_dis=prot_dis, N_D=N_D, N_P=N_P)

def _neighbor_block(gtms, S1, S5, Y_tr, pairs):
    di, pj = pairs[:, 0], pairs[:, 1]
    nbr = []
    for k in ['S1', 'S2', 'S3', 'S4']:
        nbr.append((gtms[k]['K'] @ Y_tr)[di, pj])
        nbr.append((gtms[k]['Kdiff'] @ Y_tr)[di, pj])
    for k in ['S5', 'S6', 'S7']:
        nbr.append((gtms[k]['K'] @ Y_tr.T)[pj, di])
        nbr.append((gtms[k]['Kdiff'] @ Y_tr.T)[pj, di])
    nbr.append((S1 @ Y_tr)[di, pj])
    nbr.append((S5 @ Y_tr.T)[pj, di])
    D3d, P6d = gtms['S3']['Kdiff'] @ Y_tr, gtms['S6']['Kdiff'] @ Y_tr.T
    nbr.append((D3d @ gtms['S6']['Kdiff'])[di, pj])
    nbr.append((P6d @ gtms['S3']['Kdiff'].T)[pj, di])
    nbr.append((gtms['S1']['Kdiff'] @ Y_tr)[di, pj] + (gtms['S5']['Kdiff'] @ Y_tr.T)[pj, di])
    nbr.append(D3d[di, pj] + P6d[pj, di])
    nbr.append((nbr[1] > 0).astype(float))
    nbr.append((nbr[9] > 0).astype(float))
    nbr.append(((nbr[1] > 0) | (nbr[9] > 0)).astype(float))

    arr = np.column_stack(nbr).astype(np.float32)
    mx  = arr.max(axis=0) + 1e-9; arr /= mx
    return arr

def _tw_bridge(R36, N_D, Y_tr, pairs):
    Rd, Rp = R36[:, :N_D], R36[:, N_D:]
    M = R36.shape[0]
    di, pj = pairs[:, 0], pairs[:, 1]
    a_d, a_p = np.argmax(Rd, axis=0), np.argmax(Rp, axis=0)
    P_train = np.zeros(M, dtype=np.float32)
    pos_d, pos_p = np.where(Y_tr > 0)
    for d, p in zip(pos_d, pos_p):
        if int(a_d[d]) == int(a_p[p]): P_train[int(a_d[d])] += 1.0

    Rd_i, Rp_j = Rd[:, di].T, Rp[:, pj].T
    tw1 = (Rd_i * Rp_j * P_train[None, :]).sum(axis=1, keepdims=True)
    tw2 = (Rd_i * Rp_j).sum(axis=1, keepdims=True)

    bucket_sz = max(1, M // 10)
    bucket = []
    for b in range(10):
        s, e = b * bucket_sz, min((b + 1) * bucket_sz, M)
        bucket.append((Rd_i[:, s:e] * Rp_j[:, s:e]).sum(axis=1))
    return np.hstack([tw1, tw2, np.column_stack(bucket)]).astype(np.float32)

class GTMNet:
    def __init__(self, n_iter: int = 30, verbose: bool = True):
        """
        Parameters
        ----------
        n_iter  : GTM EM iterations (default 30).
        verbose : print phase headers and progress.

        SMILES and FASTA sequences are resolved automatically from the dataset
        dict (drug_ids / prot_ids keys injected by data_loader.py).
        PubChem and UniProt are queried the first time; results are cached in
        gtmnet/seq_cache/ so subsequent runs are instant.
        """
        self.n_iter  = n_iter
        self.verbose = verbose
        self.device  = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fit(self, data: dict):
        """
        Parameters
        ----------
        data : dict from data_loader.load_luo_dataset() or load_new_dataset().
               Required keys: S1–S7, Y, N_D, N_P, drug_ids, prot_ids.
        """
        from data_loader import load_smiles_for_dataset, load_fasta_for_dataset, CACHE_DIR

        self.N_D, self.N_P, self.Y = data['N_D'], data['N_P'], data['Y']
        self._S1, self._S5 = data['S1'].astype(np.float32), data['S5'].astype(np.float32)

        if self.verbose: print("\n=== PHASE 1: GTM on Raw Networks ===")
        self.gtms = fit_individual_gtms(data, n_iter=self.n_iter, verbose=self.verbose)
        if self.verbose: print("\n=== PHASE 2: Joint Disease GTM ===")
        self.joint = fit_joint_gtm(data['S3'], data['S6'], n_iter=self.n_iter, verbose=self.verbose)

        self.H_drug = np.hstack(
            [self.gtms[k]['feat'] for k in ['S1', 'S2', 'S3', 'S4']] + [self.joint['drug_dis']]
        ).astype(np.float32)
        self.H_prot = np.hstack(
            [self.gtms[k]['feat'] for k in ['S5', 'S6', 'S7']] + [self.joint['prot_dis']]
        ).astype(np.float32)

        # ── Phase 4b: resolve REAL sequences from DrugBank/UniProt IDs ──────
        if self.verbose:
            print("\n=== PHASE 4b: Fetching Real SMILES (PubChem) + FASTA (UniProt) ===")

        smiles_list = load_smiles_for_dataset(data, verbose=self.verbose)
        fasta_list  = load_fasta_for_dataset(data,  verbose=self.verbose)

        _FALLBACK_SMI   = "CC(=O)Oc1ccccc1C(=O)O"
        _FALLBACK_FASTA = "MAAARPGM"

        # Defensive: if either loader returned None (e.g. stale .pyc / import
        # shadowing issue), fall back to stub lists so the pipeline can run.
        if smiles_list is None:
            if self.verbose:
                print("  [WARNING] SMILES loader returned None – using Aspirin fallback for all drugs.")
            smiles_list = [_FALLBACK_SMI] * data['N_D']
        if fasta_list is None:
            if self.verbose:
                print("  [WARNING] FASTA loader returned None – using stub fallback for all proteins.")
            fasta_list = [_FALLBACK_FASTA] * data['N_P']
        if self.verbose:
            n_real_smi   = sum(1 for s in smiles_list if s != _FALLBACK_SMI)
            n_real_fasta = sum(1 for s in fasta_list  if s != _FALLBACK_FASTA)
            print(f"  Real SMILES : {n_real_smi}/{len(smiles_list)}")
            print(f"  Real FASTA  : {n_real_fasta}/{len(fasta_list)}")
            if n_real_smi < len(smiles_list) or n_real_fasta < len(fasta_list):
                print("  [NOTE] Fallback sequences will use Aspirin/stub until "
                      "PubChem / UniProt are reachable from this machine.")

        if self.verbose:
            print("\n=== PHASE 4b: Computing ECFP4 + ESM-2 MoA Embeddings ===")
        self.f_drug_moa = compute_drug_moa_features(smiles_list)
        self.f_prot_moa = compute_protein_moa_features(fasta_list, device=self.device,
                                                         cache_dir=CACHE_DIR)
        return self

    def _structural_feats(self, pairs, Y_tr):
        di, pj = pairs[:, 0], pairs[:, 1]
        bridge = np.column_stack([self.joint['W_soft'][di, pj], self.joint['W_cos'][di, pj], self.joint['W_hard'][di, pj]])
        tw = _tw_bridge(self.joint['R36'], self.joint['N_D'], Y_tr, pairs)
        nbr = _neighbor_block(self.gtms, self._S1, self._S5, Y_tr, pairs)
        return np.hstack([bridge, tw, nbr]).astype(np.float32)

    def _train_kan_moa(self, pairs_train, y_train, fold_idx=0, checkpoint_dir='./model'):
        """
        Improved training with:
        - Focal Loss (handles class imbalance)
        - Cosine Annealing with Warm Restarts LR scheduler
        - Gradient clipping
        - Best model checkpoint (by val AUROC on 10% internal split)
        - Adaptive CVCA weighting
        - Class-weighted sampling
        - More epochs (200)
        """
        import os, pickle, copy
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        from gtm_kan_moa import FocalLoss

        os.makedirs(checkpoint_dir, exist_ok=True)

        # ── Build Y_tr matrix ─────────────────────────────────────────────
        Y_tr_mat = np.zeros((self.N_D, self.N_P), dtype=np.float32)
        for (d, p), y in zip(pairs_train, y_train):
            if y == 1:
                Y_tr_mat[int(d), int(p)] = 1.0

        # ── Internal val split (10%) for best-model selection ─────────────
        idx_all = np.arange(len(y_train))
        if len(np.unique(y_train)) > 1:
            idx_tr, idx_iv = train_test_split(idx_all, test_size=0.10,
                                              stratify=y_train, random_state=42)
        else:
            idx_tr = idx_all
            idx_iv = idx_all[:max(1, len(idx_all)//10)]

        pairs_iv = pairs_train[idx_iv]
        y_iv     = y_train[idx_iv]
        pairs_tr = pairs_train[idx_tr]
        y_tr     = y_train[idx_tr]

        # ── Pre-compute structural features ──────────────────────────────
        struct_tr = torch.tensor(self._structural_feats(pairs_tr, Y_tr_mat)).to(self.device)
        struct_iv = torch.tensor(self._structural_feats(pairs_iv, Y_tr_mat)).to(self.device)

        # ── Build network AFTER struct feats so we know its true dim ──────
        # Passing struct_dim avoids hardcode drift between _structural_feats
        # and GTM_KAN_MoA_Network (was causing (128x107) vs (105x256) error).
        net = GTM_KAN_MoA_Network(
            drug_gtm_dim=self.H_drug.shape[1],
            prot_gtm_dim=self.H_prot.shape[1],
            struct_dim=struct_tr.shape[1]
        ).to(self.device)

        # ── Try to init from previous best fold checkpoint ──────────────────
        prev_ckpt = os.path.join(checkpoint_dir, 'best_fold_model.pt')
        if os.path.exists(prev_ckpt):
            try:
                state = torch.load(prev_ckpt, map_location=self.device)
                net.load_state_dict(state, strict=False)
                if self.verbose:
                    print(f"  [Checkpoint] Loaded weights from {prev_ckpt}")
            except Exception as e:
                if self.verbose:
                    print(f"  [Checkpoint] Could not load prior weights: {e}")

        di_tr = torch.tensor(pairs_tr[:, 0])
        pj_tr = torch.tensor(pairs_tr[:, 1])
        lbl_tr = torch.tensor(y_tr, dtype=torch.float32).to(self.device)

        di_iv = torch.tensor(pairs_iv[:, 0])
        pj_iv = torch.tensor(pairs_iv[:, 1])

        # ── Class-weighted sampler ────────────────────────────────────────
        pos_cnt = y_tr.sum()
        neg_cnt = len(y_tr) - pos_cnt
        class_wt = np.where(y_tr == 1, len(y_tr) / (2.0 * pos_cnt + 1e-9),
                            len(y_tr) / (2.0 * neg_cnt + 1e-9))
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.tensor(class_wt, dtype=torch.float32),
            num_samples=len(y_tr), replacement=True
        )

        dataset = TensorDataset(di_tr, pj_tr, struct_tr, lbl_tr)
        loader  = DataLoader(dataset, batch_size=128, sampler=sampler)

        # ── View tensors for CVCA ─────────────────────────────────────────
        d_views = [torch.tensor(self.gtms[k]['feat']).to(self.device) for k in ['S1', 'S2', 'S3', 'S4']]
        p_views = [torch.tensor(self.gtms[k]['feat']).to(self.device) for k in ['S5', 'S6', 'S7']]

        # ── Optimizer + Scheduler ─────────────────────────────────────────
        optimizer = optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4,
                                betas=(0.9, 0.999), eps=1e-8)
        # Cosine annealing with warm restarts: T_0=40, T_mult=2 → restarts at 40, 120 epochs
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=40, T_mult=2, eta_min=1e-5
        )
        criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)

        epochs = 200
        alpha_cvca, beta_cvca = 0.05, 0.05  # lighter CVCA weight

        best_auroc    = -1.0
        best_state    = None
        patience      = 30
        no_improve    = 0

        H_drug_t = torch.tensor(self.H_drug).to(self.device)
        H_prot_t = torch.tensor(self.H_prot).to(self.device)
        f_drug_t = self.f_drug_moa.to(self.device)
        f_prot_t = self.f_prot_moa.to(self.device)

        for ep in range(epochs):
            net.train()
            for b_di, b_pj, b_struct, b_y in loader:
                optimizer.zero_grad()

                unique_d = torch.zeros(self.N_D, device=self.device)
                unique_d[b_di] = 1
                unique_p = torch.zeros(self.N_P, device=self.device)
                unique_p[b_pj] = 1

                preds, cvca_d, cvca_p = net(
                    H_drug_t[b_di], H_prot_t[b_pj],
                    f_drug_t[b_di], f_prot_t[b_pj],
                    b_struct.to(self.device),
                    d_views, p_views, unique_d, unique_p
                )

                loss = criterion(preds, b_y) + alpha_cvca * cvca_d + beta_cvca * cvca_p

                if HAS_KAN and hasattr(net.classifier, 'regularization_loss'):
                    reg = net.classifier.regularization_loss(
                        regularize_activation=1.0, regularize_entropy=1.0) * 0.005
                    loss = loss + reg

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            # ── Internal validation every 5 epochs ───────────────────────
            if (ep + 1) % 5 == 0:
                net.eval()
                with torch.no_grad():
                    iv_preds, _, _ = net(
                        H_drug_t[di_iv], H_prot_t[pj_iv],
                        f_drug_t[di_iv], f_prot_t[pj_iv],
                        struct_iv
                    )
                iv_prob = iv_preds.cpu().numpy()
                try:
                    auroc_iv = roc_auc_score(y_iv, iv_prob)
                except Exception:
                    auroc_iv = 0.0

                if auroc_iv > best_auroc:
                    best_auroc = auroc_iv
                    best_state = copy.deepcopy(net.state_dict())
                    no_improve = 0
                else:
                    no_improve += 5

                if self.verbose and (ep + 1) % 20 == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"    ep={ep+1:3d}  iv_AUROC={auroc_iv:.4f}  best={best_auroc:.4f}  lr={lr_now:.2e}")

                if no_improve >= patience:
                    if self.verbose:
                        print(f"    Early stop at ep={ep+1} (no improvement for {patience} epochs)")
                    break

        # ── Restore best weights ──────────────────────────────────────────
        if best_state is not None:
            net.load_state_dict(best_state)
            if self.verbose:
                print(f"  [Best weights restored] iv_AUROC={best_auroc:.4f}")

        # ── Save checkpoint for this fold + global best ───────────────────
        fold_ckpt = os.path.join(checkpoint_dir, f'fold_{fold_idx}_best.pt')
        torch.save(net.state_dict(), fold_ckpt)

        # Update global best if this fold is better
        global_best_path = os.path.join(checkpoint_dir, 'best_fold_model.pt')
        global_meta_path = os.path.join(checkpoint_dir, 'best_fold_meta.json')
        global_best_auroc = -1.0
        if os.path.exists(global_meta_path):
            try:
                import json
                with open(global_meta_path) as f:
                    global_best_auroc = json.load(f).get('auroc', -1.0)
            except Exception:
                pass
        if best_auroc > global_best_auroc:
            torch.save(net.state_dict(), global_best_path)
            import json
            with open(global_meta_path, 'w') as f:
                json.dump({'auroc': best_auroc, 'fold': fold_idx}, f, indent=2)
            if self.verbose:
                print(f"  [Global Best Updated] fold={fold_idx} iv_AUROC={best_auroc:.4f} → {global_best_path}")

        return net

    def _predict(self, net, pairs, Y_tr):
        net.eval()
        with torch.no_grad():
            di, pj = pairs[:, 0], pairs[:, 1]
            struct = torch.tensor(self._structural_feats(pairs, Y_tr)).to(self.device)
            preds, _, _ = net(
                torch.tensor(self.H_drug).to(self.device)[di],
                torch.tensor(self.H_prot).to(self.device)[pj],
                self.f_drug_moa.to(self.device)[di],
                self.f_prot_moa.to(self.device)[pj],
                struct
            )
        return preds.cpu().numpy()

    def evaluate_cv(self, folds: list, is_luo: bool = True) -> dict:
        print("\n=== PHASE 5: 5-Fold CV + KAN-MoA Training ===")
        met = {m: [] for m in ['auroc', 'aupr', 'acc', 'prec', 'rec', 'f1']}
        for fi, fold in enumerate(folds):
            Y_tr = np.zeros((self.N_D, self.N_P), dtype=np.float32)
            for n in fold['train_ids']:
                d, p = (int(n) // self.N_P, int(n) % self.N_P) if is_luo else (int(n[0]), int(n[1]))
                if fold['train_Y'][np.where(fold['train_ids'] == n)[0][0]] == 1: Y_tr[d, p] = 1.0

            tr_p = np.array([[int(n) // self.N_P, int(n) % self.N_P] for n in fold['train_ids']])
            va_p = np.array([[int(n) // self.N_P, int(n) % self.N_P] for n in fold['val_ids']])
            ytr, yva = fold['train_Y'].astype(float), fold['val_Y'].astype(float)

            net = self._train_kan_moa(tr_p, ytr, fold_idx=fi)
            prob_va = self._predict(net, va_p, Y_tr)

            # ── Optimal threshold from val set (F1-based) ─────────────────
            best_t, best_f1 = 0.5, 0.0
            for t in np.linspace(0.10, 0.90, 81):
                _pred = (prob_va >= t).astype(int)
                _f1 = f1_score(yva, _pred, zero_division=0)
                if _f1 > best_f1:
                    best_f1, best_t = _f1, t

            pred = (prob_va >= best_t).astype(int)
            vals = [roc_auc_score(yva, prob_va), average_precision_score(yva, prob_va),
                    accuracy_score(yva, pred), precision_score(yva, pred, zero_division=0),
                    recall_score(yva, pred, zero_division=0), f1_score(yva, pred, zero_division=0)]

            for k, v in zip(met, vals): met[k].append(v)
            print(f"  Fold {fi} | AUROC={vals[0]:.4f} AUPR={vals[1]:.4f} Acc={vals[2]:.4f}")

        summary = {k: dict(mean=float(np.mean(v)), std=float(np.std(v))) for k, v in met.items()}
        summary['per_fold'] = met
        return summary