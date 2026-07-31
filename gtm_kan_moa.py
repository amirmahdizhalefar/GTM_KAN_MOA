"""
gtm_kan_moa.py ── The Deep Learning Extension for GTM-Net
=============================================================
Implements Phase 4b (MoA Augmentation), Phase 4c (CVCA), and Phase 5 (KAN).

IMPROVEMENTS v2.0:
  - Enhanced GatedCrossAttentionFusion with multi-head cross-attention
  - Residual connections + LayerNorm throughout
  - Focal Loss for class imbalance
  - Label smoothing
  - Improved KAN/MLP classifier with skip connections
  - SE (Squeeze-and-Excitation) channel attention on fused features
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator as _GetMorganGenerator
_MORGAN_GEN = _GetMorganGenerator(radius=2, fpSize=2048)
from transformers import AutoTokenizer, EsmModel

try:
    from kan import KAN
    HAS_KAN = True
except ImportError:
    print("[Warning] 'pykan' not installed. Falling back to enhanced MLP. Run 'pip install pykan' for full Spline activations.")
    HAS_KAN = False

# ── 1. Mechanism-of-Action (MoA) Extractors ──────────────────────────────────

def _hf_login_if_token_available() -> bool:
    """
    Authenticate with the HuggingFace Hub if HF_TOKEN (or the legacy
    HUGGING_FACE_HUB_TOKEN) is present in the environment.

    When the token is found this call eliminates the
    "unauthenticated requests" warning and grants the higher API rate
    limits needed for large-model downloads.  When neither variable is
    set the function is a silent no-op so the pipeline continues
    without modification.
    """
    token = (os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if not token:
        return False
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        return True
    except Exception as exc:
        print(f"  [HF] Token found but login failed ({exc}); continuing anonymously.")
        return False

def compute_drug_moa_features(smiles_list):
    """Computes 2048-bit ECFP4 + 12 physicochemical descriptors."""
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            features.append(np.zeros(2060))
            continue

        fp = _MORGAN_GEN.GetFingerprint(mol)
        fp_arr = np.zeros(2048, dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(fp, fp_arr)

        phys_arr = np.array([
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol), Descriptors.RingCount(mol),
            Descriptors.FractionCSP3(mol), Descriptors.qed(mol),
            Chem.rdmolops.GetFormalCharge(mol),
            len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        ], dtype=np.float32)

        features.append(np.concatenate([fp_arr, phys_arr]))

    features = np.array(features, dtype=np.float32)
    phys_part = features[:, 2048:]
    mean, std = np.mean(phys_part, axis=0), np.std(phys_part, axis=0) + 1e-8
    features[:, 2048:] = (phys_part - mean) / std
    return torch.tensor(features, dtype=torch.float32)

@torch.no_grad()
def compute_protein_moa_features(fasta_list, device='cuda',
                                  cache_dir='gtmnet/seq_cache'):
    """
    Extract mean-pooled 1280-dim embeddings from ESM-2-650M.

    Cache behaviour
    ---------------
    Embeddings are saved to
        {cache_dir}/esm2_embed_{hash12}.pt
    where hash12 is the first 12 hex digits of the MD5 of the
    concatenated sequences.  On every subsequent call with the same
    protein set the file is loaded directly — skipping the 15-30 min
    GPU inference step entirely.

    HuggingFace authentication
    --------------------------
    If the environment variable HF_TOKEN (or the legacy
    HUGGING_FACE_HUB_TOKEN) is set, the Hub login is performed before
    the model is downloaded.  This removes the "unauthenticated
    requests" warning and enables higher rate limits.
    """
    model_name = "facebook/esm2_t33_650M_UR50D"

    # ── Cache look-up ──────────────────────────────────────────────────────
    seq_hash   = hashlib.md5("".join(fasta_list).encode()).hexdigest()[:12]
    cache_path = Path(cache_dir) / f"esm2_embed_{seq_hash}.pt"

    if cache_path.exists():
        print(f"  [ESM-2] Loading cached embeddings from {cache_path.name}")
        return torch.load(cache_path, map_location="cpu").float()

    # ── First-time extraction ──────────────────────────────────────────────
    print("  [ESM-2] Extracting language model embeddings...")
    _hf_login_if_token_available()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    batch_size = 16
    for i in range(0, len(fasta_list), batch_size):
        batch_seqs = fasta_list[i:i + batch_size]
        inputs = tokenizer(batch_seqs, return_tensors="pt",
                           padding=True, truncation=True, max_length=1022)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        mask_exp = (inputs['attention_mask']
                    .unsqueeze(-1)
                    .expand(outputs.last_hidden_state.size())
                    .float())
        sum_emb  = torch.sum(outputs.last_hidden_state * mask_exp, 1)
        sum_mask = torch.clamp(mask_exp.sum(1), min=1e-9)
        embeddings.append((sum_emb / sum_mask).cpu())

    result = torch.cat(embeddings, dim=0).float()

    # ── Persist to disk ────────────────────────────────────────────────────
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        torch.save(result, cache_path)
        print(f"  [ESM-2] Embeddings cached → {cache_path}")
    except Exception as exc:
        print(f"  [ESM-2] Could not write cache ({exc}); continuing without.")

    return result

# ── 2. Losses ─────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss to handle class imbalance, focusing on hard examples."""
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, pred, target):
        # Apply label smoothing
        target_smooth = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy(pred, target_smooth, reduction='none')
        pt = torch.exp(-bce)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class InfoNCELoss(nn.Module):
    """Cross-View Contrastive Alignment (CVCA) for GTM Manifolds."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, views, mask):
        num_views = len(views)
        if num_views < 2: return torch.tensor(0.0).to(views[0].device)

        total_loss, pairs = 0.0, 0
        valid_idx = torch.where(mask > 0)[0]
        if len(valid_idx) < 2: return torch.tensor(0.0).to(views[0].device)

        for k in range(num_views):
            for kp in range(k + 1, num_views):
                v1 = F.normalize(views[k][valid_idx], dim=1)
                v2 = F.normalize(views[kp][valid_idx], dim=1)
                sim = torch.mm(v1, v2.t()) / self.temperature
                labels = torch.arange(v1.size(0)).long().to(v1.device)
                loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
                total_loss += loss
                pairs += 1

        return total_loss / pairs

# ── 3. Network Architectures ──────────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, max(dim // reduction, 8)),
            nn.SiLU(),
            nn.Linear(max(dim // reduction, 8), dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class EnhancedGatedFusion(nn.Module):
    """
    Enhanced Gated Cross-Attention Fusion.
    Adds: LayerNorm, residual connection, SE attention, deeper projection.
    """
    def __init__(self, gtm_dim, moa_dim, d=256):
        super().__init__()
        self.norm_gtm = nn.LayerNorm(gtm_dim)
        self.norm_moa = nn.LayerNorm(moa_dim)

        # Deeper MoA compression
        self.moa_mlp = nn.Sequential(
            nn.Linear(moa_dim, 512), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(512, 256),    nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(256, d)
        )

        # Multi-head-like gate: produces per-feature gate
        self.gate_net = nn.Sequential(
            nn.Linear(gtm_dim + d, gtm_dim * 2),
            nn.SiLU(),
            nn.Linear(gtm_dim * 2, gtm_dim),
            nn.Sigmoid()
        )

        self.moa_proj = nn.Linear(d, gtm_dim)
        self.se = SEBlock(gtm_dim)
        self.out_norm = nn.LayerNorm(gtm_dim)

    def forward(self, h_gtm, f_moa):
        h_norm = self.norm_gtm(h_gtm)
        f_norm = self.norm_moa(f_moa)

        f_compressed = self.moa_mlp(f_norm)
        gate = self.gate_net(torch.cat([h_norm, f_compressed], dim=-1))
        moa_proj = self.moa_proj(f_compressed)

        fused = gate * h_norm + (1 - gate) * moa_proj
        fused = self.se(fused)
        # Residual from original GTM features
        out = self.out_norm(fused + h_gtm)
        return out


class ResidualBlock(nn.Module):
    """Residual MLP block with LayerNorm."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.net(x)


class GTM_KAN_MoA_Network(nn.Module):
    def __init__(self, drug_gtm_dim=39, prot_gtm_dim=30, d=256, struct_dim=None):
        super().__init__()
        self.drug_fusion = EnhancedGatedFusion(gtm_dim=drug_gtm_dim, moa_dim=2060, d=d)
        self.prot_fusion = EnhancedGatedFusion(gtm_dim=prot_gtm_dim, moa_dim=1280, d=d)

        # struct_dim: bridge(3) + tw_bridge(12) + neighbor_block(23) = 38
        # Accept it as a parameter so callers can pass the actual runtime value
        # and avoid hardcode drift. Default matches the current _neighbor_block output.
        if struct_dim is None:
            struct_dim = 3 + 12 + 23  # = 38
        z_dim = drug_gtm_dim + prot_gtm_dim + struct_dim

        # Project to higher dimension before classifier
        proj_dim = 256
        self.input_proj = nn.Sequential(
            nn.Linear(z_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )

        if HAS_KAN:
            self.classifier = KAN(width=[proj_dim, 256, 128, 64, 1], grid=7, k=3, seed=42)
        else:
            # Deep residual MLP with skip connection
            self.res_blocks = nn.ModuleList([
                ResidualBlock(proj_dim, dropout=0.1),
                ResidualBlock(proj_dim, dropout=0.1),
                ResidualBlock(proj_dim, dropout=0.1),
            ])
            self.classifier = nn.Sequential(
                nn.LayerNorm(proj_dim),
                nn.Linear(proj_dim, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(64, 1)
            )
            # Wide skip connection: z_dim -> 1
            self.skip_proj = nn.Linear(z_dim, 1)

        self.cvca = InfoNCELoss(temperature=0.07)
        self._z_dim = z_dim

    def forward(self, h_d, h_p, moa_d, moa_p, structural_feats,
                d_views=None, p_views=None, unique_d=None, unique_p=None):
        h_d_star = self.drug_fusion(h_d, moa_d)
        h_p_star = self.prot_fusion(h_p, moa_p)

        z = torch.cat([h_d_star, h_p_star, structural_feats], dim=-1)
        z_proj = self.input_proj(z)

        if HAS_KAN:
            out = torch.sigmoid(self.classifier(z_proj))
        else:
            h = z_proj
            for blk in self.res_blocks:
                h = blk(h)
            out = torch.sigmoid(self.classifier(h) + self.skip_proj(z))

        loss_cvca_d, loss_cvca_p = torch.tensor(0.0), torch.tensor(0.0)
        if self.training and d_views is not None and p_views is not None:
            loss_cvca_d = self.cvca(d_views, unique_d)
            loss_cvca_p = self.cvca(p_views, unique_p)

        return out.squeeze(-1), loss_cvca_d, loss_cvca_p