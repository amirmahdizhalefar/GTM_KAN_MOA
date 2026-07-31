"""
data_loader.py ── Raw network matrices + Real Sequence Loaders
=======================================================================
GTM is applied DIRECTLY to the 7 raw biological network matrices.
SMILES  : fetched from PubChem REST API using DrugBank drug names.
FASTA   : fetched from UniProt REST API using UniProt protein IDs.
Both are cached on disk (smiles_cache.json / fasta_cache.json) so the
network is hit only once per machine.

Dataset mappings are resolved automatically:
  • Luo  dataset (708 drugs, 1512 proteins) → DTINet DrugBank / UniProt IDs
  • New  dataset (151 drugs,  285 proteins) → subset of the same IDs
"""

import os
import json
import pickle
import time
import requests
import numpy as np

# ── Repo-root detection ──────────────────────────────────────────────────────
def _find_repo_root() -> str:
    """Walk upward from __file__ (or cwd) until we find data/sevenNets or
    new_datasets, whichever is present first."""
    anchors = ["data/sevenNets", "new_datasets"]
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = os.path.abspath(os.getcwd())
    path = here
    for _ in range(8):
        for anchor in anchors:
            if os.path.isdir(os.path.join(path, anchor)):
                return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return here

_ROOT        = _find_repo_root()
DATA_DIR     = os.path.join(_ROOT, "data")          # ddi.txt, ppi.txt
DATA_SEVEN   = os.path.join(_ROOT, "data", "sevenNets")  # mat_* (Luo)
DATA_NEW     = os.path.join(_ROOT, "new_datasets")  # mat_* (new dataset)
DATA_DTI     = os.path.join(_ROOT, "dti")           # fold splits + labels
OUT_DIR      = os.path.join(_ROOT, "gtmnet", "results")
CACHE_DIR    = os.path.join(_ROOT, "gtmnet", "seq_cache")

os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Canonical ID lists (downloaded once from DTINet GitHub at import time) ───
# These lists define the row-order for every matrix in BOTH datasets:
#   Luo  → LUO_DRUG_IDS[i]  is drug at matrix row i   (708  entries)
#   Luo  → LUO_PROT_IDS[j]  is protein at matrix col j (1512 entries)
#   new  → NEW_DRUG_IDS[i]  is a subset of LUO_DRUG_IDS (151  entries)
#   new  → NEW_PROT_IDS[j]  is a subset of LUO_PROT_IDS (285  entries)

_DTINET_BASE = "https://raw.githubusercontent.com/luoyunan/DTINet/master/data"

def _fetch_id_list(filename: str, local_path: str) -> list:
    """Return list of IDs from a local file, or download from DTINet GitHub."""
    if os.path.isfile(local_path):
        return [x.strip() for x in open(local_path).read().splitlines() if x.strip()]
    print(f"  [IDs] Downloading {filename} from DTINet GitHub …")
    try:
        r = requests.get(f"{_DTINET_BASE}/{filename}", timeout=30)
        r.raise_for_status()
        ids = [x.strip() for x in r.text.splitlines() if x.strip()]
        with open(local_path, "w") as fh:
            fh.write(r.text)
        return ids
    except Exception as e:
        print(f"  [WARNING] Could not fetch {filename}: {e}")
        return []

def _fetch_name_map(filename: str, local_path: str) -> dict:
    """Return {DrugBank/UniProt_ID: common_name} mapping."""
    if os.path.isfile(local_path):
        m = {}
        for line in open(local_path):
            line = line.strip()
            if ":" in line:
                k, v = line.split(":", 1)
                m[k.strip()] = v.strip()
        return m
    print(f"  [Map] Downloading {filename} from DTINet GitHub …")
    try:
        r = requests.get(f"{_DTINET_BASE}/{filename}", timeout=30)
        r.raise_for_status()
        m = {}
        for line in r.text.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                m[k.strip()] = v.strip()
        with open(local_path, "w") as fh:
            fh.write(r.text)
        return m
    except Exception as e:
        print(f"  [WARNING] Could not fetch {filename}: {e}")
        return {}

# Load canonical ID lists (cached in CACHE_DIR)
LUO_DRUG_IDS  = _fetch_id_list("drug.txt",    os.path.join(CACHE_DIR, "luo_drug_ids.txt"))
LUO_PROT_IDS  = _fetch_id_list("protein.txt", os.path.join(CACHE_DIR, "luo_prot_ids.txt"))
LUO_DRUG_NAMES = _fetch_name_map("drug_dict_map.txt",    os.path.join(CACHE_DIR, "luo_drug_dict_map.txt"))
LUO_PROT_NAMES = _fetch_name_map("protein_dict_map.txt", os.path.join(CACHE_DIR, "luo_prot_dict_map.txt"))

# New dataset IDs come from the local new_datasets/ folder
def _read_local_ids(path: str) -> list:
    if os.path.isfile(path):
        return [x.strip() for x in open(path).read().splitlines() if x.strip()]
    return []

NEW_DRUG_IDS = _read_local_ids(os.path.join(DATA_NEW, "drug.txt"))
NEW_PROT_IDS = _read_local_ids(os.path.join(DATA_NEW, "protein.txt"))

def _read_local_map(path: str) -> dict:
    m = {}
    if os.path.isfile(path):
        for line in open(path):
            line = line.strip()
            if ":" in line:
                k, v = line.split(":", 1)
                m[k.strip()] = v.strip()
    return m

NEW_DRUG_NAMES = _read_local_map(os.path.join(DATA_NEW, "drug_dict_map.txt"))

# ── Matrix helpers ───────────────────────────────────────────────────────────
def _adj(edge_file: str, n: int) -> np.ndarray:
    """Build symmetric adjacency matrix from an edge-list file."""
    try:
        edges = np.loadtxt(edge_file, dtype=int)
        A = np.zeros((n, n), dtype=np.float32)
        for r, c in edges:
            if 0 <= r < n and 0 <= c < n:
                A[r, c] = A[c, r] = 1.0
        return A
    except FileNotFoundError:
        print(f"  [WARNING] Edge file not found: {edge_file}  (fallback: random sparse)")
        rng = np.random.RandomState(42)
        A = rng.choice([0.0, 1.0], size=(n, n), p=[0.95, 0.05]).astype(np.float32)
        return np.maximum(A, A.T)

def _txt(path: str, fallback_shape=None) -> np.ndarray:
    try:
        return np.loadtxt(path, dtype=np.float32)
    except FileNotFoundError:
        if fallback_shape is not None:
            print(f"  [WARNING] File not found: {path}  (fallback: random)")
            return np.random.rand(*fallback_shape).astype(np.float32)
        raise

def _tile(M: np.ndarray, n: int) -> np.ndarray:
    reps = int(np.ceil(n / M.shape[0]))
    return np.tile(M, (reps, 1))[:n].astype(np.float32)

# ── Dataset loaders ──────────────────────────────────────────────────────────
def load_luo_dataset(verbose: bool = True) -> dict:
    """
    Luo (DTINet) benchmark dataset.
      N_D = 708 drugs,  N_P = 1512 proteins.
    Matrices are loaded from data/sevenNets/ (requires sevenNets.rar to be
    extracted).  ddi.txt / ppi.txt are always available in data/.
    """
    N_D, N_P = 708, 1512
    if verbose:
        print("Loading Luo dataset (raw matrices, no pre-processing) …")

    S1 = _adj(os.path.join(DATA_DIR, "ddi.txt"), N_D)
    S4 = S1.copy()
    S5 = _adj(os.path.join(DATA_DIR, "ppi.txt"), N_P)
    S7 = S5.copy()
    S2 = _tile(_txt(os.path.join(DATA_SEVEN, "mat_drug_se.txt"),      (151, 4192)), N_D)
    S3 = _tile(_txt(os.path.join(DATA_SEVEN, "mat_drug_disease.txt"), (151, 5603)), N_D)
    S6 = _tile(_txt(os.path.join(DATA_SEVEN, "mat_protein_disease.txt"), (285, 5603)), N_P)

    # Labels: load from dti/positive_labels_node_1923.pkl
    pkl_path = os.path.join(DATA_DTI, "positive_labels_node_1923.pkl")
    Y = np.zeros((N_D, N_P), dtype=np.float32)
    try:
        with open(pkl_path, "rb") as fh:
            pos = pickle.load(fh)          # shape (1923, 2): col-1 = flat node id
        for ni in pos[:, 1]:
            d, p = int(ni) // N_P, int(ni) % N_P
            if d < N_D and p < N_P:
                Y[d, p] = 1.0
    except FileNotFoundError:
        print(f"  [WARNING] {pkl_path} not found – using random labels.")
        rng = np.random.RandomState(0)
        Y = rng.choice([0.0, 1.0], size=(N_D, N_P), p=[0.99, 0.01]).astype(np.float32)

    return dict(S1=S1, S2=S2, S3=S3, S4=S4, S5=S5, S6=S6, S7=S7,
               Y=Y, N_D=N_D, N_P=N_P,
               drug_ids=LUO_DRUG_IDS, prot_ids=LUO_PROT_IDS)

def load_new_dataset(verbose: bool = True) -> dict:
    """
    Newly-constructed benchmark dataset (BMC Biology).
      N_D = 151 drugs,  N_P = 285 proteins.
    All matrices are under new_datasets/.
    """
    N_D, N_P = 151, 285
    if verbose:
        print("Loading new dataset (raw matrices, no pre-processing) …")

    S1 = _txt(os.path.join(DATA_NEW, "mat_drug_drug.txt"),     (N_D, N_D))
    S4 = S1.copy()
    S5 = _txt(os.path.join(DATA_NEW, "mat_protein_protein.txt"), (N_P, N_P))
    S7 = S5.copy()
    S2 = _txt(os.path.join(DATA_NEW, "mat_drug_se.txt"),       (N_D, 4192))
    S3 = _txt(os.path.join(DATA_NEW, "mat_drug_disease.txt"),  (N_D, 5603))
    S6 = _txt(os.path.join(DATA_NEW, "mat_protein_disease.txt"), (N_P, 5603))
    Y  = _txt(os.path.join(DATA_NEW, "mat_drug_protein.txt"),  (N_D, N_P))

    return dict(S1=S1, S2=S2, S3=S3, S4=S4, S5=S5, S6=S6, S7=S7,
               Y=Y, N_D=N_D, N_P=N_P,
               drug_ids=NEW_DRUG_IDS, prot_ids=NEW_PROT_IDS)

# ── Fold loaders ─────────────────────────────────────────────────────────────
def load_luo_folds(n_folds: int = 5) -> list:
    """
    Load the 5-fold CV splits from dti/train_val_data_3846_5val{i}.pkl.
    Each fold is a 6-tuple; we use positions [3]=train_ids, [4]=val_ids.
    Positive/negative labels come from dti/positive_labels_node_1923.pkl.
    Falls back to random mock splits if files are missing.
    """
    folds = []
    try:
        pos_path = os.path.join(DATA_DTI, "positive_labels_node_1923.pkl")
        with open(pos_path, "rb") as fh:
            pos = pickle.load(fh)
        pos_node_ids = set(int(x) for x in pos[:, 1])

        for i in range(n_folds):
            fname = f"train_val_data_3846_5val{i}.pkl"
            fpath = os.path.join(DATA_DTI, fname)
            with open(fpath, "rb") as fh:
                fd = pickle.load(fh)
            tr_ids  = np.array(fd[3], dtype=np.int64)
            val_ids = np.array(fd[4], dtype=np.int64)
            tr_Y  = np.array([1 if int(n) in pos_node_ids else 0 for n in tr_ids],  dtype=np.float32)
            val_Y = np.array([1 if int(n) in pos_node_ids else 0 for n in val_ids], dtype=np.float32)
            folds.append(dict(train_ids=tr_ids, val_ids=val_ids,
                              train_Y=tr_Y, val_Y=val_Y))
        print(f"  Loaded {n_folds} real folds from {DATA_DTI}/")
    except FileNotFoundError as e:
        print(f"  [WARNING] Fold files not found ({e}). Using random mock splits.")
        rng = np.random.RandomState(99)
        for i in range(n_folds):
            tr  = rng.randint(0, 708 * 1512, size=3846)
            val = rng.randint(0, 708 * 1512, size=768)
            folds.append(dict(
                train_ids=tr, val_ids=val,
                train_Y=rng.choice([0, 1], size=len(tr),  p=[0.8, 0.2]).astype(np.float32),
                val_Y  =rng.choice([0, 1], size=len(val), p=[0.8, 0.2]).astype(np.float32),
            ))
    return folds

# ════════════════════════════════════════════════════════════════════════════
# SMILES fetching – multi-source cascade
# ════════════════════════════════════════════════════════════════════════════
_SMILES_CACHE_PATH = os.path.join(CACHE_DIR, "smiles_cache.json")
_FALLBACK_SMILES   = "CC(=O)Oc1ccccc1C(=O)O"   # Aspirin

def _load_smiles_cache() -> dict:
    if os.path.isfile(_SMILES_CACHE_PATH):
        with open(_SMILES_CACHE_PATH) as fh:
            return json.load(fh)
    return {}

def _save_smiles_cache(cache: dict) -> None:
    with open(_SMILES_CACHE_PATH, "w") as fh:
        json.dump(cache, fh, indent=2)

def _safe_get(session: requests.Session, url: str, timeout: int = 15,
              retries: int = 2, headers: "dict | None" = None) -> "requests.Response | None":
    """GET with exponential back-off on 429 / 503 and silent exception catch.
    Optional *headers* dict is merged on top of the session headers."""
    h = dict(session.headers)
    if headers:
        h.update(headers)
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=timeout, headers=h)
            if r.status_code in (429, 503):
                time.sleep(3 * (attempt + 1))
                continue
            return r
        except Exception:
            time.sleep(1)
    return None

# ── Source 1: PubChem ────────────────────────────────────────────────────────
def _fetch_smiles_pubchem(query: str, session: requests.Session) -> str | None:
    """
    PubChem: synonym → CID → CanonicalSMILES, then direct name → property.
    Works for DrugBank IDs registered as PubChem synonyms and for common names.
    """
    import urllib.parse
    enc = urllib.parse.quote(query, safe="")
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    # Strategy A: synonym search → CID → SMILES
    r = _safe_get(session, f"{base}/compound/name/{enc}/cids/JSON")
    if r is not None and r.status_code == 200:
        try:
            cid = r.json()["IdentifierList"]["CID"][0]
            r2 = _safe_get(session, f"{base}/compound/cid/{cid}/property/CanonicalSMILES/TXT")
            if r2 is not None and r2.status_code == 200:
                smi = r2.text.strip().splitlines()[0].strip()
                if smi:
                    return smi
        except Exception:
            pass

    # Strategy B: direct name → IsomericSMILES / CanonicalSMILES
    for prop, fmt in [("IsomericSMILES", "TXT"), ("CanonicalSMILES", "TXT")]:
        r = _safe_get(session, f"{base}/compound/name/{enc}/property/{prop}/{fmt}")
        if r is not None and r.status_code == 200:
            smi = r.text.strip().splitlines()[0].strip()
            if smi and not smi.startswith("Status"):
                return smi
    return None

# ── Source 2: UniChem (EBI) – maps DrugBank IDs across databases ─────────────
# UniChem source IDs: 2 = DrugBank, 22 = PubChem
def _fetch_smiles_unichem(drugbank_id: str, session: requests.Session) -> str | None:
    """
    UniChem cross-reference: DrugBank ID → PubChem CID → SMILES.
    UniChem is the most reliable database-to-database ID mapper.
    """
    try:
        # REST API: get all cross-references for this DrugBank compound
        url = f"https://www.ebi.ac.uk/unichem/rest/src_compound_id/{drugbank_id}/2"
        r = _safe_get(session, url, timeout=20)
        if r is None or r.status_code != 200:
            return None
        refs = r.json()   # list of {src_id, src_compound_id}
        # Find a PubChem CID (src_id == "22")
        for ref in refs:
            if str(ref.get("src_id")) == "22":
                cid = ref["src_compound_id"]
                base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
                r2 = _safe_get(session, f"{base}/compound/cid/{cid}/property/CanonicalSMILES/TXT")
                if r2 is not None and r2.status_code == 200:
                    smi = r2.text.strip().splitlines()[0].strip()
                    if smi:
                        return smi
    except Exception:
        pass
    return None

# ── Source 3: ChEMBL ─────────────────────────────────────────────────────────
def _fetch_smiles_chembl(name: str, session: requests.Session) -> str | None:
    """
    ChEMBL molecule search by preferred name or synonym.
    Returns canonical SMILES or None.
    """
    try:
        import urllib.parse
        enc  = urllib.parse.quote(name, safe="")
        # Search by preferred name first
        url  = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?pref_name__iexact={enc}&limit=1"
        r    = _safe_get(session, url, timeout=20)
        if r is not None and r.status_code == 200:
            mols = r.json().get("molecules", [])
            if mols:
                smi = (mols[0].get("molecule_structures") or {}).get("canonical_smiles")
                if smi:
                    return smi
        # Synonym search (slower but catches trade names / INN)
        url2 = f"https://www.ebi.ac.uk/chembl/api/data/molecule_synonyms.json?synonyms__iexact={enc}&limit=1"
        r2   = _safe_get(session, url2, timeout=20)
        if r2 is not None and r2.status_code == 200:
            hits = r2.json().get("molecule_synonyms", [])
            if hits:
                chembl_id = hits[0].get("molecule_chembl_id")
                if chembl_id:
                    r3 = _safe_get(session, f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json", timeout=20)
                    if r3 is not None and r3.status_code == 200:
                        smi = (r3.json().get("molecule_structures") or {}).get("canonical_smiles")
                        if smi:
                            return smi
    except Exception:
        pass
    return None

# ── Source 4: ChEBI (EMBL-EBI) ───────────────────────────────────────────────
def _fetch_smiles_chebi(name: str, session: requests.Session) -> str | None:
    """
    ChEBI REST search by name → get InChI or SMILES.
    Useful for common small-molecule drugs that may not be in ChEMBL.
    """
    try:
        import urllib.parse
        enc = urllib.parse.quote(name, safe="")
        url = f"https://www.ebi.ac.uk/webservices/chebi/2.0/test/getCompleteEntityBySearch?searchString={enc}&maximumResults=1&starsInput=3"
        r   = _safe_get(session, url, timeout=20)
        if r is None or r.status_code != 200:
            return None
        # ChEBI returns XML; extract smiles from <smiles> tag if present
        text = r.text
        import re
        m = re.search(r"<smiles>(.+?)</smiles>", text)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return None

# ── Source 5: Open Chemistry (CIR / NCI resolver) ────────────────────────────
def _fetch_smiles_cir(name: str, session: requests.Session) -> str | None:
    """
    NCI Chemical Identifier Resolver – broad synonym coverage.
    https://cactus.nci.nih.gov/chemical/structure/{name}/smiles
    """
    try:
        import urllib.parse
        enc = urllib.parse.quote(name, safe="")
        url = f"https://cactus.nci.nih.gov/chemical/structure/{enc}/smiles"
        r   = _safe_get(session, url, timeout=15)
        if r is not None and r.status_code == 200:
            smi = r.text.strip().splitlines()[0].strip()
            if smi and not smi.lower().startswith("<!"):
                return smi
    except Exception:
        pass
    return None

def _fetch_smiles_all(drugbank_id: str, name: str | None,
                      session: requests.Session) -> str | None:
    """
    Try every source in priority order; return first successful SMILES.
    Sources:
      1. PubChem  (DrugBank ID as synonym)
      2. UniChem  (DrugBank → PubChem CID)
      3. PubChem  (common drug name)
      4. ChEMBL   (common drug name)
      5. CIR/NCI  (common drug name)
      6. ChEBI    (common drug name)
    """
    smi = _fetch_smiles_pubchem(drugbank_id, session)
    if smi:
        return smi

    smi = _fetch_smiles_unichem(drugbank_id, session)
    if smi:
        return smi

    if name:
        smi = _fetch_smiles_pubchem(name, session)
        if smi:
            return smi

        smi = _fetch_smiles_chembl(name, session)
        if smi:
            return smi

        smi = _fetch_smiles_cir(name, session)
        if smi:
            return smi

        smi = _fetch_smiles_chebi(name, session)
        if smi:
            return smi

    return None

def _check_reachability(session: requests.Session, urls: list[str],
                        label: str, timeout: int = 10) -> bool:
    """Return True if at least one URL in the list responds with a 2xx/4xx."""
    for url in urls:
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code < 500:
                return True
        except Exception:
            pass
    print(f"  [WARNING] {label} appears unreachable – check firewall / VPN.")
    return False

def load_smiles(drug_ids: list, drug_name_map: dict, verbose: bool = True) -> list:
    """
    Return SMILES strings for every drug, one per matrix row.

    Multi-source cascade (tried in order until a valid SMILES is found):
      1. Disk cache  (smiles_cache.json)  – stale Aspirin entries are purged
      2. PubChem REST  – DrugBank ID as synonym
      3. UniChem EBI   – DrugBank ID → PubChem CID cross-reference
      4. PubChem REST  – common drug name
      5. ChEMBL REST   – common drug name (preferred name + synonym search)
      6. NCI/CIR       – common drug name
      7. ChEBI REST    – common drug name
      8. Aspirin fallback (logged, NOT written to cache so it is retried next run)
    """
    cache   = _load_smiles_cache()
    session = requests.Session()
    session.headers.update({"User-Agent": "GTM-KAN-MoA/1.0 (research)"})

    # Purge stale Aspirin-fallback entries so they are retried.
    stale = [k for k, v in cache.items() if v == _FALLBACK_SMILES]
    if stale:
        if verbose:
            print(f"  [SMILES] Clearing {len(stale)} stale fallback entries from cache …")
        for k in stale:
            del cache[k]

    missing_ids = [did for did in drug_ids if did not in cache]

    if not missing_ids:
        if verbose:
            print(f"  [SMILES] All {len(drug_ids)} drugs loaded from cache.")
        return [cache.get(did, _FALLBACK_SMILES) for did in drug_ids]

    # Reachability pre-check (probe multiple services so we know what's up)
    pubchem_ok = _check_reachability(session,
        ["https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/property/CanonicalSMILES/TXT"],
        "PubChem")
    unichem_ok = _check_reachability(session,
        ["https://www.ebi.ac.uk/unichem/rest/src_compound_id/DB00001/2"],
        "UniChem (EBI)")
    chembl_ok  = _check_reachability(session,
        ["https://www.ebi.ac.uk/chembl/api/data/molecule/CHEMBL25.json"],
        "ChEMBL")
    cir_ok     = _check_reachability(session,
        ["https://cactus.nci.nih.gov/chemical/structure/aspirin/smiles"],
        "NCI CIR")
    chebi_ok   = _check_reachability(session,
        ["https://www.ebi.ac.uk/webservices/chebi/2.0/test/getCompleteEntityBySearch?searchString=aspirin&maximumResults=1"],
        "ChEBI")

    sources_up = [s for s, ok in [("PubChem", pubchem_ok), ("UniChem", unichem_ok),
                                   ("ChEMBL", chembl_ok), ("CIR", cir_ok),
                                   ("ChEBI", chebi_ok)] if ok]
    if verbose:
        print(f"  [SMILES] Active sources: {', '.join(sources_up) or 'NONE – all fallback'}")
        print(f"  [SMILES] Fetching {len(missing_ids)} drugs "
              f"({len(drug_ids) - len(missing_ids)} already cached) …")

    # Track per-source hit counts for the summary line.
    source_hits: dict[str, int] = {}

    for idx, did in enumerate(missing_ids):
        name = drug_name_map.get(did)
        smi  = None
        src  = None

        # --- PubChem by DrugBank ID ---
        if pubchem_ok:
            s = _fetch_smiles_pubchem(did, session)
            if s:
                smi, src = s, "PubChem(ID)"

        # --- UniChem cross-ref (DrugBank → PubChem CID) ---
        if smi is None and unichem_ok:
            s = _fetch_smiles_unichem(did, session)
            if s:
                smi, src = s, "UniChem"

        # --- PubChem by common name ---
        if smi is None and pubchem_ok and name:
            s = _fetch_smiles_pubchem(name, session)
            if s:
                smi, src = s, "PubChem(name)"

        # --- ChEMBL by common name ---
        if smi is None and chembl_ok and name:
            s = _fetch_smiles_chembl(name, session)
            if s:
                smi, src = s, "ChEMBL"

        # --- NCI/CIR by common name ---
        if smi is None and cir_ok and name:
            s = _fetch_smiles_cir(name, session)
            if s:
                smi, src = s, "CIR"

        # --- ChEBI by common name ---
        if smi is None and chebi_ok and name:
            s = _fetch_smiles_chebi(name, session)
            if s:
                smi, src = s, "ChEBI"

        if smi:
            cache[did] = smi
            source_hits[src] = source_hits.get(src, 0) + 1
        else:
            if verbose:
                print(f"    [WARNING] No SMILES found for {did} ({name}). Using Aspirin fallback.")
            smi = _FALLBACK_SMILES
            # Do NOT write fallback to cache so it is retried next run.

        time.sleep(0.22)   # ~4.5 req/s across all services combined
        if verbose and (idx + 1) % 50 == 0:
            print(f"    … {idx + 1}/{len(missing_ids)} done")

    _save_smiles_cache(cache)
    if verbose:
        n_real = sum(1 for did in drug_ids if cache.get(did, _FALLBACK_SMILES) != _FALLBACK_SMILES)
        hit_summary = "  ".join(f"{s}:{n}" for s, n in sorted(source_hits.items()))
        print(f"  [SMILES] {n_real}/{len(drug_ids)} real SMILES obtained  [{hit_summary}]")
        print(f"           Cache saved → {_SMILES_CACHE_PATH}")

    return [cache.get(did, _FALLBACK_SMILES) for did in drug_ids]


# ════════════════════════════════════════════════════════════════════════════
# FASTA fetching – multi-source cascade
# ════════════════════════════════════════════════════════════════════════════
_FASTA_CACHE_PATH = os.path.join(CACHE_DIR, "fasta_cache.json")
_FALLBACK_SEQ     = "MAAARPGM"

def _load_fasta_cache() -> dict:
    if os.path.isfile(_FASTA_CACHE_PATH):
        with open(_FASTA_CACHE_PATH) as fh:
            return json.load(fh)
    return {}

def _save_fasta_cache(cache: dict) -> None:
    with open(_FASTA_CACHE_PATH, "w") as fh:
        json.dump(cache, fh, indent=2)

def _parse_fasta_sequence(fasta_text: str) -> str:
    """Strip FASTA header lines and return the raw amino-acid sequence."""
    lines = [l.strip() for l in fasta_text.splitlines() if l.strip()]
    seq   = "".join(l for l in lines if not l.startswith(">"))
    return seq if seq else ""

# ── Source 1 & 2: UniProt (new + legacy endpoints) ───────────────────────────
def _fetch_fasta_uniprot(uid: str, session: requests.Session) -> str | None:
    for tmpl in [
        "https://rest.uniprot.org/uniprotkb/{uid}.fasta",
        "https://www.uniprot.org/uniprot/{uid}.fasta",
        "https://rest.uniprot.org/uniprotkb/{uid}?format=fasta",
    ]:
        try:
            r = _safe_get(session, tmpl.format(uid=uid), timeout=20)
            if r is not None and r.status_code == 200 and ">" in r.text:
                seq = _parse_fasta_sequence(r.text)
                if seq:
                    return seq
        except Exception:
            pass
    return None

# ── Source 3: EBI Proteins API ───────────────────────────────────────────────
def _fetch_fasta_ebi(uid: str, session: requests.Session) -> str | None:
    """EBI Proteins REST API – returns JSON with sequence field."""
    try:
        url = f"https://www.ebi.ac.uk/proteins/api/proteins/{uid}"
        r   = _safe_get(session, url, timeout=20,
                        headers={"Accept": "application/json"})
        if r is not None and r.status_code == 200:
            seq = r.json().get("sequence", {}).get("sequence", "")
            if seq:
                return seq
    except Exception:
        pass
    return None

# ── Source 4: NCBI eutils ────────────────────────────────────────────────────
def _fetch_fasta_ncbi(uid: str, session: requests.Session) -> str | None:
    """
    NCBI eutils: search for UniProt accession in the protein database,
    then fetch the FASTA sequence.
    """
    try:
        # Step 1: esearch – find GI/accession from UniProt ID
        search_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=protein&term={uid}[accn]&retmax=1&retmode=json&tool=gtmnet&email=research@gtmnet"
        )
        r = _safe_get(session, search_url, timeout=15)
        if r is None or r.status_code != 200:
            return None
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return None
        gi = ids[0]

        # Step 2: efetch – retrieve FASTA
        fetch_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db=protein&id={gi}&rettype=fasta&retmode=text&tool=gtmnet&email=research@gtmnet"
        )
        r2 = _safe_get(session, fetch_url, timeout=20)
        if r2 is not None and r2.status_code == 200 and ">" in r2.text:
            seq = _parse_fasta_sequence(r2.text)
            if seq:
                return seq
    except Exception:
        pass
    return None

# ── Source 5: ExPASy SIB (Swiss-Prot direct) ────────────────────────────────
def _fetch_fasta_expasy(uid: str, session: requests.Session) -> str | None:
    """
    ExPASy/SIB direct FASTA fetch – a separate mirror of Swiss-Prot.
    """
    try:
        url = f"https://www.expasy.org/uniprot/{uid}.fasta"
        r   = _safe_get(session, url, timeout=20)
        if r is not None and r.status_code == 200 and ">" in r.text:
            seq = _parse_fasta_sequence(r.text)
            if seq:
                return seq
    except Exception:
        pass
    return None

def _fetch_fasta_all(uid: str, session: requests.Session) -> "tuple[str | None, str]":
    """
    Try every FASTA source; return (sequence, source_label) or (None, '').
    Sources tried in order:
      1. UniProt REST v2   (rest.uniprot.org)
      2. UniProt legacy    (www.uniprot.org)
      3. EBI Proteins API  (www.ebi.ac.uk/proteins)
      4. NCBI eutils       (esearch + efetch)
      5. ExPASy SIB        (www.expasy.org)
    """
    seq = _fetch_fasta_uniprot(uid, session)
    if seq:
        return seq, "UniProt"

    seq = _fetch_fasta_ebi(uid, session)
    if seq:
        return seq, "EBI-Proteins"

    seq = _fetch_fasta_ncbi(uid, session)
    if seq:
        return seq, "NCBI"

    seq = _fetch_fasta_expasy(uid, session)
    if seq:
        return seq, "ExPASy"

    return None, ""

def load_fasta(prot_ids: list, verbose: bool = True) -> list:
    """
    Return amino-acid sequence strings for every protein, one per matrix column.

    Multi-source cascade (tried in order):
      1. Disk cache       (fasta_cache.json) – stale stubs purged and retried
      2. UniProt REST v2  (rest.uniprot.org)
      3. UniProt legacy   (www.uniprot.org)
      4. EBI Proteins API (www.ebi.ac.uk/proteins)
      5. NCBI eutils      (esearch protein + efetch fasta)
      6. ExPASy SIB       (www.expasy.org)
      7. Stub fallback    'MAAARPGM' – logged, NOT written to cache
    """
    cache   = _load_fasta_cache()
    session = requests.Session()
    session.headers.update({"User-Agent": "GTM-KAN-MoA/1.0 (research)"})

    # Purge stale stub entries so they are retried this run.
    stale = [k for k, v in cache.items() if v == _FALLBACK_SEQ]
    if stale:
        if verbose:
            print(f"  [FASTA] Clearing {len(stale)} stale stub entries from cache …")
        for k in stale:
            del cache[k]

    missing_ids = [pid for pid in prot_ids if pid not in cache]

    if not missing_ids:
        if verbose:
            print(f"  [FASTA] All {len(prot_ids)} proteins loaded from cache.")
        return [cache.get(pid, _FALLBACK_SEQ) for pid in prot_ids]

    if verbose:
        print(f"  [FASTA] Fetching {len(missing_ids)} proteins "
              f"({len(prot_ids) - len(missing_ids)} already cached) …")

    source_hits: dict[str, int] = {}

    for idx, pid in enumerate(missing_ids):
        seq, src = _fetch_fasta_all(pid, session)

        if seq:
            cache[pid] = seq
            source_hits[src] = source_hits.get(src, 0) + 1
        else:
            if verbose:
                print(f"    [WARNING] No FASTA for {pid} from any source. Using stub.")
            # Do NOT write stub to cache so it is retried next run.

        time.sleep(0.2)   # ≈ 5 req/s
        if verbose and (idx + 1) % 100 == 0:
            hit_so_far = sum(source_hits.values())
            print(f"    … {idx + 1}/{len(missing_ids)} done  ({hit_so_far} real so far)")

    _save_fasta_cache(cache)
    if verbose:
        n_real    = sum(1 for pid in prot_ids if cache.get(pid, _FALLBACK_SEQ) != _FALLBACK_SEQ)
        hit_summary = "  ".join(f"{s}:{n}" for s, n in sorted(source_hits.items()))
        print(f"  [FASTA] {n_real}/{len(prot_ids)} real sequences obtained  [{hit_summary}]")
        print(f"          Cache saved → {_FASTA_CACHE_PATH}")

    return [cache.get(pid, _FALLBACK_SEQ) for pid in prot_ids]

# ── Convenience wrappers (called from GTMNet.fit) ────────────────────────────
def load_smiles_for_dataset(data: dict, verbose: bool = True) -> list:
    """
    Auto-selects the correct drug-ID list and name map based on
    whichever dataset dict was loaded (Luo or new).
    """
    drug_ids = data.get("drug_ids", [])
    if not drug_ids:
        raise ValueError("data dict is missing 'drug_ids'. "
                         "Use load_luo_dataset() or load_new_dataset().")
    # Determine which name map to use
    if len(drug_ids) == 708:        # Luo
        name_map = {**LUO_DRUG_NAMES, **NEW_DRUG_NAMES}
    else:                           # new dataset (also covered by NEW_DRUG_NAMES)
        name_map = {**NEW_DRUG_NAMES, **LUO_DRUG_NAMES}
    return load_smiles(drug_ids, name_map, verbose=verbose)

def load_fasta_for_dataset(data: dict, verbose: bool = True) -> list:
    """
    Auto-selects the correct protein-ID list based on the dataset dict.
    """
    prot_ids = data.get("prot_ids", [])
    if not prot_ids:
        raise ValueError("data dict is missing 'prot_ids'. "
                         "Use load_luo_dataset() or load_new_dataset().")
    return load_fasta(prot_ids, verbose=verbose)