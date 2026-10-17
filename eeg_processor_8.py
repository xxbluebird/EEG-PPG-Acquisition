import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, welch, iirnotch, hilbert, find_peaks
from scipy.linalg import eigh
from scipy.ndimage import convolve1d
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis
import mne


# ============================================================
# 1. HELPER FUNCTIONS (FILTERS & ZAPLINE)
# ============================================================
# ============================================================
# MAICA SPLIT CONFIG — GUI lama
# ============================================================
MAICA_SPLIT_8CH_ENABLED = True
MAICA_MAX_COMPONENTS_PER_4CH_GROUP = 12
MAICA_MAX_FILTERS_PER_GROUP = 2

# Urutan channel GUI lama:
# 0=Fp1, 1=Fp2, 2=F3, 3=F4, 4=T3, 5=T4, 6=P3, 7=P4
MAICA_PHYSICAL_IDX = [0, 1, 4, 5]  # Fp1, Fp2, T3, T4
MAICA_VIRTUAL_GT_IDX = [2, 3, 6, 7]  # F3, F4, P3, P4

MAICA_PHYSICAL_NAMES = ["Fp1", "Fp2", "T3", "T4"]
MAICA_VIRTUAL_GT_NAMES = ["F3", "F4", "P3", "P4"]

def bandpass_filter(x, fs, low=1.0, high=40.0, order=4):
    """Bandpass butterworth, x shape: (n_ch, n_samp)"""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, x, axis=-1)

def bandpass_band(x, fs, low, high, order=4):
    """Bandpass untuk satu band spesifik, x shape: (n_ch, n_samp)"""
    nyq = fs / 2.0
    low  = max(low,  0.5)
    high = min(high, nyq - 0.5)
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, x, axis=-1)

def post_notch(x, fs, freq, Q=50):
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, x, axis=-1)

def zapline_strong_auto_plus1(data, fs, fline=50, width=1.5):
    n_ch, n_time = data.shape
    harmonics = [fline, fline * 2]
    ref = data.copy()
    for f in harmonics:
        if f < fs / 2:
            q = f / width
            b, a = iirnotch(f, q, fs)
            ref = filtfilt(b, a, ref, axis=-1)
    
    noise_ref = data - ref
    C1 = (noise_ref @ noise_ref.T) / n_time
    C0 = (data @ data.T) / n_time
    C0 += np.eye(n_ch) * (1e-6 * np.trace(C0) / n_ch)
    
    vals, vecs = eigh(C1, C0)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    
    threshold = np.mean(vals) + np.std(vals)
    n_auto = np.sum(vals > threshold)
    nremove = min(n_auto, max(1, n_ch // 2))
    
    W = vecs[:, :nremove]
    S = W.T @ data
    T = (data @ S.T) @ np.linalg.inv(S @ S.T + np.eye(nremove) * 1e-9)
    return data - T @ S, nremove

# ============================================================
# 2. HELPER FUNCTIONS (ICA & MIF)
# ============================================================

def create_virtual_channel(x, N=2):
    b = np.ones(N) / N
    return filtfilt(b, 1, x)

def get_extrema_count(x):
    peaks, _ = signal.find_peaks(x)
    troughs, _ = signal.find_peaks(-x)
    return len(peaks) + len(troughs)

def double_average_filter(x, L):
    if L < 3: return x
    kernel = np.ones(L) / L
    y = convolve1d(x, kernel, mode='reflect')
    y = convolve1d(y, kernel, mode='reflect')
    return y
def run_mif(data, alpha=1.6, max_imfs=10, sift_iter=8):
    n_ch, N = data.shape
    residual = data.copy()
    mimfs = []
    
    for k in range(max_imfs):
        imf = residual.copy()
        
        # 1. PERBAIKAN: Hitung E_max & L (Sesuai Paper: 2L+1)
        E_max = max([get_extrema_count(imf[c]) for c in range(n_ch)])
        if E_max <= 2: break
        
        # Hitung L dasar
        L_val = int(np.floor(alpha * N / (E_max + 1)))
        
        # --- PERBAIKAN: Gamma Protection untuk MIMF 1 ---
        if k == 0:
            L_val = min(L_val, 11) # Menjamin MIMF1 menangkap Gamma/50Hz
        
        # Panjang filter sesuai paper adalah 2L+1
        filter_len = 2 * L_val + 1
        filter_len = max(3, min(filter_len, N // 2))

        # Sifting Loop
        for _ in range(sift_iter):
            imf_old = imf.copy()
            trend = np.zeros_like(imf)
            for c in range(n_ch): 
                trend[c] = double_average_filter(imf[c], filter_len)
            
            imf = imf - trend
            
            # 2. PERBAIKAN: Cek Konvergensi Sifting (Agar stabil)
            norm_diff = np.linalg.norm(imf - imf_old) / (np.linalg.norm(imf_old) + 1e-12)
            if norm_diff < 0.01:
                break

        # 3. PERBAIKAN: Stop Condition Multivariate
        # Harus cek SEMUA channel, bukan cuma channel [0]
        if max([get_extrema_count(imf[c]) for c in range(n_ch)]) <= 2: 
            break
            
        mimfs.append(imf)
        residual = residual - imf
        
        # Stop jika energi sisa sudah sangat kecil
        if np.std(residual) < 0.05 * np.std(data): 
            break
            
    mimfs.append(residual)
    return np.array(mimfs)

# ============================================================
# 3. HELPER FUNCTIONS (RHYTHM & POWER)
# ============================================================

def get_robust_mf(imf, fs):
    analytic = hilbert(imf) 
    amp = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    inst_f = np.diff(phase) / (2*np.pi) * fs
    inst_f = np.insert(inst_f, 0, inst_f[0])
    mask = amp > 0.1 * np.max(amp)
    return np.median(inst_f[mask]) if np.any(mask) else 0.0

def compute_total_power(x, fs, f_low, f_high):
    f, Pxx = welch(x, fs=fs, nperseg=int(fs*2))
    mask = (f >= f_low) & (f <= f_high)
    return np.trapz(Pxx[mask], f[mask]) if np.any(mask) else 1e-10

# ============================================================
# 4. COMPREHENSIVE FEATURE EXTRACTION
# ============================================================

def extract_features_comprehensive(rhythm_signals, band_names, band_ranges, phase_signals, fs):
    eps = 1e-10
    feature_dict = {}
    n_ch = rhythm_signals.shape[1]
    
    # Absolute & Relative Power
    abs_power = {band: [] for band in band_names} 
    rel_power = {band: [] for band in band_names}
    for b_idx, band in enumerate(band_names):
        lo, hi = band_ranges[band]
        if lo is None: 
            abs_power[band] = [0.0] * n_ch
            continue
        abs_power[band] = []
        for ch in range(n_ch):
            p = compute_total_power(rhythm_signals[b_idx, ch], fs, lo, hi)
            abs_power[band].append(p)
            feature_dict[f"Abs_{band}_ch{ch}"] = p

    for ch in range(n_ch):
        total = sum(abs_power[b][ch] for b in abs_power)
        for band in abs_power:
            rel_power.setdefault(band, [])
            val = abs_power[band][ch] / (total + eps)
            rel_power[band].append(val)
            feature_dict[f"Rel_{band}_ch{ch}"] = val

    # Ratios
    theta_key = next((b for b in abs_power if b.lower()=="theta"), None)
    alpha_key = next((b for b in abs_power if b.lower()=="alpha"), None)
    beta_key  = next((b for b in abs_power if b.lower()=="beta"), None)

    for ch in range(n_ch):
        if theta_key and alpha_key:
            feature_dict[f"Ratio_theta_alpha_ch{ch}"] = abs_power[theta_key][ch] / (abs_power[alpha_key][ch] + eps)
        if alpha_key and beta_key:
            feature_dict[f"Ratio_alpha_beta_ch{ch}"] = abs_power[alpha_key][ch] / (abs_power[beta_key][ch] + eps)

    # Asymmetry (Pairs: 0-1, 2-3, 4-5)
    pairs = [(0,1), (2,3), (4,5), (6,7)]

    valid_pairs = [(L, R) for L, R in pairs if L < n_ch and R < n_ch]

    for band in abs_power:
        for L, R in valid_pairs:
            val = (abs_power[band][L] - abs_power[band][R]) / (
                abs_power[band][L] + abs_power[band][R] + eps
            )
            feature_dict[f"Asym_{band}_ch{L}_{R}"] = val
            
    # SASI & Phase Connectivity
    for ch in range(n_ch):
        if beta_key and alpha_key:
            feature_dict[f"SASI_ch{ch}"] = (abs_power[beta_key][ch] - abs_power[alpha_key][ch]) / (abs_power[beta_key][ch] + abs_power[alpha_key][ch] + eps)

    if alpha_key:
        alpha_idx = band_names.index(alpha_key)
        for L, R in valid_pairs:
            dtheta = phase_signals[alpha_idx, L] - phase_signals[alpha_idx, R]
            feature_dict[f"PhaseLag_ch{L}_{R}"] = np.mean(dtheta)
            feature_dict[f"PhaseCoh_ch{L}_{R}"] = np.abs(np.mean(np.exp(1j * dtheta)))
            
    return feature_dict, pd.DataFrame(list(feature_dict.items()), columns=['Feature Name', 'Value']), abs_power, rel_power

def reconstruct_from_ica(intermediates, k_thresh=1.6, bcg_thresh=0.065, vbr_thresh=6.0):
    """
    Rekonstruksi sinyal EEG bersih dari intermediate ICA yang sudah ada,
    menggunakan threshold kurtosis baru.

    Parameter:
        intermediates : dict dari 'ica_intermediates' hasil process_eeg_data()
        k_thresh      : threshold kurtosis (IC ditolak jika kurt >= k_thresh)

    Return:
        dict berisi 'ica_clean', 'ics_updated', 'bands', 'features'
    """
    if isinstance(intermediates, dict) and intermediates.get("split_mode", False):
        return reconstruct_from_split_maica(
            intermediates,
            k_thresh=k_thresh,
            bcg_thresh=bcg_thresh,
            vbr_thresh=vbr_thresh
        )
    S_ica    = intermediates['S_ica']
    scaler   = intermediates['scaler']
    ica_obj  = intermediates['ica_obj']
    n_ch     = intermediates['n_ch']
    L_bank   = intermediates['L_bank']
    fs       = intermediates['fs']

    S_clean_ica = S_ica.copy()
    rejected_indices = []

    # Ambil pulse_smooth dari intermediates jika tersedia
    pulse_smooth = intermediates.get('pulse_smooth', None)

    ica_list_new = []
    S_clean_new  = S_ica.copy()

    for i in range(S_ica.shape[1]):
        sig_ic = S_ica[:, i]
        k = kurtosis(sig_ic)

        f_ic, p_ic = welch(
            sig_ic,
            fs,
            nperseg=min(int(fs * 2), len(sig_ic) // 2)
        )

        # Hitung corr_pulse jika pulse_smooth tersedia
        corr_pulse = 0.0
        if pulse_smooth is not None:
            corr_matrix = np.corrcoef(pulse_smooth, sig_ic)
            corr_pulse = abs(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0

        # Hanya kurtosis sebagai penentu reject
        is_bcg = corr_pulse >= bcg_thresh
        # Tambah VBR di loop IC
        vbr = variance_burst_ratio(sig_ic, fs)
        is_emg = vbr > vbr_thresh   # vbr_thresh dikirim sebagai parameter baru

        is_rejected = (k >= k_thresh) or is_bcg or is_emg

        ica_list_new.append({
            'sig'       : sig_ic,
            'kurt'      : k,
            'corr_pulse': corr_pulse,
            'vbr'       : vbr,
            'rejected'  : is_rejected,
            'freqs'     : f_ic,
            'psd'       : p_ic
        })

        if is_rejected:
            S_clean_new[:, i] = 0
            rejected_indices.append(i)

    rejected_indices = sorted(list(set(rejected_indices)))
   
    # 3. Rekonstruksi sinyal EEG
    # Satukan logika rekonstruksi menggunakan inverse_transform bawaan Scikit-Learn
    # Ini menjamin inversi PCA internal tetap berjalan aman meskipun n_components < jumlah kolom augmented
    X_rec = scaler.inverse_transform(ica_obj.inverse_transform(S_clean_new))
    
    # Hitung jumlah komponen yang dihapus untuk dilempar ke output
    removed = np.any(S_clean_new != S_ica, axis=0).sum()
    
    orig_col_indices = intermediates.get('orig_col_indices',
                                         list(range(0, n_ch * (len(L_bank)+1), len(L_bank)+1)))
    ica_clean = np.array([X_rec[:, col_idx] for col_idx in orig_col_indices])
    # ica_clean shape: (n_ch, n_samp)

    # 4. Ekstraksi band + fitur (sama persis dengan process_eeg_data)
    EEG_BANDS = {
        'Delta': (0.5, 4.0),
        'Theta': (4.0, 8.0),
        'Alpha': (8.0, 13.0),
        'Beta' : (13.0, 30.0),
    }
    band_names  = list(EEG_BANDS.keys())
    n_ch_r      = ica_clean.shape[0]
    n_samp_r    = ica_clean.shape[1]

    rhythm_signals = np.zeros((len(band_names), n_ch_r, n_samp_r))
    band_ranges    = {}
    band_list      = []

    for b_idx, (band, (lo, hi)) in enumerate(EEG_BANDS.items()):
        filtered = bandpass_band(ica_clean, fs, lo, hi)
        rhythm_signals[b_idx] = filtered
        band_ranges[band]     = [lo, hi]
        sig_b  = filtered[0]
        f_b, p_b = welch(sig_b, fs, nperseg=min(int(fs * 2), len(sig_b)))
        band_list.append({'name': band, 'sig': sig_b, 'f': f_b, 'p': p_b,
                          'lo': lo, 'hi': hi})

    phase_signals = np.zeros_like(rhythm_signals)
    for b_idx in range(len(band_names)):
        for ch in range(n_ch_r):
            seg = rhythm_signals[b_idx, ch]
            if np.any(seg):
                phase_signals[b_idx, ch] = np.angle(hilbert(seg))

    feat_dict, feat_table, abs_p, rel_p = extract_features_comprehensive(
        rhythm_signals, band_names, band_ranges, phase_signals, fs
    )
    # =====================================================================
    # PERBAIKAN: Hitung SNR dengan merekonstruksi S_ica tanpa nol (Sinyal Asli)
    # =====================================================================
    # 1. Rekonstruksi seluruh matriks augmented sebelum ada komponen yang di-zero-out
    X_orig_full = scaler.inverse_transform(ica_obj.inverse_transform(S_ica))
    
    # 2. Ekstrak hanya 8 channel EEG fisik asli menggunakan orig_col_indices yang tersimpan
    X_original = np.array([X_orig_full[:, col_idx] for col_idx in orig_col_indices])
    
    # 3. Hitung SNR menggunakan sinyal asli rekonstruksi vs sinyal bersih
    snr_channels, snr_total = calculate_eeg_snr(X_original, ica_clean)
    # =====================================================================
    return {
        'ica_clean'   : ica_clean,
        'ics_updated' : ica_list_new,
        'bands': {
            'signals'  : rhythm_signals,
            'abs_power': abs_p,
            'rel_power': rel_p,
            'powers'   : {b: np.mean(rel_p[b]) if b in rel_p else 0.0 for b in band_names},
            'ranges'   : {b: tuple(v) for b, v in band_ranges.items()},
            'imf_map'  : {b: [] for b in band_names},
            'names'    : band_names
        },
        'features': {
            'table': feat_table,
            'dict' : feat_dict
        },
        'mif': {
            'mimfs': band_list
        },
        'n_rejected': int(removed),
        'rejected_indices': rejected_indices,
        'snr_per_channel': snr_channels,
        'avg_snr': snr_total
    }
    
def _recompute_one_maica_group_from_intermediate(
    group_inter,
    k_thresh=1.6,
    bcg_thresh=0.065,
    vbr_thresh=6.0
):
    """
    Recompute threshold untuk satu grup MAICA.
    """
    fs = group_inter["fs"]
    S_ica = group_inter["S_ica"]
    scaler = group_inter["scaler"]
    ica_obj = group_inter["ica_obj"]
    orig_col_indices = group_inter["orig_col_indices"]
    pulse_smooth = group_inter.get("pulse_smooth", None)
    group_name = group_inter.get("group_name", "MAICA Group")

    S_clean = S_ica.copy()
    rejected_indices = []
    ics = []

    for i in range(S_ica.shape[1]):
        sig_ic = S_ica[:, i]

        k_val = kurtosis(sig_ic)
        vbr_val = variance_burst_ratio(sig_ic, fs)
        is_emg = vbr_val >= vbr_thresh

        corr_pulse = 0.0
        is_bcg = False

        if pulse_smooth is not None:
            c_matrix = np.corrcoef(pulse_smooth, sig_ic)
            corr_pulse = abs(c_matrix[0, 1]) if not np.isnan(c_matrix[0, 1]) else 0.0
            is_bcg = corr_pulse >= bcg_thresh

        is_kurt = k_val >= k_thresh
        is_rejected = bool(is_kurt or is_bcg or is_emg)

        if is_rejected:
            S_clean[:, i] = 0.0
            rejected_indices.append(i)

        f_ic, p_ic = welch(
            sig_ic,
            fs,
            nperseg=min(int(fs * 2), max(8, len(sig_ic) // 2))
        )

        ics.append({
            "sig": sig_ic.copy(),
            "kurt": float(k_val),
            "corr_pulse": float(corr_pulse),
            "vbr": float(vbr_val),
            "freqs": f_ic,
            "psd": p_ic,
            "rejected": is_rejected,
            "group": group_name,
            "group_local_idx": i,
        })

    X_rec = scaler.inverse_transform(ica_obj.inverse_transform(S_clean))
    clean_group = np.array([X_rec[:, col_idx] for col_idx in orig_col_indices])

    return clean_group, ics, rejected_indices


def _finalize_clean_eeg_for_gui(ica_clean, fs, ch_names_orig=None):
    """
    Finalisasi setelah ICA:
    clean EEG -> band signals -> features -> mif data.
    Dipakai oleh reconstruct split agar outputnya sama dengan reconstruct lama.
    """
    EEG_BANDS_LOCAL = {
        "Delta": (0.5, 4.0),
        "Theta": (4.0, 8.0),
        "Alpha": (8.0, 13.0),
        "Beta":  (13.0, 30.0),
    }

    band_names = list(EEG_BANDS_LOCAL.keys())
    n_ch = ica_clean.shape[0]
    n_samp = ica_clean.shape[1]

    rhythm_signals = np.zeros((len(band_names), n_ch, n_samp))
    band_ranges = {}
    band_list = []

    for b_idx, (band, (lo, hi)) in enumerate(EEG_BANDS_LOCAL.items()):
        filtered = bandpass_band(ica_clean, fs, lo, hi)
        rhythm_signals[b_idx] = filtered
        band_ranges[band] = [lo, hi]

        sig_b = filtered[0]
        f_b, p_b = welch(
            sig_b,
            fs,
            nperseg=min(int(fs * 2), len(sig_b))
        )

        band_list.append({
            "name": band,
            "sig": sig_b,
            "f": f_b,
            "p": p_b,
            "lo": lo,
            "hi": hi
        })

    phase_signals = np.zeros_like(rhythm_signals)

    n_phase_ch = rhythm_signals.shape[1]

    for b_idx in range(len(band_names)):
        for ch in range(n_phase_ch):
            sig_segment = rhythm_signals[b_idx, ch]

            if np.any(sig_segment):
                phase_signals[b_idx, ch] = np.angle(hilbert(sig_segment))
            else:
                phase_signals[b_idx, ch] = 0.0
    feat_dict, feat_table, abs_p, rel_p = extract_features_comprehensive(
        rhythm_signals,
        band_names,
        band_ranges,
        phase_signals,
        fs
    )

    return {
        "bands": {
            "signals": rhythm_signals,
            "abs_power": abs_p,
            "rel_power": rel_p,
            "powers": {b: np.mean(rel_p[b]) if b in rel_p else 0.0 for b in band_names},
            "ranges": {b: tuple(v) for b, v in band_ranges.items()},
            "imf_map": {b: [] for b in band_names},
            "names": band_names
        },
        "features": {
            "table": feat_table,
            "dict": feat_dict
        },
        "mif": {
            "mimfs": band_list
        }
    }


def reconstruct_from_split_maica(
    intermediates,
    k_thresh=1.6,
    bcg_thresh=0.065,
    vbr_thresh=6.0
):
    """
    Recompute threshold untuk split MAICA.
    """
    fs = intermediates["fs"]
    groups = intermediates["groups"]

    phys_clean, phys_ics, phys_rej = _recompute_one_maica_group_from_intermediate(
        groups["physical"],
        k_thresh=k_thresh,
        bcg_thresh=bcg_thresh,
        vbr_thresh=vbr_thresh
    )

    virt_clean, virt_ics, virt_rej = _recompute_one_maica_group_from_intermediate(
        groups["virtual_gt"],
        k_thresh=k_thresh,
        bcg_thresh=bcg_thresh,
        vbr_thresh=vbr_thresh
    )

    ica_clean = np.zeros((8, phys_clean.shape[1]), dtype=float)

    ica_clean[MAICA_PHYSICAL_IDX, :] = phys_clean
    ica_clean[MAICA_VIRTUAL_GT_IDX, :] = virt_clean

    ica_list_new = []
    rejected_indices = []

    ics_split = {
        "physical": {
            "title": "Physical 4CH: Fp1, Fp2, T3, T4",
            "ics": [],
            "rejected_indices": [],
            "n_rejected": len(phys_rej),
            "n_total": len(phys_ics),
        },
        "virtual_gt": {
            "title": "GT Virtual 4CH: F3, F4, P3, P4",
            "ics": [],
            "rejected_indices": [],
            "n_rejected": len(virt_rej),
            "n_total": len(virt_ics),
        }
    }

    for local_idx, ic in enumerate(phys_ics):
        flat_idx = len(ica_list_new)
        ic2 = dict(ic)
        ic2["flat_idx"] = flat_idx
        ica_list_new.append(ic2)
        ics_split["physical"]["ics"].append(ic2)

        if local_idx in phys_rej:
            rejected_indices.append(flat_idx)
            ics_split["physical"]["rejected_indices"].append(local_idx)

    for local_idx, ic in enumerate(virt_ics):
        flat_idx = len(ica_list_new)
        ic2 = dict(ic)
        ic2["flat_idx"] = flat_idx
        ica_list_new.append(ic2)
        ics_split["virtual_gt"]["ics"].append(ic2)

        if local_idx in virt_rej:
            rejected_indices.append(flat_idx)
            ics_split["virtual_gt"]["rejected_indices"].append(local_idx)

    split_stats = {
        "physical_rejected": len(phys_rej),
        "physical_total": len(phys_ics),
        "physical_percent": 100.0 * len(phys_rej) / max(1, len(phys_ics)),
        "virtual_gt_rejected": len(virt_rej),
        "virtual_gt_total": len(virt_ics),
        "virtual_gt_percent": 100.0 * len(virt_rej) / max(1, len(virt_ics)),
        "total_rejected": len(rejected_indices),
        "total_ic": len(ica_list_new),
    }

    final = _finalize_clean_eeg_for_gui(ica_clean, fs)

    return {
        "ica_clean": ica_clean,
        "ics_updated": ica_list_new,
        "ics_split": ics_split,
        "split_stats": split_stats,
        "bands": final["bands"],
        "features": final["features"],
        "mif": final["mif"],
        "n_rejected": int(len(rejected_indices)),
        "rejected_indices": rejected_indices,
    }
    

def compute_maica_L(x1, fs, min_orig=0.6, max_orig=0.98, max_outputs=0.4):
    """
    Algoritma Figure 2 dari paper MAICA (Rejer & Górski, 2019).
    Mencari vector L (window sizes) secara adaptif berdasarkan korelasi.

    Returns
    -------
    L : list of int   — window sizes yang valid
    """
    from numpy import corrcoef

    def zero_phase_ma(x, N):
        """Zero-phase moving average (filtfilt style manual)."""
        b = np.ones(N) / N
        return filtfilt(b, [1.0], x)

    def pearson_abs(a, b):
        c = corrcoef(a, b)
        return abs(c[0, 1]) if not np.isnan(c[0, 1]) else 0.0

    L = []
    previous_input = None

    for N in range(2, len(x1) // 4):  # batas atas 
        new_input = zero_phase_ma(x1, N)

        # Stop condition 1: korelasi dengan sinyal asli terlalu rendah
        if pearson_abs(new_input, x1) < min_orig:
            break

        # Skip: terlalu mirip dengan sinyal asli atau previous input
        if pearson_abs(new_input, x1) > max_orig:
            continue
        if previous_input is not None and pearson_abs(new_input, previous_input) > max_orig:
            continue

        previous_input = new_input
        L.append(N)

        # Stop condition 2: jika sudah cukup banyak filter, cek redundansi
       # Untuk mode low-channel GUI, batasi 2 filter agar 4CH -> max sekitar 12 IC
        if len(L) >= 2:
            break

    return L if L else [2, 3] # fallback minimal jika tidak ada yang lolos

def _make_pulse_smooth_from_beats(beats, n_samples, fs):
    """
    Membuat referensi pulsa halus untuk deteksi BCG.
    """
    if beats is None or len(beats) <= 2:
        return None

    pulse_ref = np.zeros(n_samples)
    for bt in beats:
        if bt < n_samples:
            pulse_ref[int(bt)] = 1.0

    window_len = max(1, int(fs * 0.3))
    pulse_smooth = np.convolve(
        pulse_ref,
        np.ones(window_len) / window_len,
        mode="same"
    )

    return pulse_smooth


def _zero_phase_ma_local(x, N):
    b = np.ones(N) / N
    return filtfilt(b, [1.0], x)


def _build_maica_augmented_matrix(X_raw_data, fs, max_filters=2):
    """
    Bentuk matriks augmented MAICA untuk satu grup channel.
    Input:
        X_raw_data shape = (n_samples, n_ch_group)
    Return:
        X_aug, L_bank, orig_col_indices
    """
    n_samples, n_ch = X_raw_data.shape

    L_bank = compute_maica_L(
        X_raw_data[:, 0],
        fs,
        min_orig=0.6,
        max_orig=0.98,
        max_outputs=0.4
    )

    # Untuk 4CH, batasi agar max IC sekitar 12:
    # 4 channel x (1 original + 2 filter) = 12
    L_bank = list(L_bank)[:max_filters]

    print(f"[SPLIT MAICA] L_bank dipakai: {L_bank}")

    X_aug_list = []
    orig_col_indices = []

    for i in range(n_ch):
        orig = X_raw_data[:, i]
        orig_col_indices.append(len(X_aug_list))
        X_aug_list.append(orig)

        for N in L_bank:
            virt = _zero_phase_ma_local(orig, N)

            corr_ok = True
            for existing in X_aug_list:
                c = abs(np.corrcoef(virt, existing)[0, 1])
                if c > 0.98:
                    corr_ok = False
                    break

            if corr_ok:
                X_aug_list.append(virt)

    X_aug = np.column_stack(X_aug_list)

    return X_aug, L_bank, orig_col_indices


def _run_maica_one_group_for_gui(
    data_group,
    fs,
    beats=None,
    group_name="physical",
    group_channel_names=None,
    max_components=12,
    k_thresh=1.6,
    bcg_thresh=0.065,
    vbr_thresh=6.0
):
    """
    Jalankan MAICA untuk satu grup 4 channel.
    Output kompatibel dengan GUI:
    - clean_group
    - ics
    - rejected_indices
    - intermediates untuk recompute
    """
    data_group = np.asarray(data_group, dtype=float)

    if data_group.ndim != 2:
        raise ValueError(f"data_group harus 2D, diterima {data_group.shape}")

    n_ch, n_samples = data_group.shape

    if group_channel_names is None:
        group_channel_names = [f"CH{i+1}" for i in range(n_ch)]

    X_raw_data = data_group.T  # (n_samples, n_ch)

    # ============================================================
    # Augmentasi MAICA
    # ============================================================
    X_aug, L_bank, orig_col_indices = _build_maica_augmented_matrix(
        X_raw_data,
        fs,
        max_filters=MAICA_MAX_FILTERS_PER_GROUP
    )

    print(
        f"[SPLIT MAICA] {group_name} | "
        f"X_aug={X_aug.shape} | orig_col_indices={orig_col_indices}"
    )

    # ============================================================
    # Normalisasi + FastICA
    # ============================================================
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_aug)

    n_components = min(max_components, X_aug.shape[1])

    ica = FastICA(
        n_components=n_components,
        algorithm="parallel",
        whiten="unit-variance",
        whiten_solver="eigh",
        max_iter=3000,
        tol=1e-4,
        random_state=42
    )

    S_ica = ica.fit_transform(X_norm)

    # ============================================================
    # BCG reference
    # ============================================================
    pulse_smooth = _make_pulse_smooth_from_beats(beats, n_samples, fs)

    # ============================================================
    # Reject IC
    # ============================================================
    S_clean = S_ica.copy()
    ics = []
    rejected_indices = []

    for i in range(S_ica.shape[1]):
        sig_ic = S_ica[:, i]

        k_val = kurtosis(sig_ic)
        vbr_val = variance_burst_ratio(sig_ic, fs)
        is_emg = vbr_val >= vbr_thresh

        corr_pulse = 0.0
        is_bcg = False

        if pulse_smooth is not None:
            c_matrix = np.corrcoef(pulse_smooth, sig_ic)
            corr_pulse = abs(c_matrix[0, 1]) if not np.isnan(c_matrix[0, 1]) else 0.0
            is_bcg = corr_pulse >= bcg_thresh

        is_kurt = k_val >= k_thresh
        is_rejected = bool(is_kurt or is_bcg or is_emg)

        if is_rejected:
            S_clean[:, i] = 0.0
            rejected_indices.append(i)

        f_ic, p_ic = welch(
            sig_ic,
            fs,
            nperseg=min(int(fs * 2), max(8, len(sig_ic) // 2))
        )

        ics.append({
            "sig": sig_ic.copy(),
            "kurt": float(k_val),
            "corr_pulse": float(corr_pulse),
            "vbr": float(vbr_val),
            "freqs": f_ic,
            "psd": p_ic,
            "rejected": is_rejected,
            "group": group_name,
            "group_local_idx": i,
            "channel_names": list(group_channel_names),
        })

    # ============================================================
    # Rekonstruksi channel asli grup
    # ============================================================
    X_rec = scaler.inverse_transform(ica.inverse_transform(S_clean))
    clean_group = np.array([X_rec[:, col_idx] for col_idx in orig_col_indices])

    intermediates = {
        "group_name": group_name,
        "group_channel_names": list(group_channel_names),
        "S_ica": S_ica,
        "scaler": scaler,
        "ica_obj": ica,
        "orig_col_indices": orig_col_indices,
        "fs": fs,
        "pulse_smooth": pulse_smooth,
        "L_bank": L_bank,
        "max_components": int(max_components),
    }

    return {
        "clean": clean_group,
        "ics": ics,
        "rejected_indices": rejected_indices,
        "n_rejected": len(rejected_indices),
        "n_total": len(ics),
        "intermediates": intermediates,
    }


def run_maica_split_8ch_for_gui(
    subset_clean,
    fs,
    beats=None,
    k_thresh=1.6,
    bcg_thresh=0.065,
    vbr_thresh=6.0,
    max_components_per_group=12
):
    """
    Split MAICA untuk GUI lama:
    - Physical 4CH: Fp1,Fp2,T3,T4
    - Virtual/GT 4CH: F3,F4,P3,P4
    Output tetap 8CH agar pipeline lama tidak berubah.
    """
    subset_clean = np.asarray(subset_clean, dtype=float)

    if subset_clean.shape[0] < 8:
        raise ValueError(
            f"Split MAICA butuh 8 channel. Diterima shape={subset_clean.shape}"
        )

    phys_raw = subset_clean[MAICA_PHYSICAL_IDX, :]
    virt_raw = subset_clean[MAICA_VIRTUAL_GT_IDX, :]

    phys_res = _run_maica_one_group_for_gui(
        phys_raw,
        fs,
        beats=beats,
        group_name="Physical 4CH",
        group_channel_names=MAICA_PHYSICAL_NAMES,
        max_components=max_components_per_group,
        k_thresh=k_thresh,
        bcg_thresh=bcg_thresh,
        vbr_thresh=vbr_thresh
    )

    virt_res = _run_maica_one_group_for_gui(
        virt_raw,
        fs,
        beats=beats,
        group_name="GT Virtual 4CH",
        group_channel_names=MAICA_VIRTUAL_GT_NAMES,
        max_components=max_components_per_group,
        k_thresh=k_thresh,
        bcg_thresh=bcg_thresh,
        vbr_thresh=vbr_thresh
    )

    # ============================================================
    # Gabung kembali ke urutan 8CH lama:
    # Fp1,Fp2,F3,F4,T3,T4,P3,P4
    # ============================================================
    subset_ica_clean = np.zeros_like(subset_clean)

    subset_ica_clean[MAICA_PHYSICAL_IDX, :] = phys_res["clean"]
    subset_ica_clean[MAICA_VIRTUAL_GT_IDX, :] = virt_res["clean"]

    # ============================================================
    # Flatten IC untuk kompatibilitas plot lama
    # ============================================================
    ica_list = []
    rejected_indices = []

    ics_split = {
        "physical": {
            "title": "Physical 4CH: Fp1, Fp2, T3, T4",
            "ics": [],
            "rejected_indices": [],
            "n_rejected": phys_res["n_rejected"],
            "n_total": phys_res["n_total"],
        },
        "virtual_gt": {
            "title": "GT Virtual 4CH: F3, F4, P3, P4",
            "ics": [],
            "rejected_indices": [],
            "n_rejected": virt_res["n_rejected"],
            "n_total": virt_res["n_total"],
        }
    }

    # Physical
    for local_idx, ic in enumerate(phys_res["ics"]):
        flat_idx = len(ica_list)
        ic_flat = dict(ic)
        ic_flat["flat_idx"] = flat_idx
        ica_list.append(ic_flat)
        ics_split["physical"]["ics"].append(ic_flat)

        if local_idx in phys_res["rejected_indices"]:
            rejected_indices.append(flat_idx)
            ics_split["physical"]["rejected_indices"].append(local_idx)

    # Virtual / GT
    for local_idx, ic in enumerate(virt_res["ics"]):
        flat_idx = len(ica_list)
        ic_flat = dict(ic)
        ic_flat["flat_idx"] = flat_idx
        ica_list.append(ic_flat)
        ics_split["virtual_gt"]["ics"].append(ic_flat)

        if local_idx in virt_res["rejected_indices"]:
            rejected_indices.append(flat_idx)
            ics_split["virtual_gt"]["rejected_indices"].append(local_idx)

    split_stats = {
        "physical_rejected": phys_res["n_rejected"],
        "physical_total": phys_res["n_total"],
        "physical_percent": 100.0 * phys_res["n_rejected"] / max(1, phys_res["n_total"]),
        "virtual_gt_rejected": virt_res["n_rejected"],
        "virtual_gt_total": virt_res["n_total"],
        "virtual_gt_percent": 100.0 * virt_res["n_rejected"] / max(1, virt_res["n_total"]),
        "total_rejected": len(rejected_indices),
        "total_ic": len(ica_list),
    }

    print(
        "[SPLIT MAICA SUMMARY] "
        f"Physical={split_stats['physical_rejected']}/{split_stats['physical_total']} "
        f"({split_stats['physical_percent']:.1f}%) | "
        f"GT Virtual={split_stats['virtual_gt_rejected']}/{split_stats['virtual_gt_total']} "
        f"({split_stats['virtual_gt_percent']:.1f}%)"
    )

    ica_intermediates = {
        "split_mode": True,
        "fs": fs,
        "n_ch": 8,
        "groups": {
            "physical": {
                "indices": MAICA_PHYSICAL_IDX,
                "names": MAICA_PHYSICAL_NAMES,
                **phys_res["intermediates"],
            },
            "virtual_gt": {
                "indices": MAICA_VIRTUAL_GT_IDX,
                "names": MAICA_VIRTUAL_GT_NAMES,
                **virt_res["intermediates"],
            },
        },
        "split_stats": split_stats,
        "rejected_indices": rejected_indices,
    }

    return subset_ica_clean, ica_list, rejected_indices, ics_split, split_stats, ica_intermediates

def calculate_eeg_snr(raw_signal, clean_signal):
    """
    Menghitung nilai SNR (in dB) untuk setiap channel EEG.
    
    Parameter:
        raw_signal   : numpy array shape (n_channels, n_samples) -> sinyal kotor sebelum ICA
        clean_signal : numpy array shape (n_channels, n_samples) -> sinyal bersih setelah ICA
    
    Return:
        snr_per_channel : list berisi nilai SNR (dB) untuk masing-masing channel (1-8)
        avg_snr         : rata-rata SNR dari seluruh channel
    """
    # Kebisingan/Artefak didapat dari selisih sinyal sebelum dan sesudah ICA
    noise_signal = raw_signal - clean_signal
    
    snr_per_channel = []
    
    for ch in range(raw_signal.shape[0]):
        # Hitung Root Mean Square (RMS) dari sinyal bersih
        rms_signal = np.sqrt(np.mean(clean_signal[ch] ** 2))
        
        # Hitung Root Mean Square (RMS) dari noise yang dibuang
        rms_noise = np.sqrt(np.mean(noise_signal[ch] ** 2))
        
        # Antisipasi pembagian dengan nol jika tidak ada komponen yang di-reject (noise = 0)
        if rms_noise < 1e-12:
            snr_db = 40.0  # Batas atas default (sinyal sangat bersih)
        else:
            # Rumus SNR berbasis Amplitudo (RMS)
            snr_db = 20 * np.log10(rms_signal / rms_noise)
            
        snr_per_channel.append(snr_db)
        
    avg_snr = np.mean(snr_per_channel)
    return snr_per_channel, avg_snr

def variance_burst_ratio(sig, fs, win_sec=0.5, step_frac=0.5):
    """
    Hitung rasio variansi lokal maksimum vs median.
    EMG bursty → rasio tinggi. EEG stasioner → rasio rendah.
    """
    win  = int(win_sec * fs)
    step = max(1, int(win * step_frac))
    if len(sig) < win:
        return 1.0
    local_vars = np.array([
        np.var(sig[i : i + win])
        for i in range(0, len(sig) - win, step)
    ])
    median_var = np.median(local_vars)
    return float(np.max(local_vars) / (median_var + 1e-12))


def kustom_proses_eeg_anda(self):
        # 1. Tahap prapemrosesan awal / Zapline
        data_filtered = lakukan_zapline(self.session_data)
        
        # 2. Cek apakah user memilih untuk Skip ICA atau tidak
        if self.skip_ica_var.get():
            print("ICA dilewati sesuai permintaan user.")
            # Jika skip ICA, data hasil rekonstruksi langsung disamakan dengan data filter awal
            eeg_clean = data_filtered 
        else:
            print("Menjalankan pipeline ICA...")
            # Jalankan proses ICA normal Anda di sini
            # contoh: ics = jalankan_ica(data_filtered)
            #         eeg_clean = reconstruct_from_ica(ics)
            pass

        # 3. Tahap berikutnya: Ekstraksi Band Frekuensi
        # Lanjutkan ekstraksi fitur menggunakan variable 'eeg_clean'
        fitur = lakukan_ekstraksi_band_frekuensi(eeg_clean)
        

def apply_hann_taper(data, fs, taper_sec=1.0):
    """
    Terapkan Hann window taper di tepi data untuk mengurangi edge artifact
    sebelum ICA, tanpa membuang data.

    data : (n_ch, n_samples)
    Returns data yang sudah di-taper, shape sama.
    """
    n_samp = data.shape[1]
    taper_len = int(taper_sec * fs)
    taper_len = min(taper_len, n_samp // 4)

    window = np.ones(n_samp)
    # Ramp naik di awal
    ramp_up = np.hanning(taper_len * 2)[:taper_len]
    # Ramp turun di akhir
    ramp_down = np.hanning(taper_len * 2)[taper_len:]

    window[:taper_len] = ramp_up
    window[-taper_len:] = ramp_down

    return data * window[np.newaxis, :]  # broadcast ke semua channel

# ============================================================
# 5. FUNGSI UTAMA: process_eeg_data
# ============================================================
def process_eeg_data(
    filename,
    fs=250,
    skip_ica=False,
    skip_zapline=False,
    k_thresh=1.6,
    bcg_thresh=0.065,
    vbr_thresh=6.0,
    n_eeg_channels=8,
    channel_names=None,
    maica_n_components=16,
    maica_split_8ch=True,
    maica_group_max_components=12
):
    # 1. LOAD DATA
    data = pd.read_csv(filename, sep='\t', skiprows=2)
    data = data.apply(pd.to_numeric, errors='coerce').dropna(how='any').reset_index(drop=True)  # ← tambah ini

    print("DATA SHAPE:", data.shape)
    print("COLUMNS:", data.columns.tolist())
    n_cols = data.shape[1]

    raw_eeg = data.iloc[:, :min(8, n_cols)].values.T.astype(np.float64)
    print("n_samples raw_eeg:", raw_eeg.shape[1])
    print("sample pertama ch0:", raw_eeg[0, 0])
    print("sample terakhir ch0:", raw_eeg[0, -1])
    if n_cols > 8:
        ppg_digital = data.iloc[:, 8].values.astype(float)
    else:
        ppg_digital = np.zeros(data.shape[0])
        
    # ch_names_orig = ['Fp1', 'Fp2', 'T3', 'T4', 'P3', 'P4']
    ch_names_orig = ['Fp1', 'Fp2', 'F3', 'F4', 'T3', 'T4', 'P3', 'P4']

    
    # 2. PRE-PROCESSING (ZAPLINE & FILTERS)
    # Bandpass 2-80 Hz dan pembersihan derau listrik 50 Hz
    subset_bpf = bandpass_filter(raw_eeg, fs)

    if skip_zapline:
        print("[PROCESSOR] Skip Zapline aktif — memakai sinyal bandpass sebagai input tahap berikutnya.")
        subset_zap = subset_bpf.copy()
        n_zap = 0
    else:
        print("[PROCESSOR] Menjalankan Zapline...")
        subset_zap, n_zap = zapline_strong_auto_plus1(subset_bpf, fs)

    # Tetap jalankan post-notch 50 Hz
    subset_clean = post_notch(subset_zap, fs, 50, Q=50)
    # subset_clean = subset_zap
    # Menyiapkan vektor waktu untuk sinkronisasi grafik
    t_vec = np.arange(subset_clean.shape[1]) / fs
# ============================================================
    # PPG PROCESSING — Metode Midpoint + Debounce (dari notebook)
    # ============================================================
    ppg_raw = ppg_digital

    unique_vals = np.unique(ppg_raw)
    if len(unique_vals) <= 10:
        threshold_ppg = (unique_vals.max() + unique_vals.min()) / 2
        print(f"[PPG] Mode: DIGITAL diskrit | Threshold: {threshold_ppg}")
    else:
        threshold_ppg = np.median(ppg_raw)
        print(f"[PPG] Mode: ANALOG kontinu | Threshold median: {threshold_ppg:.2f}")

    ppg_binary = (ppg_raw > threshold_ppg).astype(int)
    diff_binary = np.diff(ppg_binary)
    rising_edges  = np.where(diff_binary > 0)[0] + 1
    falling_edges = np.where(diff_binary < 0)[0] + 1

    raw_midpoints = []
    for r in rising_edges:
        f_candidates = falling_edges[falling_edges > r]
        if len(f_candidates) == 0:
            continue
        f = f_candidates[0]
        pulse_width = f - r
        if pulse_width < int(0.020 * fs) or pulse_width > int(0.600 * fs):
            continue
        raw_midpoints.append((r + f) // 2)
    raw_midpoints = np.array(raw_midpoints)

    min_rr_samples = int(0.35 * fs)
    if len(raw_midpoints) == 0:
        print("[PPG] ⚠️ Tidak ada midpoint terdeteksi.")
        ppg_peaks = np.array([])
    else:
        ppg_peaks = [raw_midpoints[0]]
        for pk in raw_midpoints[1:]:
            if pk - ppg_peaks[-1] >= min_rr_samples:
                ppg_peaks.append(pk)
        ppg_peaks = np.array(ppg_peaks)

    # Alias 'beats' dipertahankan agar kompatibel dengan kode BCG di bawah
    beats = ppg_peaks

    print(f"[PPG] Rising edges    : {len(rising_edges)}")
    print(f"[PPG] Setelah validasi: {len(raw_midpoints)}")
    print(f"[PPG] Setelah debounce: {len(ppg_peaks)}")

    if len(ppg_peaks) > 2:
        rr_samples = np.diff(ppg_peaks)
        rr_ms      = rr_samples / fs * 1000
        bpm_inst   = 60000 / rr_ms
        bpm_val    = np.mean(bpm_inst)

        bpm_data = {
            "mean_bpm"  : bpm_val,
            "sdnn"      : np.std(rr_ms),
            "rmssd"     : np.sqrt(np.mean(np.diff(rr_ms)**2)),
            "status"    : "Success",
            "ppg_signal": ppg_raw,
            "ppg_time"  : np.arange(len(ppg_raw)) / fs,
            "ppg_peaks" : ppg_peaks,
            "rr_times"  : ppg_peaks[1:] / fs,
            "rr_values" : rr_ms,
            "bpm_times" : ppg_peaks[1:] / fs,
            "bpm_values": bpm_inst,
            "avg"       : bpm_val,
        }
    else:
        bpm_data = {
            "mean_bpm"  : 0,
            "status"    : "No Pulse Detected",
            "ppg_signal": ppg_raw,
            "ppg_time"  : np.arange(len(ppg_raw)) / fs,
            "ppg_peaks" : np.array([]),
            "times": [], "values": [], "avg": 0
        }

# ====================================================================
    # 4. PARTIAL MAICA (Dengan Opsi Skip ICA)
    # ====================================================================
    X_raw_data = subset_clean.T          # Langsung gunakan data clean tanpa taper
    n_samples, n_ch = X_raw_data.shape

    # Inisialisasi variabel penampung default untuk mencegah error saat skip_ica=True
    ica_list = []
    rejected_indices = []
    S_ica = np.zeros((n_samples, 1))     # dummy array untuk metadata ica_intermediates
    scaler = None
    ica = None
    L_bank = []
    orig_col_indices = list(range(n_ch)) # Indeks default channel asli
    pulse_smooth = None

    # ============================================================
    # 6. MAICA / ICA
    # ============================================================

    ica_list = []
    rejected_indices = []
    ics_split = None
    split_stats = None

    if skip_ica:
        print("[PROCESSOR] Skip ICA aktif — MAICA dilewati.")
        subset_ica_clean = subset_clean.copy()

        # dummy IC agar GUI tidak crash
        for i in range(n_ch):
            sig_dummy = subset_clean[i, :]
            f_ic, p_ic = welch(
                sig_dummy,
                fs,
                nperseg=min(int(fs * 2), max(8, len(sig_dummy) // 2))
            )
            ica_list.append({
                "sig": sig_dummy,
                "kurt": 0.0,
                "corr_pulse": 0.0,
                "vbr": 0.0,
                "freqs": f_ic,
                "psd": p_ic,
                "rejected": False,
                "group": "Skip ICA"
            })

        ica_intermediates = {
            "split_mode": False,
            "skipped": True,
            "fs": fs,
            "n_ch": n_ch,
            "rejected_indices": [],
        }

    else:
        # ========================================================
        # MODE BARU: SPLIT MAICA 8CH
        # ========================================================
        if maica_split_8ch and n_ch >= 8:
            print("[PROCESSOR] Menjalankan SPLIT MAICA 8CH:")
            print("  - Physical 4CH   : Fp1,Fp2,T3,T4")
            print("  - GT Virtual 4CH : F3,F4,P3,P4")

            (
                subset_ica_clean,
                ica_list,
                rejected_indices,
                ics_split,
                split_stats,
                ica_intermediates
            ) = run_maica_split_8ch_for_gui(
                subset_clean,
                fs,
                beats=beats,
                k_thresh=k_thresh,
                bcg_thresh=bcg_thresh,
                vbr_thresh=vbr_thresh,
                max_components_per_group=maica_group_max_components
            )

        # ========================================================
        # FALLBACK LAMA: MAICA satu proses
        # ========================================================
        else:
            print("[PROCESSOR] Menjalankan pipeline MAICA normal satu grup...")

            X_raw_data = subset_clean.T
            n_samples, n_ch_in = X_raw_data.shape

            L_bank = compute_maica_L(
                X_raw_data[:, 0],
                fs,
                min_orig=0.6,
                max_orig=0.98,
                max_outputs=0.4
            )

            print(f"[MAICA] L_bank adaptif: {L_bank} ({len(L_bank)} filter)")

            def zero_phase_ma(x, N):
                b = np.ones(N) / N
                return filtfilt(b, [1.0], x)

            X_aug_list = []
            orig_col_indices = []

            for i in range(n_ch):
                orig = X_raw_data[:, i]
                orig_col_indices.append(len(X_aug_list))
                X_aug_list.append(orig)

                for N in L_bank:
                    virt = zero_phase_ma(orig, N)
                    corr_ok = True

                    for existing in X_aug_list:
                        c = abs(np.corrcoef(virt, existing)[0, 1])
                        if c > 0.98:
                            corr_ok = False
                            break

                    if corr_ok:
                        X_aug_list.append(virt)

            X_aug = np.column_stack(X_aug_list)
            print(f"[MAICA] Augmented matrix: {X_aug.shape}")
            print(f"[MAICA] Kolom channel asli: {orig_col_indices}")

            scaler = StandardScaler()
            X_norm = scaler.fit_transform(X_aug)

            n_components = min(maica_n_components, X_aug.shape[1])

            ica = FastICA(
                n_components=n_components,
                algorithm="parallel",
                whiten="unit-variance",
                whiten_solver="eigh",
                max_iter=3000,
                tol=1e-4,
                random_state=42
            )

            S_ica = ica.fit_transform(X_norm)

            pulse_smooth = _make_pulse_smooth_from_beats(beats, n_samples, fs)

            S_clean_ica = S_ica.copy()
            rejected_indices = []
            ica_list = []

            for i in range(S_ica.shape[1]):
                sig_ic = S_ica[:, i]

                k_val = kurtosis(sig_ic)
                vbr_val = variance_burst_ratio(sig_ic, fs)
                is_emg = vbr_val >= vbr_thresh

                corr_pulse = 0.0
                is_bcg = False

                if pulse_smooth is not None:
                    c_matrix = np.corrcoef(pulse_smooth, sig_ic)
                    corr_pulse = abs(c_matrix[0, 1]) if not np.isnan(c_matrix[0, 1]) else 0.0
                    is_bcg = corr_pulse >= bcg_thresh

                is_kurt = k_val >= k_thresh
                is_rejected = bool(is_kurt or is_bcg or is_emg)

                if is_rejected:
                    S_clean_ica[:, i] = 0.0
                    rejected_indices.append(i)

                f_ic, p_ic = welch(
                    sig_ic,
                    fs,
                    nperseg=min(int(fs * 2), max(8, len(sig_ic) // 2))
                )

                ica_list.append({
                    "sig": sig_ic.copy(),
                    "kurt": float(k_val),
                    "corr_pulse": float(corr_pulse),
                    "vbr": float(vbr_val),
                    "freqs": f_ic,
                    "psd": p_ic,
                    "rejected": is_rejected,
                    "group": "Single MAICA"
                })

            X_rec = scaler.inverse_transform(ica.inverse_transform(S_clean_ica))
            subset_ica_clean = np.array([X_rec[:, col_idx] for col_idx in orig_col_indices])

            ica_intermediates = {
                "split_mode": False,
                "S_ica": S_ica,
                "scaler": scaler,
                "ica_obj": ica,
                "n_ch": n_ch,
                "L_bank": L_bank,
                "orig_col_indices": orig_col_indices,
                "fs": fs,
                "rejected_indices": rejected_indices,
                "pulse_smooth": pulse_smooth,
            }
        
    EEG_BANDS = {
        'Delta': (0.5, 4.0),
        'Theta': (4.0, 8.0),
        'Alpha': (8.0, 13.0),
        'Beta' : (13.0, 30.0),
    }
    band_names  = list(EEG_BANDS.keys())
    n_ch_ica    = subset_ica_clean.shape[0]
    n_samp_ica  = subset_ica_clean.shape[1]

    rhythm_signals = np.zeros((len(band_names), n_ch_ica, n_samp_ica))
    band_ranges    = {}
    imf_map_dict   = {b: [] for b in band_names}   # kosong, tidak ada MIMF lagi
    band_list      = []   # untuk tab plot (pengganti mif_list)

    for b_idx, (band, (lo, hi)) in enumerate(EEG_BANDS.items()):
        filtered = bandpass_band(subset_ica_clean, fs, lo, hi)   # (n_ch, n_samp)
        rhythm_signals[b_idx] = filtered
        band_ranges[band]     = [lo, hi]

        # Simpan sinyal channel-0 + PSD untuk tab plot
        sig_b       = filtered[0]
        f_b, p_b    = welch(sig_b, fs, nperseg=min(int(fs * 2), len(sig_b)))
        band_list.append({'name': band, 'sig': sig_b, 'f': f_b, 'p': p_b,
                          'lo': lo, 'hi': hi})
        
            

    # Ekstraksi Fase menggunakan Hilbert untuk fitur konektivitas
# Ekstraksi Fase menggunakan Hilbert (DENGAN PROTEKSI SINYAL KOSONG)
    phase_signals = np.zeros_like(rhythm_signals)

    n_phase_ch = rhythm_signals.shape[1]

    for b_idx in range(len(band_names)):
        for ch in range(n_phase_ch):
            sig_segment = rhythm_signals[b_idx, ch]

            if np.any(sig_segment):
                phase_signals[b_idx, ch] = np.angle(hilbert(sig_segment))
            else:
                phase_signals[b_idx, ch] = 0.0    
    # 7. FEATURE EXTRACTION
    feat_dict, feat_table, abs_p, rel_p = extract_features_comprehensive(
        rhythm_signals, band_names, band_ranges, phase_signals, fs
    )

    # 8. PREPARE MNE INFO
    info_mne = mne.create_info(ch_names=ch_names_orig, sfreq=fs, ch_types='eeg')
    info_mne.set_montage(
        mne.channels.make_standard_montage('standard_1020'),
        on_missing='ignore'
    )

    # Final Return dengan struktur yang tepat untuk GUI.py
    return {
        'zapline': {
            't': t_vec,
            'raw': subset_bpf,
            'clean': subset_clean,
            'n_removed': n_zap,
            'skipped': skip_zapline
        },
        'ica': {
            'ics': ica_list,

            # ========================================================
            # Tambahan untuk split MAICA
            # ========================================================
            'ics_split': ics_split,
            'split_stats': split_stats,

            'data': subset_ica_clean,
            'before': subset_clean,
            'after': subset_ica_clean,

            'rejected_indices': rejected_indices,
            'n_rejected': len(rejected_indices),
        },
        # === TAMBAHAN: simpan intermediate agar bisa rekonstruksi ulang ===
        'ica_intermediates': ica_intermediates,
        # ====================================================================
        'mif': {
            'mimfs': band_list
        },
        'bpm_data': bpm_data,
        'bands': {
            'signals': rhythm_signals,
            'abs_power': abs_p,
            'rel_power': rel_p,
            'powers': {b: np.mean(rel_p[b]) if b in rel_p else 0.0 for b in band_names},
            'ranges': {b: tuple(v) for b, v in band_ranges.items()},
            'imf_map': imf_map_dict,
            'names': band_names
        },
        'features': {
            'table': feat_table,
            'dict': feat_dict
        },
        'topo_info': info_mne
    }