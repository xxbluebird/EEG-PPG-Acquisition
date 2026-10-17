import time
import threading
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

from scipy.signal import butter, filtfilt, welch
import mne  
from matplotlib.colors import ListedColormap


import eeg_processor_8
from eeg_processor_8 import reconstruct_from_ica

# --- KONSTANTA GLOBAL  ---
FS = 250
WINDOW_SEC = 10
MAX_SAMPLES = FS * WINDOW_SEC
NUM_CHANNELS = 8
NUM_DIGITAL = 3   # D11, D12, D13


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: Hapus edge artifact dari sinyal ICA hasil rekonstruksi
# Spike di ujung IC terjadi karena filter/ICA edge effect (transien matematis).
# Solusi: trim beberapa sample di awal & akhir, lalu interpolasi linear agar
# sinyal tetap kontinu (tidak ada jump ke nol).
# ──────────────────────────────────────────────────────────────────────────────
def trim_ic_edges(ic_signal, trim_samples=None, fs=FS):
    """
    Hapus edge artifact pada satu sinyal IC 1-D.

    Parameters
    ----------
    ic_signal    : np.ndarray  shape (n_samples,)
    trim_samples : int | None  — jumlah sample yang di-trim di tiap ujung.
                                 Default = 0.2 detik (50 sample pada 250 Hz).
    fs           : int         — sampling frequency (untuk hitung default trim)

    Returns
    -------
    np.ndarray shape (n_samples,) — sinyal asli dengan tepi yang di-smooth,
    panjang TIDAK berubah (tidak di-crop) agar index tetap konsisten.
    """
    if trim_samples is None:
        trim_samples = int(fs * 0.2)  # default: 0.2 detik = 50 sample

    n = len(ic_signal)
    if n <= 2 * trim_samples:
        return ic_signal  # terlalu pendek, lewati

    result = ic_signal.copy()

    # Nilai "stabil" diambil dari rata-rata 10 sample setelah zona trim
    stable_start = np.mean(ic_signal[trim_samples : trim_samples + 10])
    stable_end   = np.mean(ic_signal[-(trim_samples + 10) : -trim_samples])

    # Interpolasi linear dari nol ke nilai stabil di tepi kiri
    result[:trim_samples] = np.linspace(0, stable_start, trim_samples)

    # Interpolasi linear dari nilai stabil ke nol di tepi kanan
    result[-trim_samples:] = np.linspace(stable_end, 0, trim_samples)

    return result


def trim_ica_clean(data, trim_samples=None, fs=FS):
    """
    Terapkan trim_ic_edges ke setiap channel pada data multidimensi.
    data : np.ndarray  shape (n_channels, n_samples)
    """
    if data is None or data.ndim != 2:
        return data
    return np.array([trim_ic_edges(data[ch], trim_samples, fs)
                     for ch in range(data.shape[0])])


def trim_ic_components(ics, trim_samples=None, fs=FS):
    """
    Terapkan trim_ic_edges ke field 'sig' pada setiap dict IC.
    ics : list of dict — masing-masing dict berisi key 'sig'
    """
    trimmed = []
    for ic in ics:
        ic_copy = dict(ic)
        ic_copy['sig'] = trim_ic_edges(ic['sig'], trim_samples, fs)
        trimmed.append(ic_copy)
    return trimmed

def make_light_jet(minval=0.12, maxval=1.0, n=256):
    """
    Membuat colormap jet tanpa bagian biru tergelap.
    minval makin besar -> warna minimum makin terang.
  
    """
    base = plt.get_cmap("jet")
    new_colors = base(np.linspace(minval, maxval, n))
    return ListedColormap(new_colors, name="light_jet")

LIGHT_JET = make_light_jet(minval=0.2)
def calculate_impedance_openbci(raw_data):
    if len(raw_data) < FS:
        return 0.0

    CURRENT_RMS = 6e-9
    TARGET_FREQ = 31.25

    data = raw_data - np.mean(raw_data)

    t = np.arange(len(data)) / FS
    ref_sin = np.sin(2 * np.pi * TARGET_FREQ * t)
    ref_cos = np.cos(2 * np.pi * TARGET_FREQ * t)

    mixed_i = data * ref_sin
    mixed_q = data * ref_cos

    avg_i = np.mean(mixed_i)
    avg_q = np.mean(mixed_q)

    v_rms_uV = np.sqrt(avg_i**2 + avg_q**2)
    v_rms_volt = v_rms_uV * 1e-6

    z = v_rms_volt / CURRENT_RMS

    return max(0, z / 1000.0)


class EEGAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenBCI EEG - 8 Channel Research Suite")
        # Mengambil resolusi layar aktual milik Raspberry Pi
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        # Metric normatif untuk tab PLI
        # Pilihan: "pli_proxy", "wpli_proxy", "imcoh_proxy"
        self.avg_pli_norm_metric = "imcoh_proxy"
        

        # Mengatur ukuran GUI sebesar 90% dari resolusi layar asli
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)
        self.avg_asym_conn_threshold = 0.20
        # Threshold garis PLI pada Avg Feature Topomap
        self.avg_pli_conn_threshold = 0.07
        # ============================================================
        # NORMATIVE Z-SCORE SETTINGS (Avg Feature Topomap)
        # ============================================================
        self.normative_asym_csv = r"D:\PRATA AMBA' RATA\normative database\EyesClose\normative_asymmetry_stats_8ch.csv"
        self.normative_pli_csv  = r"D:\PRATA AMBA' RATA\normative database\EyesClose\normative_pli_proxy_stats_8ch.csv"

        # Threshold Z-score
        self.avg_zscore_threshold = 1.4 
        self.avg_zscore_cap = 3.09         # untuk skala ketebalan garis

        # Cache
        self._norm_stats_cache = {}

        # Channel GUI
        self.avg_topo_ch_names = ["Fp1", "Fp2", "F3", "F4", "T3", "T4", "P3", "P4"]

        # Menerapkan ukuran baru secara dinamis
        self.root.geometry(f"{window_width}x{window_height}+0+0")
        self.show_before_ica_var = tk.BooleanVar(value=True)
        self.show_ica_compare_legend_var = tk.BooleanVar(value=True)
        
        # Di dalam def __init__(self, root):
        self.loaded_data_from_file = None  # Tempat menyimpan data TXT
        self.current_source = "LIVE"       # Flag untuk menandai sumber data (LIVE atau FILE)
        
        # --- KONFIGURASI WARNA ---
        # Cari di dalam def __init__(self, root):
        self.colors = {
            "bg": "#ffffff",            # Putih Bersih (Background Utama)
            "bg_card": "#f8f9fa",       # Abu-abu sangat muda (Untuk Frame)
            "secondary_bg": "#f8f9fa",  # Alias agar tidak error
            "accent": "#1a73e8",        # Biru Google/Medical (Aksen Utama)
            "accent_blue": "#1a73e8",   # Alias biru
            "green": "#28a745",         # Hijau standar (Connect)
            "accent_green": "#28a745",  # Alias hijau
            "red": "#dc3545",           # Merah standar (Stop)
            "accent_red": "#dc3545",    # Alias merah
            "yellow": "#ffc107",        # Kuning standar
            "text": "#202124",          # Hitam/Abu gelap (Teks Utama)
            "text_dim": "#5f6368"       # Abu-abu (Teks Sekunder)
        }
        self.root.configure(bg=self.colors["bg"])

        self.board = None
        self.streaming = False
        self.is_configuring = False
        self.last_known_imp = [None] * NUM_CHANNELS
        self.is_recording = False
        self._countdown_job = None
        self._countdown_job = None  
        self.session_data = []      
        self.data_buffer = np.zeros((NUM_CHANNELS + 1, MAX_SAMPLES))
        self.time_axis = np.linspace(-5, 5, MAX_SAMPLES)
        self.last_filename = ""
        self.labels_imp_text = []   
        self.imp_states = [False] * NUM_CHANNELS 
        self.imp_buttons = [] 
        self.last_impedance_time = [0] * NUM_CHANNELS
        self.IMP_INTERVAL = 1.0  
        # Average topomap
        self.avg_topo_files = []
        self.avg_topo_start_var = tk.StringVar(value="0")
        self.avg_topo_end_var = tk.StringVar(value="")
        self.avg_topo_label_var = tk.StringVar(value="No files selected")
     
        self.skip_ica_var = tk.BooleanVar(value=False)  # Default False: ICA tetap berjalan
        self.skip_zapline_var = tk.BooleanVar(value=False)
        self.ica_kurt_var = tk.DoubleVar(value=1.6)
        self.ica_bcg_var  = tk.DoubleVar(value=0.065)  
        self.ica_vbr_var  = tk.DoubleVar(value=6.0)    
        # Topomap subtraction / difference
        self.diff_topo_files = []
        self.diff_topo_label_var = tk.StringVar(value="No files selected")

        self.diff1_start_var = tk.StringVar(value="0")
        self.diff1_end_var   = tk.StringVar(value="")

        self.diff2_start_var = tk.StringVar(value="")
        self.diff2_end_var   = tk.StringVar(value="")

        self._setup_styles()
        self._build_main_gui()
        self._build_realtime_plot()
        
    def compute_lzc_maps_from_eeg(self, eeg_data, fs=FS):
        """
        Menghitung LZC topomap dari sinyal EEG bersih.

        Input:
            eeg_data shape = (n_channel, n_sample)

        Output:
            {
                "LZC_Delta": [nilai per channel],
                "LZC_Theta": [nilai per channel],
                "LZC_Alpha": [nilai per channel],
                "LZC_Beta":  [nilai per channel]
            }
        """
        import numpy as np
        from scipy.signal import butter, filtfilt

        eeg_data = np.asarray(eeg_data, dtype=float)

        if eeg_data.ndim != 2:
            raise ValueError(f"eeg_data harus 2D, shape sekarang: {eeg_data.shape}")

        # Kalau formatnya (sample, channel), transpose otomatis
        if eeg_data.shape[0] > eeg_data.shape[1] and eeg_data.shape[1] <= 16:
            eeg_data = eeg_data.T

        n_ch, n_samp = eeg_data.shape

        if n_ch < 2:
            raise ValueError(f"Jumlah channel terlalu sedikit untuk topomap LZC: {n_ch}")

        band_defs = {
            "LZC_Delta": (0.5, 4.0),
            "LZC_Theta": (4.0, 8.0),
            "LZC_Alpha": (8.0, 13.0),
            "LZC_Beta":  (13.0, 30.0),
        }

        nyq = fs / 2.0
        lzc_maps = {}

        for band_name, (lo, hi) in band_defs.items():
            low = max(lo, 0.1)
            high = min(hi, nyq - 0.1)

            b, a = butter(
                4,
                [low / nyq, high / nyq],
                btype="band"
            )

            x_band = filtfilt(b, a, eeg_data, axis=-1)

            vals = []

            for ch in range(n_ch):
                x = np.asarray(x_band[ch], dtype=float)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

                # Binarisasi berdasarkan median channel tersebut
                med = np.median(x)
                binary = (x > med).astype(int)

                vals.append(self.lzc_binary_complexity(binary))

            lzc_maps[band_name] = vals

        return lzc_maps


    def lzc_binary_complexity(self, binary_seq):
        """
        Lempel-Ziv Complexity untuk sequence biner.
        Input: binary_seq 1D berisi 0/1.
        Output: nilai LZC ternormalisasi.
        """
        import numpy as np
        import math

        binary_seq = np.asarray(binary_seq).astype(int).reshape(-1)
        s = "".join(binary_seq.astype(str).tolist())
        n = len(s)

        if n < 2:
            return 0.0

        i = 0
        c = 1
        l = 1
        k = 1
        k_max = 1

        while True:
            if i + k >= n or l + k >= n:
                c += 1
                break

            if s[i + k] == s[l + k]:
                k += 1
                if l + k > n:
                    c += 1
                    break
            else:
                if k > k_max:
                    k_max = k

                i += 1

                if i == l:
                    c += 1
                    l += k_max

                    if l + 1 > n:
                        break

                    i = 0
                    k = 1
                    k_max = 1
                else:
                    k = 1

        norm = n / math.log2(n) if n > 1 else 1.0
        return float(c / norm)
  
    def read_offline_eeg_file(self, file_path):
        """
        Membaca file offline EEG dan menstandarkan formatnya agar kompatibel
        dengan eeg_processor_8.py.

        Format yang didukung:
        1. TXT OpenBCI/tab-separated:
        kolom EEG berada pada 8 kolom pertama, PPG/digital kolom ke-9 jika ada.

        2. CSV hasil linear mixing:
        time_s,Fp1,Fp2,F3,F4,T3,T4,P3,P4,PPG
        atau
        time_s,Fp1,Fp2,F3,F4,T7,T8,P3,P4,PPG

        3. CSV hasil save GUI:
        rec_id,EEG_ch1,EEG_ch2,...,EEG_ch8,PPG

        Output:
        DataFrame standar:
        EXG Channel 0..7, Digital
        """
        import os
        import numpy as np
        import pandas as pd

        ext = os.path.splitext(file_path)[1].lower()

        # ------------------------------------------------------------
        # 1. Baca file
        # ------------------------------------------------------------
        if ext == ".csv":
            df = pd.read_csv(file_path)
        else:
            # format TXT OpenBCI / tab-separated
            df = pd.read_csv(file_path, comment='%', sep='\t')

        if df.empty:
            raise ValueError("File kosong atau tidak terbaca.")

        # Normalisasi nama kolom agar robust terhadap kapital/spasi/underscore
        def norm_col(x):
            return str(x).strip().lower().replace(" ", "").replace("_", "")

        col_map = {norm_col(c): c for c in df.columns}

        eeg_cols = None
        ppg_col = None
        detected_format = None

        # ------------------------------------------------------------
        # 2. Format CSV hasil save GUI:
        # rec_id,EEG_ch1,...,EEG_ch8,PPG
        # ------------------------------------------------------------
        if all(f"eegch{i}" in col_map for i in range(1, 9)):
            detected_format = "CSV GUI saved signal: EEG_ch1..EEG_ch8"

            eeg_cols = [
                col_map["eegch1"],
                col_map["eegch2"],
                col_map["eegch3"],
                col_map["eegch4"],
                col_map["eegch5"],
                col_map["eegch6"],
                col_map["eegch7"],
                col_map["eegch8"],
            ]

            if "ppg" in col_map:
                ppg_col = col_map["ppg"]

        # ------------------------------------------------------------
        # 3. Format CSV linear mixing:
        # time_s,Fp1,Fp2,F3,F4,T3,T4,P3,P4,PPG
        # atau T7/T8
        # ------------------------------------------------------------
        elif (
            "fp1" in col_map and
            "fp2" in col_map and
            "f3" in col_map and
            "f4" in col_map and
            "p3" in col_map and
            "p4" in col_map and
            (
                ("t3" in col_map and "t4" in col_map) or
                ("t7" in col_map and "t8" in col_map)
            )
        ):
            detected_format = "CSV linear mixing: Fp1,Fp2,F3,F4,T3/T4,P3/P4"

            t_left = col_map["t3"] if "t3" in col_map else col_map["t7"]
            t_right = col_map["t4"] if "t4" in col_map else col_map["t8"]

            eeg_cols = [
                col_map["fp1"],
                col_map["fp2"],
                col_map["f3"],
                col_map["f4"],
                t_left,
                t_right,
                col_map["p3"],
                col_map["p4"],
            ]

            if "ppg" in col_map:
                ppg_col = col_map["ppg"]

        # ------------------------------------------------------------
        # 4. Fallback TXT/OpenBCI:
        # ambil 8 kolom numerik pertama sebagai EEG
        # kolom ke-9 sebagai Digital/PPG jika ada
        # ------------------------------------------------------------
        else:
            detected_format = "Fallback TXT/OpenBCI: first 8 numeric columns"

            if df.shape[1] < NUM_CHANNELS:
                raise ValueError(
                    f"File tidak punya minimal {NUM_CHANNELS} kolom EEG. "
                    f"Kolom tersedia: {list(df.columns)}"
                )

            eeg_cols = list(df.columns[:NUM_CHANNELS])

            if df.shape[1] > NUM_CHANNELS:
                ppg_col = df.columns[NUM_CHANNELS]

        # ------------------------------------------------------------
        # 5. Bangun dataframe standar untuk processor
        # ------------------------------------------------------------
        out = pd.DataFrame()

        for i, col in enumerate(eeg_cols):
            out[f"EXG Channel {i}"] = pd.to_numeric(df[col], errors="coerce")

        if ppg_col is not None:
            out["Digital"] = pd.to_numeric(df[ppg_col], errors="coerce")
        else:
            # Kalau tidak ada PPG/digital, isi nol agar processor tetap jalan
            out["Digital"] = 0.0

        # Isi NaN
        n_nan = int(out.isna().sum().sum())
        if n_nan > 0:
            print(f"[LOAD FILE] Ditemukan {n_nan} NaN, diisi 0.")
            out = out.fillna(0)

        # Pastikan urutan kolom fix
        ordered_cols = [f"EXG Channel {i}" for i in range(NUM_CHANNELS)] + ["Digital"]
        out = out[ordered_cols]

        print("\n[LOAD FILE] File berhasil distandarkan")
        print(f"  Path            : {file_path}")
        print(f"  Format terdeteksi: {detected_format}")
        print(f"  Shape original  : {df.shape}")
        print(f"  Shape standar   : {out.shape}")
        print(f"  EEG columns     : {eeg_cols}")
        print(f"  PPG/Digital col : {ppg_col}")

        return out, detected_format
    
    def _build_main_gui(self):
            # Frame Utama Kontrol
            ctrl_frame = ttk.Frame(self.root)
            ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=15, pady=10)

            # --- 1. GROUP KONEKSI (LIVE) ---
            conn_frame = ttk.LabelFrame(ctrl_frame, text=" LIVE CONNECTION ", padding=10)
            conn_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
            
            ttk.Label(conn_frame, text="Port:").pack(side=tk.LEFT, padx=2)
            self.com_entry = ttk.Entry(conn_frame, width=8, justify='center')
            self.com_entry.insert(0, "COM10")
            self.com_entry.pack(side=tk.LEFT, padx=5)
            
            btn_conn = tk.Button(conn_frame, text="CONNECT", command=self.prepare_session,
                                bg=self.colors["accent"], fg="white", 
                                activebackground="#174ea6", activeforeground="white",
                                font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, padx=10)
            btn_conn.pack(side=tk.LEFT, padx=5)

            # --- 2. SECTION BARU: OFFLINE DATA (LOAD FILE) ---
            off_frame = ttk.LabelFrame(ctrl_frame, text=" OFFLINE DATA ", padding=10)
            off_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)

            btn_load = tk.Button(off_frame, text="📂 LOAD FILE", command=self.load_txt_file,
                                bg="#f8f9fa", fg=self.colors["accent"], 
                                activebackground="#e8f0fe",
                                font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, 
                                highlightbackground=self.colors["accent"], highlightthickness=1,
                                padx=10)
            btn_load.pack(side=tk.LEFT, padx=5)
            
            self.file_label = ttk.Label(off_frame, text="No file", width=12)
            self.file_label.pack(side=tk.LEFT, padx=5)

            # --- 3. GROUP IMPEDANSI ---
            imp_frame = ttk.LabelFrame(ctrl_frame, text=" IMPEDANCE CHECK ", padding=10)
            imp_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)

            self.imp_buttons = []
            for i in range(NUM_CHANNELS):
                btn = tk.Button(imp_frame, text=f"CH{i+1}", width=4, 
                                bg="#f1f3f4", fg=self.colors["text"],
                                relief=tk.FLAT, bd=1, highlightthickness=1,
                                highlightbackground="#dadce0",
                                command=lambda idx=i: self.toggle_channel_impedance(idx))
                btn.pack(side=tk.LEFT, padx=2)
                self.imp_buttons.append(btn)

            # --- 4. GROUP RECORDING (ACQUISITION) ---
            rec_frame = ttk.LabelFrame(ctrl_frame, text=" ACQUISITION ", padding=10)
            rec_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
            
# --- Input durasi rekaman ---
            tk.Label(rec_frame, text="Durasi (s):", font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(5,0))
            self.duration_var = tk.IntVar(value=60)
            self.spinbox_duration = tk.Spinbox(rec_frame, from_=5, to=600, increment=5,
                                               textvariable=self.duration_var,
                                               width=5, font=('Segoe UI', 9))
            self.spinbox_duration.pack(side=tk.LEFT, padx=(2,8))

           
            tk.Label(rec_frame, text="Nama File:", font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(5,0))
            self.filename_var = tk.StringVar(value="EEG_Recording")
            self.filename_entry = tk.Entry(rec_frame, textvariable=self.filename_var,
                                        width=18, font=('Segoe UI', 9))
            self.filename_entry.pack(side=tk.LEFT, padx=(2,8))

            # --- Label countdown ---
            self.countdown_label = tk.Label(rec_frame, text="", font=('Segoe UI', 9, 'bold'),
                                            fg="red", bg="#e8eaed", width=8)
            self.countdown_label.pack(side=tk.LEFT, padx=5)

            # --- Tombol START RECORDING ---
           
            self.btn_record = tk.Button(rec_frame, text="▶ START RECORDING",
                                        command=self.start_recording,
                                        bg=self.colors["green"], fg="white",
                                        state=tk.DISABLED,
                                        activebackground="#1e7e34", activeforeground="white",
                                        relief=tk.FLAT, font=('Segoe UI', 9, 'bold'), padx=10)
            self.btn_record.pack(side=tk.LEFT, padx=5)

            self.btn_stop = tk.Button(rec_frame, text="STOP", command=self.stop_and_save,
                                    bg="#e8eaed", fg="#9aa0a6", state=tk.DISABLED,
                                    relief=tk.FLAT, font=('Segoe UI', 9, 'bold'), padx=10)
            self.btn_stop.pack(side=tk.LEFT, padx=5)
            
            # --- 5. ANALYSIS ---
            an_frame = ttk.LabelFrame(ctrl_frame, text=" ANALYSIS ", padding=10)
            an_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)

            tk.Label(an_frame, text="Start (s):", font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(5, 0))
            self.analysis_start_var = tk.StringVar(value="0")
            tk.Entry(an_frame, textvariable=self.analysis_start_var,
                     width=5, font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(2, 5))

            tk.Label(an_frame, text="End (s):", font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(5, 0))
            self.analysis_end_var = tk.StringVar(value="") 
            tk.Entry(an_frame, textvariable=self.analysis_end_var,
                     width=5, font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(2, 8))

            # --- Variabel kontrol skip proses ---
            self.skip_ica_var = tk.BooleanVar(value=False)
            self.skip_zapline_var = tk.BooleanVar(value=False)

            # --- Checkbutton Skip ICA ---
            self.chk_skip_ica = ttk.Checkbutton(
                an_frame,
                text="Skip ICA",
                variable=self.skip_ica_var
            )
            self.chk_skip_ica.pack(side=tk.LEFT, padx=(5, 6))

            # --- Checkbutton Skip Zapline ---
            self.chk_skip_zapline = ttk.Checkbutton(
                an_frame,
                text="Skip Zapline",
                variable=self.skip_zapline_var
            )
            self.chk_skip_zapline.pack(side=tk.LEFT, padx=(0, 10))

            self.btn_analyze = tk.Button(an_frame, text="RUN ANALYSIS", command=self.start_analysis_thread,
                                        bg=self.colors["accent"], fg="white", state=tk.DISABLED,
                                        relief=tk.FLAT, font=('Segoe UI', 9, 'bold'), padx=10)
            self.btn_analyze.pack(side=tk.LEFT, padx=5)
            
            # Status Bar
            self.status_label = tk.Label(self.root, text="STATUS: IDLE", 
                                        bg=self.colors["bg"],
                                        fg=self.colors["accent"], 
                                        font=("Consolas", 10, "bold"))
            self.status_label.pack(side=tk.TOP, anchor="w", padx=25, pady=0)

  
    def load_txt_file(self):
        from tkinter import filedialog
        import os

        file_path = filedialog.askopenfilename(
            title="Select EEG File",
            filetypes=[
                ("EEG files", "*.txt *.csv"),
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ]
        )

        if file_path:
            try:
                self.last_filename = file_path

                # Baca TXT/CSV lalu ubah menjadi format internal standar
                data, detected_format = self.read_offline_eeg_file(file_path)

                if data.shape[1] < NUM_CHANNELS:
                    messagebox.showerror(
                        "Error",
                        f"File tidak berisi minimal {NUM_CHANNELS} channel EEG."
                    )
                    return

                # Simpan dataframe standar agar run_analysis_pipeline tidak baca ulang format asli
                self.offline_df = data.copy()

                # Simpan data ke session_data: shape channel x sample
                self.session_data = data.iloc[:, :NUM_CHANNELS].values.T
                self.current_source = "FILE"

                #  live buffer agar terlihat di tab Live EEG
                try:
                    n_show = min(MAX_SAMPLES, data.shape[0])
                    self.data_buffer[:] = 0
                    self.data_buffer[:NUM_CHANNELS, -n_show:] = data.iloc[-n_show:, :NUM_CHANNELS].values.T
                    self.data_buffer[NUM_CHANNELS, -n_show:] = data["Digital"].values[-n_show:]
                    self.canvas.draw_idle()
                except Exception as e:
                    print(f"[LOAD FILE] Preview buffer skipped: {e}")

                # Update UI
                self.file_label.config(text=os.path.basename(file_path)[:15] + "...")
                self.status_label.config(
                    text=f"STATUS: FILE LOADED ({detected_format})",
                    fg=self.colors["accent"]
                )

                # Aktifkan tombol analisis
                self.btn_analyze.config(
                    state=tk.NORMAL,
                    bg=self.colors["accent"],
                    fg="white"
                )

                messagebox.showinfo(
                    "Success",
                    "Data berhasil dimuat dan otomatis dikonversi internal.\n\n"
                    f"Format terdeteksi:\n{detected_format}\n\n"
                    "Silakan tekan RUN ANALYSIS."
                )

            except Exception as e:
                messagebox.showerror("Error", f"Gagal membaca file:\n{e}")
                
    def build_save_tab(self, parent):
        """Membangun tab Saving Feature berisi daftar fitur + tombol save CSV."""
        for widget in parent.winfo_children():
            widget.destroy()

        # --- Header info ---
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill=tk.X, padx=10, pady=6)

        if self.last_features:
            fi = self.last_features
            info_text = (
                f"File: {fi.get('filename','?')}   |   "
                f"Subject: {fi.get('subject','?')}   |   "
                f"Rec ID: {fi.get('rec_id','?')}   |   "
                f"Stress: {fi.get('stress','?')}   "
                f"Anxiety: {fi.get('anxiety','?')}   "
                f"Depression: {fi.get('depression','?')}"
            )
        else:
            info_text = "Belum ada fitur — jalankan analisis terlebih dahulu."

        tk.Label(info_frame, text=info_text, font=('Segoe UI', 9),
                 fg=self.colors["accent"], bg=self.colors["bg"],
                 wraplength=900, justify='left').pack(side=tk.LEFT)

        # --- Treeview daftar fitur ---
        # --- Input Rec ID ---
        recid_frame = ttk.Frame(parent)
        recid_frame.pack(fill=tk.X, padx=10, pady=4)

        tk.Label(recid_frame, text="Rec ID:", font=('Segoe UI', 10, 'bold'),
                 fg=self.colors["text"], bg=self.colors["bg"]).pack(side=tk.LEFT, padx=(0, 6))

        if not hasattr(self, 'rec_id_var'):
            self.rec_id_var = tk.IntVar(value=1)

        tk.Spinbox(recid_frame, from_=1, to=9999, increment=1,
                   textvariable=self.rec_id_var,
                   width=6, font=('Segoe UI', 10)).pack(side=tk.LEFT)

        tk.Label(recid_frame,
                 text="  ← tentukan nomor rekaman sebelum menyimpan",
                 font=('Segoe UI', 9), fg=self.colors["text_dim"],
                 bg=self.colors["bg"]).pack(side=tk.LEFT, padx=8)

        # --- Treeview daftar fitur ---
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        

        tree = ttk.Treeview(tree_frame,
                            columns=("Feature", "Value"),
                            show='headings', height=25)
        tree.heading("Feature", text="Feature Name")
        tree.heading("Value",   text="Value")
        tree.column("Feature",  width=400)
        tree.column("Value",    width=200)

        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)

        # Isi treeview dari self.last_features
        if self.last_features:
            skip_keys = {'filename', 'subject', 'rec_id',
                         'stress', 'anxiety', 'depression', 'gender', 'seg_id'}
            for k, v in self.last_features.items():
                if k not in skip_keys:
                    val_str = f"{v:.6f}" if isinstance(v, float) else str(v)
                    tree.insert("", "end", values=(k, val_str))

        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # --- Tombol Save ---
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=10, pady=8)

        state = tk.NORMAL if self.last_features else tk.DISABLED
        self.btn_save_feat = tk.Button(
            btn_frame,
            text="💾  Save Features to CSV",
            command=self.save_features_to_csv,
            bg=self.colors["green"], fg="white",
            activebackground="#1e7e34", activeforeground="white",
            font=('Segoe UI', 10, 'bold'), relief=tk.FLAT,
            padx=20, pady=6,
            state=state
        )
        self.btn_save_feat.pack(side=tk.LEFT, padx=5)

        tk.Label(btn_frame,
                 text="Fitur akan ditambahkan ke baris baru di 'feature.csv'",
                 font=('Segoe UI', 9), fg=self.colors["text_dim"],
                 bg=self.colors["bg"]).pack(side=tk.LEFT, padx=15)
    
    def save_features_to_csv(self):
        """Simpan self.last_features ke feature.csv (append jika sudah ada)."""
        import os

        if not self.last_features:
            messagebox.showwarning("Tidak Ada Data",
                                   "Belum ada fitur yang diekstrak. Jalankan analisis dulu.")
            return

        csv_path = "feature.csv"

        try:
            # Ambil Rec ID dari spinbox dan masukkan ke baris sebelum disimpan
            self.last_features['rec_id'] = self.rec_id_var.get()

            new_row = pd.DataFrame([self.last_features])

            if os.path.exists(csv_path):
                existing = pd.read_csv(csv_path)
                # Gabungkan kolom: kolom baru yang tidak ada di existing akan diisi NaN
                combined = pd.concat([existing, new_row], ignore_index=True)
            else:
                combined = new_row

            # Susun urutan kolom: info dulu, lalu fitur, lalu label DASS
            info_cols  = ['filename', 'subject', 'rec_id', 'gender', 'seg_id']
            label_cols = ['stress', 'anxiety', 'depression']
            existing_info  = [c for c in info_cols  if c in combined.columns]
            existing_label = [c for c in label_cols if c in combined.columns]
            feat_cols = [c for c in combined.columns
                         if c not in existing_info + existing_label]
            combined = combined[existing_info + feat_cols + existing_label]

            combined.to_csv(csv_path, index=False)

            n_rows = len(combined)
            messagebox.showinfo(
                "Tersimpan ✓",
                f"Fitur berhasil disimpan!\n\n"
                f"File : {os.path.abspath(csv_path)}\n"
                f"Total baris di CSV: {n_rows}"
            )

            # Nonaktifkan tombol save setelah berhasil (agar tidak double-save)
            if self.btn_save_feat:
                self.btn_save_feat.config(state=tk.DISABLED,
                                          bg="#e8eaed", fg="#9aa0a6")
            # Reset last_features agar segmen berikutnya perlu Analyze dulu
            self.last_features = None

        except Exception as e:
            messagebox.showerror("Gagal Menyimpan", f"Error:\n{str(e)}")
         
    def build_signal_save_tab(self, parent):
        """Membangun tab Saving Signal: preview sinyal EEG+PPG preprocessed + tombol save CSV."""
        for widget in parent.winfo_children():
            widget.destroy()

        # --- Header info ---
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill=tk.X, padx=10, pady=6)

        has_data = (
            hasattr(self, 'analysis_results') and
            self.analysis_results and
            self.analysis_results.get('ica', {}).get('data') is not None
        )

        if has_data:
            info_text = (
                "Sinyal yang akan disimpan: EEG (ICA-cleaned, 8 ch) + PPG preprocessed. "
                "Setiap baris = 1 sampel (250 Hz). Kolom: rec_id, EEG_ch1..EEG_ch8, PPG."
            )
        else:
            info_text = "Belum ada sinyal — jalankan analisis terlebih dahulu."

        tk.Label(info_frame, text=info_text, font=('Segoe UI', 9),
                fg=self.colors["accent"], bg=self.colors["bg"],
                wraplength=900, justify='left').pack(side=tk.LEFT)

        # --- Input Rec ID ---
        recid_frame = ttk.Frame(parent)
        recid_frame.pack(fill=tk.X, padx=10, pady=4)

        tk.Label(recid_frame, text="Rec ID:", font=('Segoe UI', 10, 'bold'),
                fg=self.colors["text"], bg=self.colors["bg"]).pack(side=tk.LEFT, padx=(0, 6))

        if not hasattr(self, 'signal_rec_id_var'):
            self.signal_rec_id_var = tk.IntVar(value=1)

        tk.Spinbox(recid_frame, from_=1, to=9999, increment=1,
                textvariable=self.signal_rec_id_var,
                width=6, font=('Segoe UI', 10)).pack(side=tk.LEFT)

        tk.Label(recid_frame,
                text="  ← tentukan nomor rekaman sebelum menyimpan",
                font=('Segoe UI', 9), fg=self.colors["text_dim"],
                bg=self.colors["bg"]).pack(side=tk.LEFT, padx=8)

        # --- Preview tabel (Treeview, 20 baris pertama) ---
        preview_frame = ttk.LabelFrame(parent, text=" Preview (20 baris pertama) ", padding=6)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        ch_names = [f"EEG_ch{i+1}" for i in range(NUM_CHANNELS)]
        all_cols  = ["rec_id"] + ch_names + ["PPG"]

        tree = ttk.Treeview(preview_frame, columns=all_cols, show='headings', height=20)
        for col in all_cols:
            tree.heading(col, text=col)
            tree.column(col, width=90, anchor='center')

        sb = ttk.Scrollbar(preview_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)

        if has_data:
            eeg_clean = self.analysis_results['ica']['data']  # (8, N)
            bpm_data  = self.analysis_results.get('bpm_data', {})
            ppg_sig   = np.array(bpm_data.get('ppg_signal', []))

            n_samples = eeg_clean.shape[1]
            # Sejajarkan panjang PPG dengan EEG
            if len(ppg_sig) >= n_samples:
                ppg_aligned = ppg_sig[:n_samples]
            else:
                ppg_aligned = np.concatenate([ppg_sig,
                                            np.zeros(n_samples - len(ppg_sig))])

            rec_id_val = self.signal_rec_id_var.get()
            preview_n  = min(20, n_samples)
            for s in range(preview_n):
                eeg_vals = [f"{eeg_clean[ch, s]:.4f}" for ch in range(NUM_CHANNELS)]
                row_vals  = [str(rec_id_val)] + eeg_vals + [f"{ppg_aligned[s]:.4f}"]
                tree.insert("", "end", values=row_vals)

        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # --- Tombol Save ---
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=10, pady=8)

        state = tk.NORMAL if has_data else tk.DISABLED
        self.btn_save_signal = tk.Button(
            btn_frame,
            text="💾  Save Signal to CSV",
            command=self.save_signal_to_csv,
            bg=self.colors["accent"], fg="white",
            activebackground="#174ea6", activeforeground="white",
            font=('Segoe UI', 10, 'bold'), relief=tk.FLAT,
            padx=20, pady=6,
            state=state
        )
        self.btn_save_signal.pack(side=tk.LEFT, padx=5)

        tk.Label(btn_frame,
                text="Sinyal akan ditambahkan ke baris baru di 'signal.csv'",
                font=('Segoe UI', 9), fg=self.colors["text_dim"],
                bg=self.colors["bg"]).pack(side=tk.LEFT, padx=15)


    def save_signal_to_csv(self):
            """Simpan sinyal EEG-cleaned + PPG ke dalam folder processing_signal dengan format clean_<nama_file>.csv"""
            import os

            has_data = (
                hasattr(self, 'analysis_results') and
                self.analysis_results and
                self.analysis_results.get('ica', {}).get('data') is not None
            )
            if not has_data:
                messagebox.showwarning("Tidak Ada Data",
                                    "Belum ada sinyal. Jalankan analisis dulu.")
                return

            try:
                # 1. CEK & BUAT FOLDER: Jika folder 'processing_signal' belum ada, buat otomatis
                folder_name = "processing_signal"
                os.makedirs(folder_name, exist_ok=True)

                # =====================================================================
                # AMBIL NILAI SNR TERBARU: Dari hasil analisis/recompute terakhir
                # =====================================================================
                avg_snr = 0.0
                if hasattr(self, 'analysis_results') and self.analysis_results:
                    avg_snr = self.analysis_results.get('ica', {}).get('avg_snr', 0.0)
                # =====================================================================

                # 2. FORMAT PENAMAAN OTOMATIS BARU: Menambahkan (snr)dB ke dalam judul file
                if hasattr(self, 'last_filename') and self.last_filename:
                    base_name = os.path.basename(self.last_filename)  # contoh: "Abel1.txt"
                    name_without_ext, _ = os.path.splitext(base_name) # menghasilkan: "Abel1"
                    
                    # Format hasil akhir menjadi: clean_7.20dB_Abel1.csv
                    csv_filename = f"clean_{avg_snr:.2f}dB_{name_without_ext}.csv"     
                else:
                    # Antisipasi jika analisis dijalankan dari live streaming/recording
                    csv_filename = f"clean_{avg_snr:.2f}dB_recorded_signal_{int(time.time())}.csv"
                    
                # Gabungkan folder dengan nama file baru
                csv_path = os.path.join(folder_name, csv_filename)

                # 3. Ekstraksi data sinyal seperti biasa
                eeg_clean = self.analysis_results['ica']['data']  # (8, N)
                bpm_data  = self.analysis_results.get('bpm_data', {})
                ppg_sig   = np.array(bpm_data.get('ppg_signal', []))

                n_samples = eeg_clean.shape[1]
                if len(ppg_sig) >= n_samples:
                    ppg_aligned = ppg_sig[:n_samples]
                else:
                    ppg_aligned = np.concatenate([ppg_sig,
                                                np.zeros(n_samples - len(ppg_sig))])

                rec_id_val = self.signal_rec_id_var.get()

                # Susun struktur DataFrame
                ch_names = [f"EEG_ch{i+1}" for i in range(NUM_CHANNELS)]
                data_dict = {"rec_id": np.full(n_samples, rec_id_val, dtype=int)}
                for i, ch in enumerate(ch_names):
                    data_dict[ch] = eeg_clean[i]
                data_dict["PPG"] = ppg_aligned

                new_df = pd.DataFrame(data_dict)

                # 4. SIMPAN MANDIRI: Langsung disave ke file unik masing-masing (bukan di-append ke file global)
                new_df.to_csv(csv_path, index=False)
                n_rows = len(new_df)

                # Tampilkan pop-up informasi lokasi penyimpanan baru
                messagebox.showinfo(
                    "Tersimpan ✓",
                    f"Sinyal berhasil disimpan ke file terpisah!\n\n"
                    f"Folder : {folder_name}/\n"
                    f"File   : {csv_filename}\n"
                    f"Path   : {os.path.abspath(csv_path)}\n"
                    f"Sampel : {n_samples} baris berhasil diekspor."
                )

                # Nonaktifkan tombol agar user tahu proses simpan sukses dilakukan
                if self.btn_save_signal:
                    self.btn_save_signal.config(state=tk.DISABLED,
                                                bg="#e8eaed", fg="#9aa0a6")

            except Exception as e:
                messagebox.showerror("Gagal Menyimpan", f"Error:\n{str(e)}")                            
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Styling Notebook (Tabs) - Putih dengan indikator biru
        style.configure("TNotebook", background=self.colors["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", 
                        background="#e8f0fe", 
                        foreground=self.colors["accent"], 
                        padding=[20, 8], font=('Segoe UI', 9, 'bold'))
        style.map("TNotebook.Tab", 
                background=[("selected", self.colors["accent"])],
                foreground=[("selected", "#ffffff")])

        # Styling LabelFrame - Border biru tipis
        style.configure("TLabelframe", 
                        background=self.colors["bg"], 
                        foreground=self.colors["accent"])
        style.configure("TLabelframe.Label", 
                        background=self.colors["bg"], 
                        foreground=self.colors["accent"], 
                        font=('Segoe UI', 10, 'bold'))

        # Styling Label Umum
        style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["text"])
        


    def display_feature_table(self, df, parent):
            for widget in parent.winfo_children():
                widget.destroy()
            # Frame untuk Treeview
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.BOTH, expand=True)

            tree = ttk.Treeview(frame, columns=("Feature", "Value"), show='headings')
            tree.heading("Feature", text="Feature Name")
            tree.heading("Value", text="Value")
            
            # Scrollbar
            sb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=sb.set)

            for _, row in df.iterrows():
                val = f"{row['Value']:.4f}"
                tree.insert("", "end", values=(row['Feature Name'], val))

            tree.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")
                        
    def _load_normative_stats_dict(self, metric_kind):
        """
        Load statistik normatif ke bentuk:
            stats[(band, pair)] = {"mean": ..., "std": ..., "n": ...}

        metric_kind:
            "asymmetry_ai"
            "pli_proxy"
            "wpli_proxy"
            "imcoh_proxy"
        """
        import os
        import pandas as pd
        import numpy as np

        if metric_kind in self._norm_stats_cache:
            return self._norm_stats_cache[metric_kind]

        if metric_kind == "asymmetry_ai":
            csv_path = self.normative_asym_csv

        elif metric_kind in ["pli_proxy", "wpli_proxy", "imcoh_proxy"]:
            csv_path = self.normative_pli_csv

        else:
            raise ValueError(f"metric_kind tidak dikenal: {metric_kind}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File normatif tidak ditemukan:\n{csv_path}")

        df = pd.read_csv(csv_path)

        print(f"[NORM LOAD] metric={metric_kind}")
        print(f"[NORM LOAD] csv={csv_path}")
        print(f"[NORM LOAD] columns={list(df.columns)}")
        print(f"[NORM LOAD] rows before filter={len(df)}")

        # Jika file berisi banyak metric, filter sesuai metric_kind
        if "metric" in df.columns:
            df = df[df["metric"].astype(str).str.strip().eq(metric_kind)].copy()
            print(f"[NORM LOAD] rows after metric filter={len(df)}")

        if len(df) == 0:
            print(f"[NORM LOAD WARNING] Tidak ada baris untuk metric={metric_kind}")

        # Deteksi nama kolom mean/std secara fleksibel
        if "mean_value" in df.columns:
            mean_col = "mean_value"
        elif "mean_ai" in df.columns:
            mean_col = "mean_ai"
        else:
            raise KeyError(
                "Kolom mean tidak ditemukan. Harus ada mean_value atau mean_ai."
            )

        if "std_value" in df.columns:
            std_col = "std_value"
        elif "std_ai" in df.columns:
            std_col = "std_ai"
        else:
            raise KeyError(
                "Kolom std tidak ditemukan. Harus ada std_value atau std_ai."
            )

        stats = {}

        for _, row in df.iterrows():
            band = str(row["band"]).strip()
            pair = str(row["pair"]).strip()

            mu = float(row[mean_col]) if pd.notna(row[mean_col]) else np.nan
            sd = float(row[std_col]) if pd.notna(row[std_col]) else np.nan
            n = int(row["n"]) if "n" in df.columns and pd.notna(row["n"]) else 0

            stats[(band, pair)] = {
                "mean": mu,
                "std": sd,
                "n": n
            }

        print(f"[NORM LOAD] valid stats keys={len(stats)}")
        print(f"[NORM LOAD] example keys={list(stats.keys())[:8]}")

        self._norm_stats_cache[metric_kind] = stats
        return stats

    def _avg_feat_compute_ai_pair_dict(self, power_vec, ch_names=None):
        """
        power_vec: array 8 channel [Fp1,Fp2,F3,F4,T3,T4,P3,P4]
        return: dict pair -> AI
        """
        import numpy as np
        from itertools import combinations

        if ch_names is None:
            ch_names = list(self.avg_topo_ch_names)

        p = np.asarray(power_vec, dtype=float).reshape(-1)
        out = {}

        for ch1, ch2 in combinations(ch_names, 2):
            i = ch_names.index(ch1)
            j = ch_names.index(ch2)

            if i >= len(p) or j >= len(p):
                out[f"{ch1}-{ch2}"] = np.nan
                continue

            p1, p2 = p[i], p[j]

            if not np.isfinite(p1) or not np.isfinite(p2):
                out[f"{ch1}-{ch2}"] = np.nan
                continue

            denom = p1 + p2
            if abs(denom) < 1e-12:
                out[f"{ch1}-{ch2}"] = np.nan
            else:
                out[f"{ch1}-{ch2}"] = float((p1 - p2) / denom)

        return out

    def _avg_feat_build_zscore_pair_dict(self, band_name, pair_value_dict, metric_kind):
        """
        Ubah dict pair->raw_value menjadi pair->z_score
        berdasarkan statistik normatif.
        """
        import numpy as np

        band_name = str(band_name).strip()

        stats = self._load_normative_stats_dict(metric_kind)
        out = {}

        n_raw = 0
        n_found = 0
        n_valid = 0
        missing_examples = []

        for pair, raw_val in pair_value_dict.items():
            n_raw += 1
            pair = str(pair).strip()

            if not np.isfinite(raw_val):
                out[pair] = np.nan
                continue

            st = stats.get((band_name, pair), None)

            if st is None:
                out[pair] = np.nan
                if len(missing_examples) < 5:
                    missing_examples.append((band_name, pair))
                continue

            n_found += 1

            mu = st["mean"]
            sd = st["std"]

            if (not np.isfinite(mu)) or (not np.isfinite(sd)) or (sd < 1e-12):
                out[pair] = np.nan
                continue

            z = (float(raw_val) - mu) / sd
            out[pair] = float(z)
            n_valid += 1

        print(
            f"[Z BUILD] metric={metric_kind} | band={band_name} | "
            f"raw_pairs={n_raw} | found_stats={n_found} | valid_z={n_valid}"
        )

        if len(missing_examples) > 0:
            print(f"[Z BUILD] missing examples: {missing_examples}")

        return out
    def _build_avg_feature_topomap_tab(self):
        """
        Tab baru:
        Load CSV fitur agregat, lalu buat topomap rata-rata fitur berdasarkan
        status normal/tidak normal DASS untuk stress, anxiety, depression.
        """
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox

        for w in self.avg_feature_topo_tab.winfo_children():
            w.destroy()

        # ============================================================
        # STATE VARIABLE
        # ============================================================
        if not hasattr(self, "avg_feat_csv_path_var"):
            self.avg_feat_csv_path_var = tk.StringVar(value="Belum ada file fitur")
        if not hasattr(self, "avg_feat_status_var"):
            self.avg_feat_status_var = tk.StringVar(value="Load file fitur terlebih dahulu.")
        if not hasattr(self, "avg_feat_abnormal_vars"):
            self.avg_feat_abnormal_vars = {
                "stress": tk.BooleanVar(value=True),
                "anxiety": tk.BooleanVar(value=True),
                "depression": tk.BooleanVar(value=True),
            }
        if not hasattr(self, "avg_feat_subject_balanced_var"):
            self.avg_feat_subject_balanced_var = tk.BooleanVar(value=True)

        self.avg_feature_df = None

        # ============================================================
        # LAYOUT UTAMA
        # ============================================================
        container = ttk.Frame(self.avg_feature_topo_tab)
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        top_frame = ttk.LabelFrame(container, text="Average Topomap from Feature CSV", padding=8)
        top_frame.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(
            top_frame,
            textvariable=self.avg_feat_csv_path_var,
            foreground="gray"
        ).pack(side=tk.LEFT, padx=(0, 8), fill=tk.X, expand=True)

        ttk.Button(
            top_frame,
            text="Load Feature CSV",
            command=self._avg_feat_load_csv
        ).pack(side=tk.LEFT, padx=4)

        ttk.Checkbutton(
            top_frame,
            text="Subject-balanced average",
            variable=self.avg_feat_subject_balanced_var
        ).pack(side=tk.LEFT, padx=8)

        # ============================================================
        # TOMBOL PENYAKIT
        # ============================================================
        btn_frame = ttk.LabelFrame(container, text="Disease Group Selection", padding=8)
        btn_frame.pack(fill=tk.X, pady=(0, 6))

        # Stress
        stress_box = ttk.Frame(btn_frame)
        stress_box.pack(side=tk.LEFT, padx=8)

        ttk.Checkbutton(
            stress_box,
            text="Tidak normal",
            variable=self.avg_feat_abnormal_vars["stress"]
        ).pack(side=tk.TOP, anchor="w")

        ttk.Button(
            stress_box,
            text="Topomap Stress",
            command=lambda: self._avg_feat_run_topomap("stress")
        ).pack(side=tk.TOP, fill=tk.X, pady=2)

        # Anxiety
        anxiety_box = ttk.Frame(btn_frame)
        anxiety_box.pack(side=tk.LEFT, padx=8)

        ttk.Checkbutton(
            anxiety_box,
            text="Tidak normal",
            variable=self.avg_feat_abnormal_vars["anxiety"]
        ).pack(side=tk.TOP, anchor="w")

        ttk.Button(
            anxiety_box,
            text="Topomap Anxiety",
            command=lambda: self._avg_feat_run_topomap("anxiety")
        ).pack(side=tk.TOP, fill=tk.X, pady=2)

        # Depression
        depression_box = ttk.Frame(btn_frame)
        depression_box.pack(side=tk.LEFT, padx=8)

        ttk.Checkbutton(
            depression_box,
            text="Tidak normal",
            variable=self.avg_feat_abnormal_vars["depression"]
        ).pack(side=tk.TOP, anchor="w")

        ttk.Button(
            depression_box,
            text="Topomap Depression",
            command=lambda: self._avg_feat_run_topomap("depression")
        ).pack(side=tk.TOP, fill=tk.X, pady=2)

        ttk.Label(
            btn_frame,
            text="Catatan: checklist ON = DASS tidak normal, checklist OFF = DASS normal.",
            foreground="gray"
        ).pack(side=tk.LEFT, padx=16)

        # ============================================================
        # STATUS
        # ============================================================
        ttk.Label(
            container,
            textvariable=self.avg_feat_status_var,
            foreground="green"
        ).pack(fill=tk.X, pady=(0, 6))

        # ============================================================
        # RESULT AREA
        # ============================================================
        self.avg_feat_result_notebook = ttk.Notebook(container)
        self.avg_feat_result_notebook.pack(fill=tk.BOTH, expand=True)


    def _avg_feat_load_csv(self):
        """
        Load file CSV fitur.
        """
        from tkinter import filedialog, messagebox
        import pandas as pd
        import os

        path = filedialog.askopenfilename(
            title="Pilih file fitur CSV",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        try:
            df = pd.read_csv(path)

            required_labels = ["stress", "anxiety", "depression"]
            missing_labels = [c for c in required_labels if c not in df.columns]
            if missing_labels:
                raise ValueError(
                    f"Kolom label DASS tidak ditemukan: {missing_labels}\n"
                    f"Kolom tersedia: {list(df.columns)}"
                )

            self.avg_feature_df = df
            self.avg_feat_csv_path_var.set(os.path.basename(path))

            msg = (
                f"File fitur berhasil dimuat | "
                f"baris={len(df)} | kolom={len(df.columns)}"
            )
            self.avg_feat_status_var.set(msg)
            print("[AVG FEATURE]", msg)

        except Exception as e:
            messagebox.showerror("Gagal Load Feature CSV", str(e))
            self.avg_feat_status_var.set("Gagal load file fitur.")


    def _avg_feat_get_mean_row(self, disease):
        """
        Filter data berdasarkan DASS lalu hitung rata-rata fitur.

        Jika subject-balanced aktif:
        1. rata-rata dulu per subject,
        2. baru rata-rata antar-subject.
        Ini mencegah subject dengan segmen lebih banyak mendominasi hasil.
        """
        import numpy as np
        import pandas as pd

        if not hasattr(self, "avg_feature_df") or self.avg_feature_df is None:
            raise RuntimeError("Belum ada file fitur. Klik Load Feature CSV dulu.")

        df = self.avg_feature_df.copy()

        cutoff_map = {
            "stress": 14,
            "anxiety": 7,
            "depression": 9,
        }

        if disease not in cutoff_map:
            raise ValueError(f"Disease tidak dikenali: {disease}")

        if disease not in df.columns:
            raise ValueError(f"Kolom {disease} tidak ditemukan di CSV.")

        cutoff = cutoff_map[disease]
        abnormal = self.avg_feat_abnormal_vars[disease].get()

        score = pd.to_numeric(df[disease], errors="coerce")

        if abnormal:
            mask = score > cutoff
            group_name = f"{disease.upper()} TIDAK NORMAL"
            rule = f"{disease} > {cutoff}"
        else:
            mask = score <= cutoff
            group_name = f"{disease.upper()} NORMAL"
            rule = f"{disease} <= {cutoff}"

        selected = df.loc[mask].copy()

        if len(selected) == 0:
            raise RuntimeError(
                f"Tidak ada data untuk grup {group_name} dengan aturan {rule}."
            )

        # Paksa semua kolom numerik yang memungkinkan
        for col in selected.columns:
            if col not in ["filename", "subject", "gender"]:
                selected[col] = pd.to_numeric(selected[col], errors="coerce")

        numeric_cols = selected.select_dtypes(include=[np.number]).columns.tolist()

        
        exclude_cols = {
            "rec_id",
            "seg_id",
            "snr_db",
            "n_ic_rejected",
            "stress",
            "anxiety",
            "depression",
        }

        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

        if self.avg_feat_subject_balanced_var.get() and "subject" in selected.columns:
            subj_mean = selected.groupby("subject")[numeric_cols].mean(numeric_only=True)
            mean_row = subj_mean.mean(axis=0)
            n_subject = subj_mean.shape[0]
            n_segment = len(selected)
            avg_mode = "subject-balanced"
        else:
            mean_row = selected[numeric_cols].mean(axis=0, numeric_only=True)
            n_subject = selected["subject"].nunique() if "subject" in selected.columns else "-"
            n_segment = len(selected)
            avg_mode = "segment-average"

        info = {
            "disease": disease,
            "cutoff": cutoff,
            "abnormal": abnormal,
            "group_name": group_name,
            "rule": rule,
            "n_segment": n_segment,
            "n_subject": n_subject,
            "avg_mode": avg_mode,
        }

        return mean_row, info


    def _avg_feat_value(self, mean_row, col_name):
        """
        Ambil nilai fitur dari mean_row dengan fallback T3/T4 <-> T7/T8.
        """
        import numpy as np

        candidates = [
            col_name,
            col_name.replace("_T3", "_T7"),
            col_name.replace("_T4", "_T8"),
            col_name.replace("-T3", "-T7"),
            col_name.replace("-T4", "-T8"),
            col_name.replace("T3-", "T7-"),
            col_name.replace("T4-", "T8-"),
        ]

        for c in candidates:
            if c in mean_row.index:
                try:
                    return float(mean_row[c])
                except Exception:
                    return np.nan

        return np.nan


    def _avg_feat_channel_values(self, mean_row, feature_type, name):
        """
        Ubah feature mean menjadi vector 8 channel:
        [Fp1, Fp2, F3, F4, T3, T4, P3, P4]
        """
        import numpy as np

        ch_names = ["Fp1", "Fp2", "F3", "F4", "T3", "T4", "P3", "P4"]
        vals = []

        for ch in ch_names:
            if feature_type in ["AbsPow", "RelPow"]:
                col = f"{feature_type}_{name}_{ch}"

            elif feature_type == "Ratio":
                col = f"Ratio_{name}_{ch}"

            elif feature_type == "SASI":
                col = f"SASI_{ch}"

            elif feature_type == "LZC":
                if name == "Broadband":
                    col = f"LZC_{ch}"
                else:
                    col = f"LZC_{name}_{ch}"

            else:
                col = ""

            vals.append(self._avg_feat_value(mean_row, col))

        vals = np.asarray(vals, dtype=float)
        return vals


    def _avg_feat_asym_values(self, mean_row, band_name):
        """
        Asymmetry index awalnya berbasis pasangan:
            Asym_Alpha_Fp1-Fp2
            Asym_Alpha_F3-F4
            Asym_Alpha_T3-T4
            Asym_Alpha_P3-P4

        Agar bisa diplot sebagai topomap 8 titik:
            channel kiri  = +nilai asym
            channel kanan = -nilai asym

        Artinya:
            warna positif = dominan/lebih tinggi di sisi kiri pasangan
            warna negatif = dominan/lebih tinggi di sisi kanan pasangan
        """
        import numpy as np

        ch_names = ["Fp1", "Fp2", "F3", "F4", "T3", "T4", "P3", "P4"]
        out = {ch: np.nan for ch in ch_names}

        pairs = [
            ("Fp1", "Fp2"),
            ("F3", "F4"),
            ("T3", "T4"),
            ("P3", "P4"),
        ]

        for left, right in pairs:
            col = f"Asym_{band_name}_{left}-{right}"
            val = self._avg_feat_value(mean_row, col)

            out[left] = val
            out[right] = -val if np.isfinite(val) else np.nan

        return np.asarray([out[ch] for ch in ch_names], dtype=float)

    def _avg_feat_plot_asym_connectivity_group(
        self,
        map_group,
        parent,
        title,
        threshold=0.20
    ):
        """
        Plot Asymmetry Index pada Avg Feature Topomap sebagai connectivity map,
        bukan topomap warna.

        map_group:
            dict band -> vector AbsPow 8 channel
            contoh:
            {
                "Delta": [Fp1,Fp2,F3,F4,T3,T4,P3,P4],
                "Theta": [...],
                ...
            }

        Logika garis sama seperti tab Brain Mapping:
            val = (p_i - p_j) / (p_i + p_j)
            coh = abs(val)
            jika coh > threshold, gambar garis
            merah = val positif
            biru  = val negatif
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import mne
        from itertools import combinations
        from matplotlib.lines import Line2D
        from mne.channels.layout import _find_topomap_coords

        for widget in parent.winfo_children():
            widget.destroy()

        band_names = list(map_group.keys())
        n_bands = len(band_names)

        if n_bands == 0:
            ttk.Label(
                parent,
                text="Tidak ada data Asymmetry Connectivity.",
                foreground="gray"
            ).pack(padx=20, pady=20)
            return

        # Urutan GUI:
        # Fp1, Fp2, F3, F4, T3, T4, P3, P4
        ch_names_gui = ["Fp1", "Fp2", "F3", "F4", "T3", "T4", "P3", "P4"]

        # Untuk MNE montage, T3/T4 dipetakan ke T7/T8.
        info = self._avg_feat_get_mne_info()
        ch_names_mne = list(info["ch_names"])

        pos2d = _find_topomap_coords(
            info,
            picks=np.arange(len(ch_names_mne))
        )

        # Untuk label tampilan tetap pakai T3/T4
        ch_index = {ch: i for i, ch in enumerate(ch_names_gui)}
        all_pairs = list(combinations(ch_names_gui, 2))

        cols_a = min(n_bands, 4)
        rows_a = int(np.ceil(n_bands / cols_a))

        fig, axs = plt.subplots(
            rows_a,
            cols_a,
            figsize=(3.7 * cols_a, 4.0 * rows_a),
            squeeze=False
        )

        fig.suptitle(
            title,
            fontsize=13,
            fontweight="bold",
            y=0.98
        )

        for b_idx, band in enumerate(band_names):
            ax = axs[b_idx // cols_a][b_idx % cols_a]

            ax.set_title(
                f"Asym {band}",
                fontsize=9,
                fontweight="bold",
                pad=8
            )

            # Kepala + titik sensor, tanpa topomap warna
            mne.viz.plot_sensors(
                info,
                axes=ax,
                show=False,
                kind="topomap"
            )

            p = np.asarray(map_group[band], dtype=float).reshape(-1)

            # Samakan panjang ke 8 channel
            if len(p) > 8:
                p = p[:8]
            elif len(p) < 8:
                p = np.pad(p, (0, 8 - len(p)), mode="constant", constant_values=np.nan)

            n_lines = 0

            for ch1, ch2 in all_pairs:
                i = ch_index[ch1]
                j = ch_index[ch2]

                if i >= len(p) or j >= len(p):
                    continue

                if not np.isfinite(p[i]) or not np.isfinite(p[j]):
                    continue

                denom = p[i] + p[j]

                if abs(denom) < 1e-12:
                    continue

                val = (p[i] - p[j]) / denom
                coh = abs(val)

                if coh > threshold:
                    x1, y1 = pos2d[i]
                    x2, y2 = pos2d[j]

                    line_color = "red" if val > 0 else "blue"

                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color=line_color,
                        linewidth=0.5 + 1.5 * coh,
                        alpha=0.75,
                        solid_capstyle="round"
                    )

                    n_lines += 1

            ax.text(
                0.5,
                -0.08,
                f"|AI| > {threshold:.2f} | lines={n_lines}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8,
                color="gray"
            )

        # Matikan axes kosong
        for b_idx in range(n_bands, rows_a * cols_a):
            axs[b_idx // cols_a][b_idx % cols_a].axis("off")

        # Legend merah/biru
        legend_elements = [
            Line2D([0], [0], color="blue", lw=2, label="(-) Asimetri"),
            Line2D([0], [0], color="red", lw=2, label="(+) Asimetri"),
        ]

        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=2,
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, 0.035)
        )

        # Bar skala sederhana seperti Brain Mapping
        ax_bar = fig.add_axes([0.40, 0.075, 0.20, 0.015])
        cmap_scale = mcolors.ListedColormap(["blue", "red"])
        norm_scale = mcolors.Normalize(vmin=-1, vmax=1)

        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm_scale, cmap=cmap_scale),
            cax=ax_bar,
            orientation="horizontal",
            ticks=[]
        )
        cb.outline.set_visible(False)

        ax_bar.text(
            -0.2,
            0.5,
            "-",
            transform=ax_bar.transAxes,
            va="center",
            ha="right",
            fontsize=14,
            fontweight="bold",
            color="blue"
        )

        ax_bar.text(
            1.2,
            0.5,
            "+",
            transform=ax_bar.transAxes,
            va="center",
            ha="left",
            fontsize=12,
            fontweight="bold",
            color="red"
        )

        fig.subplots_adjust(
            left=0.04,
            right=0.98,
            bottom=0.13,
            top=0.88,
            wspace=0.25,
            hspace=0.35
        )

        self._avg_feat_embed_canvas(fig, parent)
        
    def _avg_feat_ori_conn_pli_values(self, mean_row, band_name):
        """
        Ambil nilai ORI_CONN_PLI dari mean_row untuk satu band.

        Format kolom yang dicari:
            ORI_CONN_PLI_Delta_Fp1-Fp2
            ORI_CONN_PLI_Theta_Fp1-F3
            ORI_CONN_PLI_Alpha_F3-F4
            ORI_CONN_PLI_Beta_Fp2-F4

        Return:
            dict pair -> value
            contoh:
            {
                "Fp1-Fp2": 0.31,
                "Fp1-F3" : 0.22,
                ...
            }
        """
        import numpy as np

        pair_values = {}

        prefix = f"ORI_CONN_PLI_{band_name}_"

        for col in mean_row.index:
            col_str = str(col)

            if not col_str.startswith(prefix):
                continue

            pair = col_str.replace(prefix, "")

            try:
                val = float(mean_row[col])
            except Exception:
                val = np.nan

            pair_values[pair] = val

        return pair_values

    def _avg_feat_make_all_maps(self, mean_row):
        """
        Buat semua map fitur yang akan diplot.
        """
        maps = {}

        bands = ["Delta", "Theta", "Alpha", "Beta"]
        ratio_names = [
            "DeltaTheta",
            "DeltaAlpha",
            "DeltaBeta",
            "ThetaAlpha",
            "ThetaBeta",
            "AlphaBeta",
        ]

        # 1. Absolute Power
        maps["Abs Power"] = {
            band: self._avg_feat_channel_values(mean_row, "AbsPow", band)
            for band in bands
        }

        # 2. Relative Power
        maps["Relative Power"] = {
            band: self._avg_feat_channel_values(mean_row, "RelPow", band)
            for band in bands
        }

        # 3. Ratio Power
        maps["Ratio Power"] = {
            ratio: self._avg_feat_channel_values(mean_row, "Ratio", ratio)
            for ratio in ratio_names
        }

        # 4. SASI
        maps["SASI"] = {
            "SASI": self._avg_feat_channel_values(mean_row, "SASI", "SASI")
        }

        # 5. Asymmetry Index
        # 5. Asymmetry Index
        maps["Asymmetry Index"] = {
            band: self._avg_feat_channel_values(mean_row, "AbsPow", band)
            for band in bands
        }

        # 6. LZC Topomap
        # 6. LZC Topomap
        lzc_names = ["Broadband", "Delta", "Theta", "Alpha", "Beta"]
        maps["LZC Topomap"] = {
            name: self._avg_feat_channel_values(mean_row, "LZC", name)
            for name in lzc_names
        }

        # 7. ORI Connectivity PLI
        # 7. PLI
        maps["PLI"] = {
            band: self._avg_feat_ori_conn_pli_values(mean_row, band)
            for band in bands
        }

        return maps

    def _avg_feat_plot_zscore_connectivity_group(
        self,
        map_group,
        parent,
        title,
        metric_kind="asymmetry_ai"
    ):
        """
        Plot connectivity map berbasis Z-score normatif.
        
        metric_kind:
            'asymmetry_ai'  -> map_group[band] = vector 8 channel power
            'pli_proxy'     -> map_group[band] = dict pair -> raw PLI
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import mne
        from itertools import combinations
        from matplotlib.lines import Line2D
        from mne.channels.layout import _find_topomap_coords

        for widget in parent.winfo_children():
            widget.destroy()

        band_names = list(map_group.keys())
        n_bands = len(band_names)

        if n_bands == 0:
            ttk.Label(parent, text="Tidak ada data.", foreground="gray").pack(padx=20, pady=20)
            return

        # Info MNE
        info = self._avg_feat_get_mne_info()
        pos2d = _find_topomap_coords(info, picks=np.arange(len(info["ch_names"])))

        ch_names = list(self.avg_topo_ch_names)
        ch_index = {ch: i for i, ch in enumerate(ch_names)}

        cols = min(n_bands, 4)
        rows = int(np.ceil(n_bands / cols))

        fig, axs = plt.subplots(
            rows, cols,
            figsize=(3.9 * cols, 4.0 * rows),
            squeeze=False
        )

        z_thr = float(self.avg_zscore_threshold)
        z_cap = float(self.avg_zscore_cap)

        main_title = f"{title}\nZ-score threshold |Z| > {z_thr:.2f}"
        fig.suptitle(main_title, fontsize=13, fontweight="bold", y=0.97)

        for b_idx, band in enumerate(band_names):
            ax = axs[b_idx // cols][b_idx % cols]
            ax.set_title(str(band), fontsize=10, fontweight="bold", pad=8)

            # Gambar kepala/sensor
            mne.viz.plot_sensors(
                info,
                axes=ax,
                show=False,
                kind="topomap"
            )
            # --------------------------------------------------------
            # Ambil raw pair value
            # --------------------------------------------------------
            raw_pair_dict = {}

            if metric_kind == "asymmetry_ai":
                # map_group[band] harus berisi power vector 8 channel
                power_vec = np.asarray(map_group[band], dtype=float).reshape(-1)

                raw_pair_dict = self._avg_feat_compute_ai_pair_dict(
                    power_vec,
                    ch_names=ch_names
                )

            elif metric_kind in ["pli_proxy", "wpli_proxy", "imcoh_proxy"]:
                # map_group[band] berisi dict pair -> nilai ORI_CONN_PLI user
                if isinstance(map_group[band], dict):
                    raw_pair_dict = dict(map_group[band])
                else:
                    raw_pair_dict = {}

            else:
                raise ValueError(f"metric_kind tidak dikenal: {metric_kind}")
            # --------------------------------------------------------
            # Ubah menjadi Z-score
            # --------------------------------------------------------
            z_pair_dict = self._avg_feat_build_zscore_pair_dict(
                band_name=str(band),
                pair_value_dict=raw_pair_dict,
                metric_kind=metric_kind
            )
            z_vals_dbg = np.array(
                [v for v in z_pair_dict.values() if np.isfinite(v)],
                dtype=float
            )

            if len(z_vals_dbg) > 0:
                print(
                    f"[AVG Z DEBUG] metric={metric_kind} | band={band} | "
                    f"minZ={np.min(z_vals_dbg):.3f} | "
                    f"maxZ={np.max(z_vals_dbg):.3f} | "
                    f"maxAbsZ={np.max(np.abs(z_vals_dbg)):.3f} | "
                    f"n={len(z_vals_dbg)}"
                )
            else:
                print(
                    f"[AVG Z DEBUG] metric={metric_kind} | band={band} | "
                    f"tidak ada Z valid"
                )

            n_lines = 0

            for pair, z in z_pair_dict.items():
                if not np.isfinite(z):
                    continue

                if abs(z) < z_thr:
                    continue

                if "-" not in pair:
                    continue

                ch1, ch2 = pair.split("-", 1)

                if ch1 not in ch_index or ch2 not in ch_index:
                    continue

                i = ch_index[ch1]
                j = ch_index[ch2]

                if i >= len(pos2d) or j >= len(pos2d):
                    continue

                x1, y1 = pos2d[i]
                x2, y2 = pos2d[j]

                # MERAH / BIRU saja
                line_color = "red" if z > 0 else "blue"

                # Ketebalan garis berdasarkan |Z|
                strength = min(abs(z) / z_cap, 1.0)
                lw = 0.8 + 2.8 * strength

                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=line_color,
                    linewidth=lw,
                    alpha=0.85,
                    solid_capstyle="round"
                )

                n_lines += 1

            ax.text(
                0.5,
                -0.08,
                f"|Z| > {z_thr:.2f} | lines={n_lines}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8,
                color="gray"
            )

        # axes kosong dimatikan
        for b_idx in range(n_bands, rows * cols):
            axs[b_idx // cols][b_idx % cols].axis("off")

        # Legend merah-biru
        legend_elements = [
            Line2D([0], [0], color="blue", lw=2, label=f"Z < -{z_thr:.2f}"),
            Line2D([0], [0], color="red",  lw=2, label=f"Z > +{z_thr:.2f}")
        ]

        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=2,
            fontsize=9,
            frameon=False,
            bbox_to_anchor=(0.5, 0.03)
        )

        # Teks keterangan
        if metric_kind == "asymmetry_ai":
            note = "Blue = asymmetry lebih rendah dari normatif | Red = lebih tinggi dari normatif"
        else:
            note = "Blue = PLI lebih rendah dari normatif | Red = lebih tinggi dari normatif"

        fig.text(
            0.5, 0.01,
            note,
            ha="center", va="bottom",
            fontsize=8, color="gray"
        )

        fig.subplots_adjust(
            left=0.04,
            right=0.98,
            bottom=0.12,
            top=0.84,
            wspace=0.25,
            hspace=0.35
        )

        self._avg_feat_embed_canvas(fig, parent)
        
    def _avg_feat_compute_vlims_like_brainmap(self, all_maps):
        """
        Hitung skala warna average-feature topomap dengan logika yang sama
        seperti tab Brain Mapping:
        - Abs Power      : min-max per band
        - Relative Power : min-max per band
        - Ratio Power    : min-max per ratio
        - Asymmetry      : simetris -max_abs sampai +max_abs
        - SASI           : simetris -max_abs sampai +max_abs
        - LZC            : min-max per map
        """
        import numpy as np

        def finite_range(vals, symmetric=False):
            arr = np.asarray(vals, dtype=float).reshape(-1)
            arr = arr[np.isfinite(arr)]

            if arr.size == 0:
                return (0.0, 1.0)

            if symmetric:
                m = float(np.max(np.abs(arr)))
                if m < 1e-12:
                    m = 1.0
                return (-m, m)

            vmin = float(np.min(arr))
            vmax = float(np.max(arr))

            if abs(vmax - vmin) < 1e-12:
                pad = abs(vmax) * 0.1 + 1e-6
                vmin -= pad
                vmax += pad

            return (vmin, vmax)

        vlims = {
            "abs_power": {},
            "rel_power": {},
            "ratios": {},
            "asym_by_band": {},
            "sasi": (-1.0, 1.0),
            "lzc_maps": {},
        }

        for name, vals in all_maps.get("Abs Power", {}).items():
            vlims["abs_power"][name] = finite_range(vals, symmetric=False)

        for name, vals in all_maps.get("Relative Power", {}).items():
            vlims["rel_power"][name] = finite_range(vals, symmetric=False)

        for name, vals in all_maps.get("Ratio Power", {}).items():
            vlims["ratios"][name] = finite_range(vals, symmetric=False)

        for name, vals in all_maps.get("Asymmetry Index", {}).items():
            vlims["asym_by_band"][name] = finite_range(vals, symmetric=True)

        if "SASI" in all_maps and "SASI" in all_maps["SASI"]:
            vlims["sasi"] = finite_range(all_maps["SASI"]["SASI"], symmetric=True)

        for name, vals in all_maps.get("LZC Topomap", {}).items():
            vlims["lzc_maps"][name] = finite_range(vals, symmetric=False)

        return vlims


    def _avg_feat_get_mne_info(self):
        """
        Info MNE untuk topomap 8 channel.
        Catatan:
        T3/T4 diposisikan sebagai T7/T8 karena standard_1020 MNE
        lebih stabil memakai T7/T8 sebagai ekuivalen temporal kiri/kanan.
        """
        import mne

        mne_ch_names = ["Fp1", "Fp2", "F3", "F4", "T7", "T8", "P3", "P4"]

        info = mne.create_info(
            ch_names=mne_ch_names,
            sfreq=250,
            ch_types="eeg"
        )

        montage = mne.channels.make_standard_montage("standard_1020")
        info.set_montage(montage, on_missing="ignore")

        return info


    def _avg_feat_safe_vlim(self, values, symmetric=False):
        """
        Buat skala warna aman.
        """
        import numpy as np

        values = np.asarray(values, dtype=float)
        valid = values[np.isfinite(values)]

        if len(valid) == 0:
            return -1.0, 1.0

        if symmetric:
            vmax = np.nanmax(np.abs(valid))
            if vmax <= 1e-12:
                vmax = 1.0
            return -vmax, vmax

        vmin = np.nanmin(valid)
        vmax = np.nanmax(valid)

        if abs(vmax - vmin) <= 1e-12:
            pad = abs(vmax) * 0.1 + 1e-6
            vmin -= pad
            vmax += pad

        return float(vmin), float(vmax)


    def _avg_feat_embed_canvas(self, fig, parent):
        """
        Embed figure ke frame Tkinter dengan toolbar.
        Dibuat terpisah agar tidak tergantung _embed_canvas lama.
        """
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        import matplotlib.pyplot as plt
        import tkinter as tk
        from tkinter import ttk

        for w in parent.winfo_children():
            w.destroy()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()

        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)

        try:
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(side=tk.LEFT, fill=tk.X)
        except TypeError:
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()

        plt.close(fig)

    def _avg_feat_plot_pli_connectivity_group(
        self,
        map_group,
        parent,
        title,
        threshold=0.20
    ):
        """
        Plot ORI_CONN_PLI sebagai connectivity map.

        map_group:
            dict band -> dict pair -> value

            contoh:
            {
                "Delta": {
                    "Fp1-Fp2": 0.31,
                    "Fp1-F3" : 0.22,
                    ...
                },
                "Theta": {...}
            }

        Logika:
            - PLI bernilai 0 sampai 1
            - garis digambar jika PLI >= threshold
            - warna garis mengikuti nilai PLI
            - linewidth makin besar jika PLI makin besar
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import mne
        from mne.channels.layout import _find_topomap_coords

        for widget in parent.winfo_children():
            widget.destroy()

        band_names = list(map_group.keys())
        n_bands = len(band_names)

        if n_bands == 0:
            ttk.Label(
                parent,
                text="Tidak ada data PLI.",
                foreground="gray"
            ).pack(padx=20, pady=20)
            return

        # Ambil info/montage yang sama dengan Avg Feature Topomap
        info = self._avg_feat_get_mne_info()
        ch_names_mne = list(info["ch_names"])

        # Label GUI tetap pakai T3/T4
        ch_names_gui = ["Fp1", "Fp2", "F3", "F4", "T3", "T4", "P3", "P4"]

        # Koordinat sensor
        pos2d = _find_topomap_coords(
            info,
            picks=np.arange(len(ch_names_mne))
        )

        ch_to_idx = {ch: i for i, ch in enumerate(ch_names_gui)}

        # Fallback T7/T8 kalau suatu saat pair memakai T7/T8
        alias = {
            "T7": "T3",
            "T8": "T4",
        }

        def normalize_ch_name(ch):
            ch = str(ch).strip()
            return alias.get(ch, ch)

        cols_a = min(n_bands, 4)
        rows_a = int(np.ceil(n_bands / cols_a))

        fig, axs = plt.subplots(
            rows_a,
            cols_a,
            figsize=(3.7 * cols_a, 4.0 * rows_a),
            squeeze=False
        )

        fig.patch.set_facecolor("white")

        fig.suptitle(
            title,
            fontsize=13,
            fontweight="bold",
            y=0.98
        )

        cmap = plt.get_cmap("jet")
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

        for b_idx, band in enumerate(band_names):
            ax = axs[b_idx // cols_a][b_idx % cols_a]

            ax.set_title(
                f"ORI_CONN_PLI {band}",
                fontsize=9,
                fontweight="bold",
                pad=8
            )

            # Gambar kepala + sensor
            mne.viz.plot_sensors(
                info,
                axes=ax,
                show=False,
                kind="topomap"
            )

            pair_dict = map_group.get(band, {})
            n_lines = 0

            if not isinstance(pair_dict, dict) or len(pair_dict) == 0:
                ax.text(
                    0.5,
                    0.5,
                    f"{band}\nkolom ORI_CONN_PLI tidak tersedia",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="gray"
                )
                continue

            for pair, val in pair_dict.items():
                if not np.isfinite(val):
                    continue

                # PLI normalnya 0..1
                pli_val = float(val)

                if pli_val < threshold:
                    continue

                # pair format: "Fp1-Fp2"
                if "-" not in pair:
                    continue

                ch1, ch2 = pair.split("-", 1)
                ch1 = normalize_ch_name(ch1)
                ch2 = normalize_ch_name(ch2)

                if ch1 not in ch_to_idx or ch2 not in ch_to_idx:
                    continue

                i = ch_to_idx[ch1]
                j = ch_to_idx[ch2]

                if i >= len(pos2d) or j >= len(pos2d):
                    continue

                x1, y1 = pos2d[i]
                x2, y2 = pos2d[j]

                line_color = cmap(norm(np.clip(pli_val, 0.0, 1.0)))

                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=line_color,
                    linewidth=0.5 + 2.5 * pli_val,
                    alpha=0.85,
                    solid_capstyle="round"
                )

                n_lines += 1

            ax.text(
                0.5,
                -0.08,
                f"PLI ≥ {threshold:.2f} | lines={n_lines}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8,
                color="gray"
            )

        # Matikan axes kosong
        for b_idx in range(n_bands, rows_a * cols_a):
            axs[b_idx // cols_a][b_idx % cols_a].axis("off")

        # Colorbar global PLI
        cax = fig.add_axes([0.35, 0.07, 0.30, 0.018])

        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=norm
        )
        sm.set_array([])

        cb = fig.colorbar(
            sm,
            cax=cax,
            orientation="horizontal"
        )
        cb.set_label("PLI", fontsize=8)
        cb.ax.tick_params(labelsize=8)

        fig.subplots_adjust(
            left=0.04,
            right=0.98,
            bottom=0.14,
            top=0.88,
            wspace=0.25,
            hspace=0.35
        )

        self._avg_feat_embed_canvas(fig, parent)
        
    def _avg_feat_plot_map_group(
        self,
        map_group,
        parent,
        title,
        diverging=False,
        vlim_map=None,
        cmap="jet"
    ):
        """
        Plot beberapa topomap dalam satu figure.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import mne

        info = self._avg_feat_get_mne_info()

        names = list(map_group.keys())
        n = len(names)

        n_cols = min(3, n)
        n_rows = int(np.ceil(n / n_cols))

        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.2 * n_cols, 3.8 * n_rows),
            dpi=90,
            squeeze=False
        )

        fig.patch.set_facecolor("white")
        fig.suptitle(title, fontsize=13, fontweight="bold")

        for i, name in enumerate(names):
            ax = axs[i // n_cols][i % n_cols]
            vals = np.asarray(map_group[name], dtype=float)

            valid = np.isfinite(vals)
            if not np.any(valid):
                ax.axis("off")
                ax.text(
                    0.5, 0.5,
                    f"{name}\nkolom tidak tersedia",
                    ha="center",
                    va="center",
                    fontsize=9
                )
                continue

            fill_val = np.nanmean(vals[valid])
            vals_plot = np.where(valid, vals, fill_val)

            # ============================================================
            # Skala warna mengikuti Brain Mapping
            # ============================================================
            if isinstance(vlim_map, dict) and name in vlim_map:
                vmin, vmax = vlim_map[name]
            else:
                vlims = self._avg_feat_compute_vlims_like_brainmap(all_maps)
            try:
                im, _ = mne.viz.plot_topomap(
                    vals_plot,
                    info,
                    axes=ax,
                    show=False,
                    cmap=cmap,
                    contours=4,
                    sensors=True,
                    extrapolate="head",
                    vlim=(vmin, vmax)
                )
            except TypeError:
                # Fallback untuk versi MNE lama
                im, _ = mne.viz.plot_topomap(
                    vals_plot,
                    info,
                    axes=ax,
                    show=False,
                    cmap=cmap,
                    contours=4,
                    sensors=True,
                    extrapolate="head",
                    vmin=vmin,
                    vmax=vmax
                )

            ax.set_title(f"{name}\n[{vmin:.3g}, {vmax:.3g}]", fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.72)

        # Matikan axes kosong
        for j in range(n, n_rows * n_cols):
            axs[j // n_cols][j % n_cols].axis("off")

        # ============================================================
        # Perbaikan layout Avg Feature Topomap:
        # beri jarak vertikal lebih besar agar judul Beta tidak menimpa
        # topomap baris atas.
        # ============================================================
        fig.subplots_adjust(
            left=0.06,
            right=0.96,
            bottom=0.06,
            top=0.86,
            wspace=0.45,
            hspace=0.90
        )

        self._avg_feat_embed_canvas(fig, parent)


    def _avg_feat_run_topomap(self, disease):
        """
        Tombol utama:
        Topomap Stress / Anxiety / Depression.
        """
        from tkinter import ttk, messagebox

        try:
            mean_row, info = self._avg_feat_get_mean_row(disease)
            all_maps = self._avg_feat_make_all_maps(mean_row)
            # ============================================================
            # Hitung skala warna dengan logika yang sama seperti Brain Mapping
            # ============================================================
            vlims = self._avg_feat_compute_vlims_like_brainmap(all_maps)
            self.avg_feat_brainmap_vlims = vlims

            # Bersihkan tab result
            for tab_id in self.avg_feat_result_notebook.tabs():
                self.avg_feat_result_notebook.forget(tab_id)

            header = (
                f"{info['group_name']} | Rule: {info['rule']} | "
                f"Segments: {info['n_segment']} | Subjects: {info['n_subject']} | "
                f"Average: {info['avg_mode']}"
            )

            self.avg_feat_status_var.set(header)
            print("[AVG FEATURE TOPO]", header)

            # Buat sub-tab hasil
            for group_name, map_group in all_maps.items():
                frame = ttk.Frame(self.avg_feat_result_notebook)
                self.avg_feat_result_notebook.add(frame, text=group_name)

                # ========================================================
                # PILIH SKALA WARNA SESUAI JENIS FITUR
                # ========================================================
                if group_name == "Abs Power":
                    vlim_map = vlims.get("abs_power", {})
                    diverging = False
                    cmap = "jet"

                elif group_name == "Relative Power":
                    vlim_map = vlims.get("rel_power", {})
                    diverging = False
                    cmap = "jet"

                elif group_name == "Ratio Power":
                    vlim_map = vlims.get("ratios", {})
                    diverging = False
                    cmap = "jet"

                elif group_name == "Asymmetry Index":
                    vlim_map = vlims.get("asym_by_band", {})
                    diverging = True
                    cmap = "jet"

                elif group_name == "SASI":
                    vlim_map = {"SASI": vlims.get("sasi", (-1.0, 1.0))}
                    diverging = True
                    cmap = "jet"

                elif group_name == "LZC Topomap":
                    vlim_map = vlims.get("lzc_maps", {})
                    diverging = False
                    cmap = "jet"
                elif group_name == "PLI":
                    # PLI tidak memakai topomap color scale,
                    # karena akan diplot sebagai connectivity map.
                    vlim_map = {}
                    diverging = False
                    cmap = "jet"
                else:
                    vlim_map = {}
                    diverging = False
                    cmap = "jet"
                    #z-scored_PLI
                # if group_name == "Asymmetry Index":
                #     self._avg_feat_plot_zscore_connectivity_group(
                #         map_group=map_group,
                #         parent=frame,
                #         title=f"Asymmetry Index Z-Score - {info['group_name']}",
                #         metric_kind="asymmetry_ai"
                #     )

                # elif group_name == "PLI":
                #     self._avg_feat_plot_zscore_connectivity_group(
                #         map_group=map_group,
                #         parent=frame,
                #         title=f"PLI Z-score ({self.avg_pli_norm_metric} norm) - {info['group_name']}",
                #         metric_kind=self.avg_pli_norm_metric
                #     )
                if group_name == "Asymmetry Index":
                    self._avg_feat_plot_asym_connectivity_group(
                        map_group=map_group,
                        parent=frame,
                        title=f"Asymmetry Index Connectivity - {info['group_name']}",
                        threshold=self.avg_asym_conn_threshold
                    )

                elif group_name == "PLI":
                    self._avg_feat_plot_pli_connectivity_group(
                        map_group=map_group,
                        parent=frame,
                        title=f"PLI Connectivity - {info['group_name']}",
                        threshold=self.avg_pli_conn_threshold
                    )
                else:
                    self._avg_feat_plot_map_group(
                        map_group=map_group,
                        parent=frame,
                        title=f"{group_name} - {info['group_name']}",
                        diverging=diverging,
                        vlim_map=vlim_map,
                        cmap=cmap
                    )
        except Exception as e:
            messagebox.showerror("Gagal Membuat Average Topomap", str(e))
            self.avg_feat_status_var.set("Gagal membuat average topomap.")
            print("[AVG FEATURE TOPO ERROR]", e)
            
            
    def _build_realtime_plot(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        self.live_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.live_tab, text=" LIVE EEG STREAM ")

        # Frame penampung Tabs Analisis (Lazy initialization)
        self.zap_tab = ttk.Frame(self.notebook)
        self.ica_tab = ttk.Frame(self.notebook)
        self.mif_tab = ttk.Frame(self.notebook)
        self.band_tab = ttk.Frame(self.notebook)
        self.bpm_tab = ttk.Frame(self.notebook)
        self.topo_tab = ttk.Frame(self.notebook)
        # Sub-tab di dalam Brain Mapping (diisi saat show_visual_results dipanggil)
        self.topo_abs_tab  = None
        self.topo_rel_tab  = None
        self.topo_ratio_tab = None
        self.topo_asym_tab = None
        self.topo_sasi_tab = None
        
        self.ica_compare_tab = ttk.Frame(self.notebook)
        self.feat_tab = ttk.Frame(self.notebook)
        self.save_tab = ttk.Frame(self.notebook)   # Saving Feature
        self.signal_save_tab = ttk.Frame(self.notebook)  # Saving Signal

        # ============================================================
        # TAB BARU: Average Topomap from Feature CSV
        # ============================================================
        self.avg_feature_topo_tab = ttk.Frame(self.notebook)

        if not hasattr(self, 'signal_rec_id_var'):
            self.signal_rec_id_var = tk.IntVar(value=1)

        self.btn_save_signal = None  # referensi tombol, dipakai untuk disable setelah save
        # ============================================================
        # Tampilkan tab Avg Feature Topomap sejak awal
        # ============================================================
        self.notebook.add(
            self.avg_feature_topo_tab,
            text=" Avg Feature Topomap "
        )

        self._build_avg_feature_topomap_tab()


        # --- UPDATE WARNA UNTUK MEDICAL CLEAN STYLE ---
        
        # --- UPDATE WARNA UNTUK MEDICAL CLEAN STYLE ---
        bg_color = self.colors["bg"]         # #ffffff (Putih)
        plot_bg = "#ffffff"                  # Putih bersih untuk area grafik
        grid_color = "#f0f0f0"               # Abu-abu sangat muda untuk grid
        
        self.fig, self.axs = plt.subplots(5, 2, figsize=(12, 12))
        self.fig.patch.set_facecolor(bg_color) 
        self.ax_flat = self.axs.flatten()
        
        self.lines = []
        self.labels_imp_text = [] 
        
        for i in range(NUM_CHANNELS):
            ax = self.ax_flat[i]
            # Sinyal EEG menggunakan Biru Royal (#1a73e8)
            line, = ax.plot(self.time_axis, np.zeros(MAX_SAMPLES), 
                            color=self.colors["accent"], lw=1.2)
            self.lines.append(line)
            
            ax.set_ylabel(f"CH {i+1}", color=self.colors["accent"], fontsize=9, fontweight='bold')
            ax.set_facecolor(plot_bg)
            
            # Warna angka sumbu menggunakan abu-abu gelap
            ax.tick_params(colors=self.colors["text_dim"], labelsize=8)
            
            # Grid yang halus/clean
            ax.grid(True, color=grid_color, linestyle='-', linewidth=0.5)
            
            # Hilangkan border atas dan kanan agar terlihat modern
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_edgecolor('#dddddd')
            ax.spines['bottom'].set_edgecolor('#dddddd')
            
            # Impedance Label Styling (Putih dengan border biru)
            lbl = ax.text(0.98, 0.90, "", transform=ax.transAxes, 
                        fontsize=9, color=self.colors["accent"], fontweight='bold',
                        ha='right', va='top',
                        bbox=dict(facecolor='white', alpha=0.9, 
                                 edgecolor=self.colors["accent"], boxstyle='round,pad=0.2'))
            self.labels_imp_text.append(lbl)

        # Digital Plot (Warna kontras tapi tetap di bg putih)
        # --- Subplot index 8: DIGITAL ---
        self.dig_ax = self.ax_flat[8]
        self.dig_line, = self.dig_ax.plot(self.time_axis, np.zeros(MAX_SAMPLES), color="#e91e63", lw=1.5)
        self.dig_ax.set_ylabel("DIGITAL", fontsize=9, color='#e91e63', fontweight='bold')
        self.dig_ax.set_facecolor(plot_bg)
        self.dig_ax.set_ylim(-0.2, 1.2)
        self.dig_ax.tick_params(colors=self.colors["text_dim"], labelsize=8)
        self.dig_ax.grid(True, color=grid_color, linestyle='-', linewidth=0.5)
        self.dig_ax.spines['top'].set_visible(False)
        self.dig_ax.spines['right'].set_visible(False)

        # --- Subplot index 9: PPG (tersembunyi sampai analysis selesai) ---
        self.ppg_ax = self.ax_flat[9]
        self.ppg_line, = self.ppg_ax.plot([], [], color="#ff9800", lw=1.2)
        self.ppg_ax.set_ylabel("PPG", fontsize=9, color="#ff9800", fontweight='bold')
        self.ppg_ax.set_facecolor(plot_bg)
        self.ppg_ax.tick_params(colors=self.colors["text_dim"], labelsize=8)
        self.ppg_ax.grid(True, color=grid_color, linestyle='-', linewidth=0.5)
        self.ppg_ax.spines['top'].set_visible(False)
        self.ppg_ax.spines['right'].set_visible(False)
        self.ppg_ax.set_visible(False)   # sembunyikan dulu
        # self.ax_flat[7].axis('off')

        plt.tight_layout()

        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

        # Canvas live EEG
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.live_tab)

        # Toolbar zoom/pan untuk live EEG
        self.live_toolbar_frame = ttk.Frame(self.live_tab)
        self.live_toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.live_toolbar = NavigationToolbar2Tk(self.canvas, self.live_toolbar_frame)
        self.live_toolbar.update()

        # Canvas diletakkan di atas toolbar
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def display_last_5s(self):
        """Tampilkan SELURUH rekaman EEG + PPG ke live plot setelah analysis selesai."""
        if self.session_data is None or len(self.session_data) == 0:
            return

        # --- Normalisasi session_data ---
        if isinstance(self.session_data, list):
            raw = np.concatenate(self.session_data, axis=1)
            eeg_ch = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)[:NUM_CHANNELS]
            eeg_data = raw[eeg_ch, :]
        else:
            eeg_data = self.session_data[:NUM_CHANNELS, :]

        n_samples = eeg_data.shape[1]
        t_plot = np.linspace(0, n_samples / FS, n_samples)
        t_max  = t_plot[-1]

        # --- Plot seluruh EEG ---
        for i in range(NUM_CHANNELS):
            sig = eeg_data[i]
            self.lines[i].set_data(t_plot, sig)

            ax = self.ax_flat[i]
            ax.set_xlim(0, t_max)                        # ← kunci: paksa batas X
            y_margin = max(np.ptp(sig) * 0.1, 1.0)
            ax.set_ylim(sig.min() - y_margin, sig.max() + y_margin)
            ax.set_xlabel("Time (s)", fontsize=7)

        # --- Plot Digital ---
        # (biarkan dig_line apa adanya, hanya sesuaikan xlim)
        self.dig_ax.set_xlim(0, t_max)

        # --- Plot PPG jika tersedia ---
        if hasattr(self, 'analysis_results') and self.analysis_results:
            bpm_data = self.analysis_results.get('bpm_data', {})
            # KODE BARU — tambah scatter untuk midpoint peaks:
            if 'ppg_signal' in bpm_data and 'ppg_time' in bpm_data:
                ppg_t   = np.array(bpm_data['ppg_time'])
                ppg_sig = np.array(bpm_data['ppg_signal'])

                self.ppg_ax.set_visible(True)
                self.ppg_ax.cla()  # bersihkan dulu agar scatter tidak numpuk

                self.ppg_ax.plot(ppg_t, ppg_sig, color="#ff9800", lw=1.2, label="PPG")

                # Tampilkan midpoint peaks jika tersedia
                ppg_peaks = bpm_data.get('ppg_peaks', np.array([]))
                if len(ppg_peaks) > 0:
                    self.ppg_ax.scatter(
                        ppg_t[ppg_peaks], ppg_sig[ppg_peaks],
                        color='blue', s=40, zorder=5, label='Peaks'
                    )

                self.ppg_ax.set_xlim(ppg_t[0], ppg_t[-1])
                y_ppg = max(np.ptp(ppg_sig) * 0.1, 0.1)
                self.ppg_ax.set_ylim(ppg_sig.min() - y_ppg, ppg_sig.max() + y_ppg)
                self.ppg_ax.set_xlabel("Time (s)", fontsize=7)
                self.ppg_ax.set_ylabel("PPG", fontsize=9, color="#ff9800", fontweight='bold')
                self.ppg_ax.grid(True, color='#eeeeee', linestyle='-', linewidth=0.5)

        self.fig.tight_layout()
        self.canvas.draw_idle()       # draw_idle lebih aman dari draw() di konteks after()
        self.notebook.select(0)
        
         
    def prepare_session(self):
        """Connect ke board di background thread agar GUI tidak freeze"""
        # Nonaktifkan tombol connect selama proses
        for widget in self.root.winfo_children():
            pass  # tidak perlu disable semua
        self.status_label.config(text="Status: Connecting...", foreground="orange")
        self.root.update_idletasks()

        def _do_connect():
            try:
                params = BrainFlowInputParams()
                params.serial_port = self.com_entry.get()
                board = BoardShim(BoardIds.CYTON_BOARD.value, params)
                board.prepare_session()

                board.config_board("v")

                for ch in range(1, NUM_CHANNELS + 1):
                    cmd = f"x{ch}060101X"
                    board.config_board(cmd)
                    time.sleep(0.2)

                board.config_board("c")
                time.sleep(0.2)
                board.config_board("/3")
                time.sleep(1.0)

                # Simpan board ke self di main thread
                self.board = board
                self.root.after(0, self._on_connect_success)

            except Exception as e:
                err_msg = str(e)   # ← simpan dulu sebelum 'e' hilang
                self.root.after(0, lambda: self._on_connect_error(err_msg))

        threading.Thread(target=_do_connect, daemon=True).start()

    def _on_connect_success(self):
        self.board_id =  BoardIds.CYTON_BOARD.value  
        self.status_label.config(text="Status: Connected (Ready)", foreground="green")
        self.start_monitoring_stream()

    def _on_connect_error(self, msg):
        """Dipanggil di main thread jika connect gagal"""
        self.board = None
        messagebox.showerror("Error Connection", msg)
        self.status_label.config(text="Status: Connection Failed", foreground="red")
        
    # def _build_realtime_plot(self):
    #         self.notebook = ttk.Notebook(self.root)
    #         self.notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
            
    #         self.live_tab = ttk.Frame(self.notebook)
    #         self.notebook.add(self.live_tab, text=" LIVE EEG STREAM ")

    #         # Frame penampung Tabs Analisis (Lazy initialization)
    #         self.zap_tab = ttk.Frame(self.notebook)
    #         self.ica_tab = ttk.Frame(self.notebook)
    #         self.mif_tab = ttk.Frame(self.notebook)
    #         self.band_tab = ttk.Frame(self.notebook)
    #         self.bpm_tab = ttk.Frame(self.notebook)
    #         self.topo_tab = ttk.Frame(self.notebook)
    #         self.feat_tab = ttk.Frame(self.notebook)

    #         # Warna plot background disamakan dengan UI background
    #         bg_color, plot_bg = self.colors["bg"], "#ffffff"
            
    #         self.fig, self.axs = plt.subplots(4, 2, figsize=(12, 10))
    #         self.fig.patch.set_facecolor(bg_color) # Ini kuncinya agar menyatu
    #         self.ax_flat = self.axs.flatten()
            
    #         self.lines = []
    #         self.labels_imp_text = [] 
            
    #         for i in range(NUM_CHANNELS):
    #             ax = self.ax_flat[i]
    #             line, = ax.plot(self.time_axis, np.zeros(MAX_SAMPLES), color=self.colors["green"], lw=1)
    #             self.lines.append(line)
                
    #             ax.set_ylabel(f"CH {i+1}", color=self.colors["accent"], fontsize=9, fontweight='bold')
    #             ax.set_facecolor(plot_bg)
    #             ax.tick_params(colors='#5f6368', labelsize=8)
    #             ax.grid(True, color='#f0f0f0', linestyle='--', alpha=0.5)
                
    #             # Impedance Label Styling
    #             lbl = ax.text(0.98, 0.90, "", transform=ax.transAxes, 
    #                         fontsize=9, color='white', fontweight='bold',
    #                         ha='right', va='top',
    #                         bbox=dict(facecolor='#252526', alpha=0.8, edgecolor=self.colors["accent"], boxstyle='round,pad=0.2'))
    #             self.labels_imp_text.append(lbl)

    #         # Digital Plot
    #         self.dig_ax = self.ax_flat[6]
    #         self.dig_line, = self.dig_ax.plot(self.time_axis, np.zeros(MAX_SAMPLES), color="#ff00ff", lw=1.5)
    #         self.dig_ax.set_ylabel("DIGITAL", fontsize=9, color='#ff00ff', fontweight='bold')
    #         self.dig_ax.set_facecolor(plot_bg)
    #         self.dig_ax.set_ylim(-0.2, 1.2)
    #         self.dig_ax.tick_params(colors='#444444', labelsize=8)

    #         self.ax_flat[7].axis('off')

    #         plt.tight_layout()
    #         self.canvas = FigureCanvasTkAgg(self.fig, master=self.live_tab)
    #         self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        
    def start_monitoring_stream(self):
        """Memulai stream untuk monitoring (belum merekam ke file)"""
        if self.board and not self.streaming:
            self.board.start_stream()
            self.streaming = True
            self.is_recording = False
            
        # SESUDAH
        if hasattr(self, 'btn_record'):
            self.btn_record.config(state=tk.NORMAL)
        if hasattr(self, 'btn_stop'):
            self.btn_stop.config(state=tk.NORMAL)
        if hasattr(self, 'status_label'):
            self.status_label.config(text="Status: Monitoring (Not Recording)", foreground="orange")
            self.update_live_plot()

    # ==============================
    # LOGIKA IMPEDANSI INDIVIDU
    # ==============================
    def toggle_channel_impedance(self, ch_idx):
        if not self.board:
            messagebox.showwarning("Warning", "Connect to board first.")
            return

        # Gunakan thread agar GUI tidak membeku saat menunggu time.sleep()
        def run_toggle():
            current_state = self.imp_states[ch_idx]
            new_state = not current_state
            self.imp_states[ch_idx] = new_state
            ch_char = str(ch_idx + 1) 

            if new_state:
                self.imp_buttons[ch_idx].config(bg="yellow")
                # Cek Impedansi: Gain 1x (0), Bias On (1), SRB2 On (1)
                cmd_config = f"x{ch_char}000101X"
                cmd_inject = f"z{ch_char}10Z"
                self._send_impedance_commands(cmd_config, cmd_inject)
            else:
                self.imp_buttons[ch_idx].config(bg="SystemButtonFace")
                # Mode Normal: Gain 24x (6), Bias On (1), SRB2 On (1)
                cmd_inject_off = f"z{ch_char}00Z"
                cmd_restore = f"x{ch_char}060101X"
                self._send_impedance_commands(cmd_inject_off, cmd_restore)
                
            # Bersihkan buffer channel ini agar data impedansi tidak mencemari EEG normal
            self.data_buffer[ch_idx].fill(0)
            self.last_known_imp[ch_idx] = None
            self.last_impedance_time[ch_idx] = 0  # Reset timer impedansi juga
            if hasattr(self, 'labels_imp_text') and ch_idx < len(self.labels_imp_text):
                # Hapus label impedansi dari plot agar tidak misleading
                self.root.after(0, lambda idx=ch_idx: self.labels_imp_text[idx].set_text(""))
                 
        threading.Thread(target=run_toggle, daemon=True).start()
                
    # Fungsi Helper untuk kirim command dengan jeda aman
    def _send_impedance_commands(self, cmd1, cmd2):
        if self.board:
            try:
                self.is_configuring = True
                
                # 1. Matikan stream jika aktif
                if self.streaming:
                    print("Stopping stream for hardware configuration...")
                    self.board.stop_stream()
                    self.streaming = False
                    time.sleep(0.5) # Jeda agar paket terakhir selesai terkirim

                # 2. AGGRESSIVE FLUSH (Kunci utama)
                # Ambil semua data sisa dari buffer internal BrainFlow
                for _ in range(3):
                    self.board.get_board_data()
                    time.sleep(0.1)

                # 3. Kirim perintah dengan Try-Except untuk UTF-8
                for cmd in [cmd1, cmd2]:
                    try:
                        print(f"Sending Command: {cmd}")
                        # Beberapa versi firmware butuh awalan 's' untuk sinkronisasi teks
                        # Namun x dan z standar biasanya langsung bisa
                        response = self.board.config_board(cmd)
                        print(f"Board Response: {response}")
                        time.sleep(0.3) # Jeda antar perintah agar firmware tidak hang
                    except Exception as e:
                        if "utf-8" in str(e).lower():
                            print(f"Terjadi sisa byte biner, tapi perintah {cmd} kemungkinan berhasil.")
                        else:
                            raise e

                # 4. Nyalakan kembali stream
                # 4. Nyalakan kembali stream
                print("Restarting stream...")
                self.board.start_stream()
                self.streaming = True

                # Flush agresif: tunggu 1 detik lalu buang semua data yang masuk
                # selama periode transisi gain (dari mode impedansi ke mode EEG normal)
                time.sleep(1.0)
                for _ in range(5):
                    flushed = self.board.get_board_data()
                    print(f"  Flush: {flushed.shape[1]} samples discarded")
                    time.sleep(0.1)

                print("Stream stable, ready.")
                
            except Exception as e:
                print(f"Error Konfigurasi: {e}")
                messagebox.showerror("Hardware Error", f"Gagal konfigurasi board: {e}")
            finally:
                self.is_configuring = False      
            
             
    def reset_all_impedance(self):
        """Mematikan semua impedansi (dipanggil saat mau rekam)"""
        for i in range(NUM_CHANNELS):
            if self.imp_states[i]:
                self.toggle_channel_impedance(i) # Turn OFF

    # ==============================
    # RECORDING
    # ==============================
    def start_recording(self):
        """Mulai merekam dan aktifkan aliran data jika belum jalan"""

        # Jika board sudah di-release (setelah stop_and_save), reconnect dulu
        if not self.board:
            try:
                self.status_label.config(text="Status: Reconnecting...", foreground="orange")
                self.root.update_idletasks()

                params = BrainFlowInputParams()
                params.serial_port = self.com_entry.get()
                self.board = BoardShim(BoardIds.CYTON_BOARD.value, params)
                self.board.prepare_session()
                self.board.config_board("v")
                for ch in range(1, NUM_CHANNELS + 1):
                    self.board.config_board(f"x{ch}060101X")
                    time.sleep(0.1)
                self.board.config_board("c")
                time.sleep(0.2)
                self.board.config_board("/3")
                time.sleep(0.5)
                self.board.start_stream()
                self.streaming = True
                
                self.update_live_plot()
            except Exception as e:
                err_msg = str(e)
                messagebox.showerror("Reconnect Error", f"Gagal reconnect ke board:\n{err_msg}")
                self.status_label.config(text="Status: Reconnect Failed", foreground="red")
                self.board = None
                return

        # Jalankan stream board jika masih idle
        if not self.streaming:
            self.board.start_stream()
            self.streaming = True
            self.update_live_plot()

        # ✅ BAGIAN INI YANG HILANG — set state dan mulai countdown
        self.is_recording = True
        self.session_data = []
        self.data_buffer.fill(0)

        self.reset_all_impedance()

        for i in range(NUM_CHANNELS):
            self.ax_flat[i].set_ylim(-100, 100)

        self.status_label.config(text="Status: RECORDING DATA...", foreground="red")
        self.btn_record.config(state=tk.DISABLED, text="▶ Recording...")
        self.record_start_time = time.time()
        self._start_countdown(self.duration_var.get())
        
        
    def _start_countdown(self, total_seconds):
        """Mulai countdown dari total_seconds."""
        if self._countdown_job:
            self.root.after_cancel(self._countdown_job)
        self._tick_countdown(total_seconds)


    def _tick_countdown(self, total_seconds):
        if not self.is_recording:
            self.countdown_label.config(text="")
            return
        
        # Hitung sisa waktu dari waktu nyata, bukan dari counter
        elapsed = time.time() - self.record_start_time
        remaining = total_seconds - elapsed
        
        if remaining <= 0:
            self.countdown_label.config(text="00:00")
            self.stop_and_save()
            return
        
        mins, secs = divmod(int(remaining), 60)
        self.countdown_label.config(text=f"{mins:02d}:{secs:02d}")
        # Jadwalkan ulang setiap ~100ms agar lebih responsif dan akurat
        self._countdown_job = self.root.after(100, self._tick_countdown, total_seconds)  
    def stop_and_save(self):
            # 1. Pastikan kita sedang merekam
            if not self.is_recording:
                messagebox.showinfo("Info", "Tidak ada proses rekaman yang berjalan.")
                return

# SESUDAH (aman):
            if getattr(self, '_countdown_job', None):
                self.root.after_cancel(self._countdown_job)
                self._countdown_job = None
            if hasattr(self, 'countdown_label'):
                self.countdown_label.config(text="")
            
            self.is_recording = False
            # time.sleep(0.2)
            
            # 2. Ambil data terakhir dari board jika masih ada
            if self.board:
                try:
                    # Ambil sisa data yang belum terbaca
                    remaining_data = self.board.get_board_data()
                    if remaining_data.shape[1] > 0:
                        self.session_data.append(remaining_data)
                    
                    # Stop stream dan release agar port COM bersih
                    self.board.stop_stream()
                    self.board.release_session()
                    self.streaming = False
                    print("Stream stopped and session released.")
                except Exception as e:
                    print(f"Error closing board: {e}")
                
                self.board = None # Reset object board

            # 3. Proses Penggabungan dan Penyimpanan
            if self.session_data:
                try:
                    full_data = np.concatenate(self.session_data, axis=1)
                    custom_name = self.filename_var.get().strip()
                    if not custom_name:
                        custom_name = "EEG_Recording"
                    # Bersihkan karakter ilegal dari nama file
                    import re
                    custom_name = re.sub(r'[\\/*?:"<>|]', "_", custom_name)
                    self.last_filename = f"{custom_name}_{int(time.time())}.txt"

                    eeg_ch = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)[:NUM_CHANNELS]
                    DIGITAL_CHANNEL = 13
                    save_data = full_data[eeg_ch + [DIGITAL_CHANNEL], :]
                    
                    # Simpan ke TXT
                    with open(self.last_filename, 'w') as f:
                        f.write("% OpenBCI EEG Recording\n")
                        f.write(f"% FS: {FS} Hz\n")
                        # ✅ Header juga diawali % agar ikut ter-skip oleh comment='%'
                        header = "% " + "\t".join([f"EXG Channel {i}" for i in range(NUM_CHANNELS)]) + "\tDigital\n"
                        f.write(header)
                        np.savetxt(f, save_data.T, delimiter='\t', fmt='%.4f')
                    # TAMPILKAN STATUS & AKTIFKAN TOMBOL ANALISIS
                    self.status_label.config(text=f"Saved: {self.last_filename}", foreground="green")
                    self.btn_analyze.config(state=tk.NORMAL) # AKTIFKAN DI SINI
                    self.btn_record.config(state=tk.NORMAL, text="Start Recording") # Siap rekam lagi
                    
                    messagebox.showinfo("Success", f"Data berhasil disimpan ke {self.last_filename}\nSekarang Anda bisa menjalankan Analysis.")

                except Exception as e:
                    messagebox.showerror("Save Error", f"Gagal menyimpan data: {e}")
            else:
                messagebox.showwarning("Warning", "Rekaman terlalu pendek, tidak ada data untuk disimpan.")
                self.status_label.config(text="Status: No Data Recorded", foreground="red")
                self.btn_record.config(state=tk.NORMAL, text="Start Recording")
                
    def compute_sasi(self, psd, freqs, fc):
        f1 = fc - 6
        f2 = fc - 2
        f3 = fc + 2
        f4 = min(fc + 26, 30)  # limit 30 Hz

        low_mask  = (freqs >= f1) & (freqs <= f2)
        high_mask = (freqs >= f3) & (freqs <= f4)

        Wl = np.sum(psd[low_mask])
        Wh = np.sum(psd[high_mask])

        if (Wh + Wl) == 0:
            return 0

        return (Wh - Wl) / (Wh + Wl)

# =========================================================
# HELPER: Embed topomap + tabel angka dalam satu frame
# =========================================================
    def _plot_topo_with_table(self, parent, fig, data_dict, band_names, ch_names):
        """
        Tampilkan figure topomap di atas dan tabel angka di bawah.
        data_dict: {band_name: list_nilai_per_channel}
        """
        # Frame atas: topomap
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.BOTH, expand=True)
        self._embed_canvas(fig, top_frame)

        # Frame bawah: tabel angka (scrollable)
        bot_frame = ttk.Frame(parent)
        bot_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        cols = ["Channel"] + band_names
        tree = ttk.Treeview(bot_frame, columns=cols, show='headings', height=len(ch_names))
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=80, anchor='center')

        for ch_i, ch in enumerate(ch_names):
            row_vals = [ch] + [f"{data_dict[b][ch_i]:.4f}" for b in band_names]
            tree.insert("", "end", values=row_vals)

        sb = ttk.Scrollbar(bot_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="x", expand=True)
        sb.pack(side="right", fill="y")

    def plot_brain_mapping(self, data, parent):
        if not self.analysis_results:
            return

        for widget in parent.winfo_children():
            widget.destroy()

        # ============================================================
        # DATA DASAR
        # ============================================================
        abs_power = data["abs_power"]
        rel_power = data["rel_power"]
        band_names = data["names"]

        info = self.analysis_results["topo_info"]
        ch_names = info["ch_names"]
        n_bands = len(band_names)
        n_ch_topo = len(ch_names)

        def local_vlim(values, symmetric=False):
            """
            Membuat vmin-vmax lokal untuk satu topomap.
            symmetric=True dipakai untuk nilai negatif-positif,
            misalnya asymmetry dan SASI.
            """
            vals = np.asarray(values, dtype=float).reshape(-1)
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                return 0.0, 1.0

            if symmetric:
                m = np.max(np.abs(vals))
                if m < 1e-12:
                    m = 1.0
                return -m, m

            vmin = float(np.min(vals))
            vmax = float(np.max(vals))

            if np.isclose(vmin, vmax):
                pad = abs(vmin) * 0.05 if abs(vmin) > 1e-12 else 1.0
                vmin -= pad
                vmax += pad

            return vmin, vmax

        def as_topo_values(values, fill_value=0.0):
            """
            Pastikan jumlah nilai sama dengan jumlah channel di topo_info.
            Ini penting agar plot_topomap tidak error.
            """
            arr = np.asarray(values, dtype=float).reshape(-1)
            arr = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)

            if len(arr) == n_ch_topo:
                return arr

            if len(arr) > n_ch_topo:
                return arr[:n_ch_topo]

            pad = np.full(n_ch_topo - len(arr), fill_value, dtype=float)
            return np.concatenate([arr, pad])

        def get_eeg_for_lzc():
            """
            Ambil sinyal EEG bersih untuk perhitungan LZC.
            Prioritas:
            1. results['ica']['data']
            2. results['ica_clean']
            3. self.session_data
            """
            # 1) Dari hasil ICA utama
            try:
                if "ica" in self.analysis_results:
                    ica_obj = self.analysis_results["ica"]

                    if isinstance(ica_obj, dict):
                        for key in ["data", "clean", "ica_clean"]:
                            if key in ica_obj and ica_obj[key] is not None:
                                return np.asarray(ica_obj[key], dtype=float)
            except Exception:
                pass

            # 2) Dari key lain
            for key in ["ica_clean", "clean_eeg", "eeg_clean"]:
                try:
                    if key in self.analysis_results and self.analysis_results[key] is not None:
                        return np.asarray(self.analysis_results[key], dtype=float)
                except Exception:
                    pass

            # 3) Dari session_data
            if hasattr(self, "session_data") and self.session_data is not None:
                if isinstance(self.session_data, list):
                    raw = np.concatenate(self.session_data, axis=1)

                    try:
                        eeg_ch = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
                        eeg_ch = eeg_ch[:min(len(eeg_ch), n_ch_topo)]

                        if max(eeg_ch) < raw.shape[0]:
                            return raw[eeg_ch, :]
                    except Exception:
                        pass

                    return raw[:n_ch_topo, :]

                raw = np.asarray(self.session_data, dtype=float)

                if raw.ndim == 2:
                    if raw.shape[0] >= n_ch_topo:
                        return raw[:n_ch_topo, :]

                    if raw.shape[1] >= n_ch_topo and raw.shape[0] > raw.shape[1]:
                        return raw[:, :n_ch_topo].T

                    return raw

            return None

        # ============================================================
        # DEFINISI RASIO DAN ASIMETRI
        # ============================================================
        ratio_defs = [
            ("Delta/Theta", "Delta", "Theta"),
            ("Delta/Alpha", "Delta", "Alpha"),
            ("Delta/Beta",  "Delta", "Beta"),
            ("Theta/Alpha", "Theta", "Alpha"),
            ("Theta/Beta",  "Theta", "Beta"),
            ("Alpha/Beta",  "Alpha", "Beta"),
        ]

        # Pair asimetri dibuat dinamis agar aman untuk 4CH atau 8CH
        candidate_asym_pairs = [
            ("Fp1-Fp2", "Fp1", "Fp2"),
            ("F3-F4",   "F3",  "F4"),
            ("T3-T4",   "T3",  "T4"),
            ("T7-T8",   "T7",  "T8"),
            ("P3-P4",   "P3",  "P4"),
        ]

        ch_index_name = {ch: idx for idx, ch in enumerate(ch_names)}

        asym_pairs = []
        for pair_label, left_ch, right_ch in candidate_asym_pairs:
            if left_ch in ch_index_name and right_ch in ch_index_name:
                asym_pairs.append(
                    (pair_label, ch_index_name[left_ch], ch_index_name[right_ch])
                )

        ch_to_pair = {}
        for pair_idx, (_, li, ri) in enumerate(asym_pairs):
            ch_to_pair[li] = pair_idx
            ch_to_pair[ri] = pair_idx

        # ============================================================
        # HITUNG / AMBIL SASI
        # ============================================================
        if "sasi_values" in data:
            sasi_values = as_topo_values(data["sasi_values"])
        else:
            from scipy.signal import welch as sp_welch

            eeg_data = get_eeg_for_lzc()

            if eeg_data is None:
                sasi_values = np.zeros(n_ch_topo)
            else:
                eeg_data = np.asarray(eeg_data, dtype=float)

                if eeg_data.ndim == 2 and eeg_data.shape[0] > eeg_data.shape[1] and eeg_data.shape[1] <= 16:
                    eeg_data = eeg_data.T

                eeg_data = eeg_data[:n_ch_topo, :]

                try:
                    freqs, psd_all = sp_welch(eeg_data, 250, nperseg=500, axis=1)
                    alpha_mask = (freqs >= 8) & (freqs <= 12)

                    if np.any(alpha_mask):
                        fc = freqs[alpha_mask][np.argmax(psd_all[:, alpha_mask].mean(axis=0))]
                        sasi_raw = np.array([
                            self.compute_sasi(psd_all[ch], freqs, fc)
                            for ch in range(psd_all.shape[0])
                        ])
                        sasi_values = as_topo_values(sasi_raw)
                    else:
                        sasi_values = np.zeros(n_ch_topo)

                except Exception as e:
                    print(f"[SASI] Gagal menghitung SASI: {e}")
                    sasi_values = np.zeros(n_ch_topo)

        # ============================================================
        # HITUNG / AMBIL POWER RATIO
        # ============================================================
        if "ratios" in data:
            ratios = {
                k: as_topo_values(v)
                for k, v in data["ratios"].items()
            }
        else:
            ratios = {}

            for label, nb, db in ratio_defs:
                if nb not in abs_power or db not in abs_power:
                    ratios[label] = np.zeros(n_ch_topo)
                    continue

                num = as_topo_values(abs_power[nb])
                den_raw = as_topo_values(abs_power[db])
                den = np.where(den_raw == 0, 1e-10, den_raw)

                ratios[label] = num / den

        # ============================================================
        # HITUNG / AMBIL ASIMETRI PER BAND
        # ============================================================
        if "asym_by_band" in data:
            asym_by_band = {
                k: as_topo_values(v).tolist()
                for k, v in data["asym_by_band"].items()
            }
        else:
            asym_by_band = {}

            for band in band_names:
                asym_topo = np.zeros(n_ch_topo)
                p = as_topo_values(abs_power[band])

                for ch_idx, pair_idx in ch_to_pair.items():
                    _, li, ri = asym_pairs[pair_idx]

                    if li < len(p) and ri < len(p):
                        L = p[li]
                        R = p[ri]
                        denom = L + R
                        asym_topo[ch_idx] = (L - R) / denom if denom != 0 else 0.0

                asym_by_band[band] = asym_topo.tolist()

        # ============================================================
        # HITUNG / AMBIL LZC MAPS
        # ============================================================
        if "lzc_maps" in data and data["lzc_maps"]:
            lzc_maps = {
                k: as_topo_values(v).tolist()
                for k, v in data["lzc_maps"].items()
            }
        else:
            lzc_maps = {}

            try:
                eeg_for_lzc = get_eeg_for_lzc()

                if eeg_for_lzc is not None:
                    lzc_raw = self.compute_lzc_maps_from_eeg(
                        eeg_for_lzc,
                        fs=FS
                    )

                    lzc_maps = {
                        k: as_topo_values(v).tolist()
                        for k, v in lzc_raw.items()
                    }

                    data["lzc_maps"] = lzc_maps

                    print("[LZC TOPO] lzc_maps berhasil dihitung di plot_brain_mapping.")
                    print("[LZC TOPO] keys:", list(lzc_maps.keys()))
                else:
                    print("[LZC TOPO] EEG clean tidak ditemukan, LZC tidak dihitung.")

            except Exception as e:
                print(f"[LZC TOPO] Gagal menghitung LZC maps: {e}")
                lzc_maps = {}

        # ============================================================
        # SUB-NOTEBOOK BRAIN MAP
        # ============================================================
        sub_nb = ttk.Notebook(parent)
        sub_nb.pack(fill=tk.BOTH, expand=True)

        frames = {}

        for label in [
            "Absolute Power",
            "Relative Power",
            "Power Ratio",
            "Asimetri Amplitudo",
            "SASI",
            "LZC Topomap",
            "Pengurangan Topomap"
        ]:
            f = ttk.Frame(sub_nb)
            sub_nb.add(f, text=f" {label} ")
            frames[label] = f

        # ============================================================
        # TAB 1: ABSOLUTE POWER
        # ============================================================
        fig1, axs1 = plt.subplots(
            1,
            n_bands,
            figsize=(3.8 * n_bands, 4),
            squeeze=False
        )
        axs1 = axs1.reshape(-1)

        for i, band in enumerate(band_names):
            vals = as_topo_values(abs_power[band])
            vmin_b, vmax_b = local_vlim(vals)

            im, _ = mne.viz.plot_topomap(
                vals,
                info,
                axes=axs1[i],
                show=False,
                cmap="jet",
                extrapolate="head",
                vlim=(vmin_b, vmax_b),
                sphere=0.1
            )

            axs1[i].set_title(
                f"{band}\n[{vmin_b:.3g}, {vmax_b:.3g}]",
                fontsize=9,
                fontweight="bold"
            )

            fig1.colorbar(im, ax=axs1[i], shrink=0.65, label="µV²/Hz")

        plt.tight_layout()
        self._plot_topo_with_table(
            frames["Absolute Power"],
            fig1,
            abs_power,
            band_names,
            ch_names
        )

        # ============================================================
        # TAB 2: RELATIVE POWER
        # ============================================================
        fig2, axs2 = plt.subplots(
            1,
            n_bands,
            figsize=(3.8 * n_bands, 4),
            squeeze=False
        )
        axs2 = axs2.reshape(-1)

        for i, band in enumerate(band_names):
            vals = as_topo_values(rel_power[band])
            vmin_b, vmax_b = local_vlim(vals)

            im2, _ = mne.viz.plot_topomap(
                vals,
                info,
                axes=axs2[i],
                show=False,
                cmap="jet",
                extrapolate="head",
                vlim=(vmin_b, vmax_b)
            )

            axs2[i].set_title(
                f"{band}\n[{vmin_b:.3g}, {vmax_b:.3g}]",
                fontsize=9,
                fontweight="bold"
            )

            fig2.colorbar(im2, ax=axs2[i], shrink=0.65, label="Relative")

        plt.tight_layout()
        self._plot_topo_with_table(
            frames["Relative Power"],
            fig2,
            rel_power,
            band_names,
            ch_names
        )

        # ============================================================
        # TAB 3: POWER RATIO
        # ============================================================
        n_ratios = len(ratio_defs)
        cols_r = min(n_ratios, 4)
        rows_r = int(np.ceil(n_ratios / cols_r))

        fig3, axs3 = plt.subplots(
            rows_r,
            cols_r,
            figsize=(3.5 * cols_r, 4 * rows_r),
            squeeze=False
        )

        ratio_labels = [lbl for lbl, _, _ in ratio_defs]

        for r_idx, (label, _, _) in enumerate(ratio_defs):
            ax = axs3[r_idx // cols_r][r_idx % cols_r]

            vals = as_topo_values(ratios[label])
            vmin_r, vmax_r = local_vlim(vals)

            im3, _ = mne.viz.plot_topomap(
                vals,
                info,
                axes=ax,
                show=False,
                cmap="jet",
                extrapolate="head",
                vlim=(vmin_r, vmax_r)
            )

            ax.set_title(
                f"{label}\n[{vmin_r:.3g}, {vmax_r:.3g}]",
                fontsize=8,
                fontweight="bold"
            )

            fig3.colorbar(im3, ax=ax, shrink=0.65)

        for r_idx in range(n_ratios, rows_r * cols_r):
            axs3[r_idx // cols_r][r_idx % cols_r].axis("off")

        plt.tight_layout()

        ratio_dict = {
            lbl: as_topo_values(ratios[lbl]).tolist()
            for lbl in ratio_labels
        }

        bot3 = ttk.Frame(frames["Power Ratio"])
        top3 = ttk.Frame(frames["Power Ratio"])

        top3.pack(fill=tk.BOTH, expand=True)
        bot3.pack(fill=tk.X, padx=10, pady=(0, 8))

        self._embed_canvas(fig3, top3)

        cols3 = ["Channel"] + ratio_labels
        tree3 = ttk.Treeview(
            bot3,
            columns=cols3,
            show="headings",
            height=len(ch_names)
        )

        for c in cols3:
            tree3.heading(c, text=c)
            tree3.column(c, width=85, anchor="center")

        for ci, ch in enumerate(ch_names):
            row3 = [ch] + [f"{ratio_dict[lb][ci]:.4f}" for lb in ratio_labels]
            tree3.insert("", "end", values=row3)

        sb3 = ttk.Scrollbar(bot3, orient="vertical", command=tree3.yview)
        tree3.configure(yscrollcommand=sb3.set)

        tree3.pack(side="left", fill="x", expand=True)
        sb3.pack(side="right", fill="y")

        # ============================================================
        # TAB 4: ASIMETRI AMPLITUDO
        # ============================================================
        from itertools import combinations
        from mne.channels.layout import _find_topomap_coords
        from matplotlib.lines import Line2D
        import matplotlib.colors as mcolors

        cols_a = min(n_bands, 4)
        rows_a = int(np.ceil(n_bands / cols_a))

        fig4, axs4 = plt.subplots(
            rows_a,
            cols_a,
            figsize=(3.5 * cols_a, 4 * rows_a),
            squeeze=False
        )

        pos2d = _find_topomap_coords(info, picks=np.arange(len(ch_names)))
        ch_index = {ch: i for i, ch in enumerate(ch_names)}
        all_pairs = list(combinations(ch_names, 2))

        for b_idx, band in enumerate(band_names):
            ax = axs4[b_idx // cols_a][b_idx % cols_a]
            ax.set_title(f"Asym {band}", fontsize=8, fontweight="bold")

            mne.viz.plot_sensors(
                info,
                axes=ax,
                show=False,
                kind="topomap"
            )

            p = as_topo_values(abs_power[band])

            for ch1, ch2 in all_pairs:
                i = ch_index[ch1]
                j = ch_index[ch2]

                denom = p[i] + p[j]
                val = (p[i] - p[j]) / denom if denom != 0 else 0.0
                coh = abs(val)

                if coh > 0.2:
                    x1, y1 = pos2d[i]
                    x2, y2 = pos2d[j]

                    line_color = "red" if val > 0 else "blue"

                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color=line_color,
                        linewidth=0.5 + 1.5 * coh,
                        alpha=0.7,
                        solid_capstyle="round"
                    )

        legend_elements = [
            Line2D([0], [0], color="blue", lw=2, label="(-) Asimetri"),
            Line2D([0], [0], color="red", lw=2, label="(+) Asimetri")
        ]

        fig4.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=2,
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, 0.02)
        )

        ax_bar = fig4.add_axes([0.4, 0.07, 0.2, 0.015])
        cmap_scale = mcolors.ListedColormap(["blue", "red"])
        norm_scale = mcolors.Normalize(vmin=-1, vmax=1)

        cb = fig4.colorbar(
            plt.cm.ScalarMappable(norm=norm_scale, cmap=cmap_scale),
            cax=ax_bar,
            orientation="horizontal",
            ticks=[]
        )
        cb.outline.set_visible(False)

        ax_bar.text(
            -0.2,
            0.5,
            "-",
            transform=ax_bar.transAxes,
            va="center",
            ha="right",
            fontsize=14,
            fontweight="bold",
            color="blue"
        )

        ax_bar.text(
            1.2,
            0.5,
            "+",
            transform=ax_bar.transAxes,
            va="center",
            ha="left",
            fontsize=12,
            fontweight="bold",
            color="red"
        )

        for b_idx in range(n_bands, rows_a * cols_a):
            axs4[b_idx // cols_a][b_idx % cols_a].axis("off")

        plt.tight_layout(rect=[0, 0.1, 1, 1])

        try:
            if not hasattr(self, "_report_fig_cache"):
                self._report_fig_cache = {}
            self._report_fig_cache["Asimetri Amplitudo"] = fig4
        except Exception:
            pass

        self._plot_topo_with_table(
            frames["Asimetri Amplitudo"],
            fig4,
            asym_by_band,
            band_names,
            ch_names
        )

        # ============================================================
        # TAB 5: SASI
        # ============================================================
        fig5, ax5 = plt.subplots(1, 1, figsize=(4, 4))

        im5, _ = mne.viz.plot_topomap(
            sasi_values,
            info,
            axes=ax5,
            cmap="jet",
            vlim=(-1, 1),
            show=False,
            extrapolate="head"
        )

        plt.colorbar(im5, ax=ax5, shrink=0.7, label="SASI")

        ax5.set_title(
            "SASI\n(Spectral Asymmetry Index)",
            fontsize=9,
            fontweight="bold"
        )

        plt.tight_layout()

        top5 = ttk.Frame(frames["SASI"])
        bot5 = ttk.Frame(frames["SASI"])

        top5.pack(fill=tk.BOTH, expand=True)
        bot5.pack(fill=tk.X, padx=10, pady=(0, 8))

        self._embed_canvas(fig5, top5)

        tree5 = ttk.Treeview(
            bot5,
            columns=["Channel", "SASI"],
            show="headings",
            height=len(ch_names)
        )

        for c in ["Channel", "SASI"]:
            tree5.heading(c, text=c)
            tree5.column(c, width=120, anchor="center")

        for ci, ch in enumerate(ch_names):
            tree5.insert("", "end", values=(ch, f"{sasi_values[ci]:.4f}"))

        sb5 = ttk.Scrollbar(bot5, orient="vertical", command=tree5.yview)
        tree5.configure(yscrollcommand=sb5.set)

        tree5.pack(side="left", fill="x", expand=True)
        sb5.pack(side="right", fill="y")

        # ============================================================
        # TAB 6: LZC TOPO MAP
        # ============================================================
        if lzc_maps:
            lzc_names = list(lzc_maps.keys())
            n_lzc = len(lzc_names)

            cols_l = min(n_lzc, 4)
            rows_l = int(np.ceil(n_lzc / cols_l))

            fig6, axs6 = plt.subplots(
                rows_l,
                cols_l,
                figsize=(3.5 * cols_l, 4 * rows_l),
                squeeze=False
            )

            for l_idx, lzc_name in enumerate(lzc_names):
                ax = axs6[l_idx // cols_l][l_idx % cols_l]

                vals = as_topo_values(lzc_maps[lzc_name])
                vmin_l, vmax_l = local_vlim(vals)

                im6, _ = mne.viz.plot_topomap(
                    vals,
                    info,
                    axes=ax,
                    show=False,
                    cmap="viridis",
                    extrapolate="head",
                    vlim=(vmin_l, vmax_l)
                )

                ax.set_title(
                    f"{lzc_name}\n[{vmin_l:.3g}, {vmax_l:.3g}]",
                    fontsize=8,
                    fontweight="bold"
                )

                fig6.colorbar(im6, ax=ax, shrink=0.65, label="LZC")

            for l_idx in range(n_lzc, rows_l * cols_l):
                axs6[l_idx // cols_l][l_idx % cols_l].axis("off")

            plt.tight_layout()

            self._plot_topo_with_table(
                frames["LZC Topomap"],
                fig6,
                lzc_maps,
                lzc_names,
                ch_names
            )

        else:
            ttk.Label(
                frames["LZC Topomap"],
                text=(
                    "LZC Topomap belum tersedia.\n\n"
                    "Pastikan analysis menghasilkan data EEG bersih pada results['ica']['data'] "
                    "atau self.session_data."
                ),
                foreground="gray"
            ).pack(padx=10, pady=10)

        # ============================================================
        # TAB 7: PENGURANGAN TOPO MAP ABSOLUTE POWER
        # ============================================================
        self.build_topomap_subtraction_tab(frames["Pengurangan Topomap"])
        
    # =========================================
    # REVISI FINAL: METODE FFT WINDOWING (MIRIP OPENBCI JAVA)
    # =========================================
    # def _calculate_impedance_ui(self, raw_data):
    #         """
    #         Metode Lock-in Amplifier: Paling robust terhadap noise 50Hz.
    #         Sangat cocok untuk riset medis TEOAE/EEG.
    #         """
    #         if len(raw_data) < FS: return 0.0

    #         # --- KONSTANTA ---
    #         CURRENT_AMP = 6.0e-9 
    #         R_SERIES = 2200.0     
    #         TARGET_FREQ = 31.25 # Frekuensi injeksi ADS1299
            
    #         # 1. Pre-Processing & Scaling
    #         data_uv = raw_data * 24.0
    #         data_ac = data_uv - np.mean(data_uv)

    #         # 2. LAPIS 1: COMB FILTER (NOTCH 50Hz SEMPURNA)
    #         # Karena 250Hz / 50Hz = 5, rata-rata 5 sampel akan menghilangkan 50Hz.
    #         data_comb = np.convolve(data_ac, np.ones(5)/5, mode='same')

    #         # 3. LAPIS 2: LOCK-IN AMPLIFIER
    #         t = np.arange(len(data_comb)) / FS
            
    #         # Referensi Sinus & Kosinus (In-Phase & Quadrature)
    #         ref_sin = np.sin(2 * np.pi * TARGET_FREQ * t)
    #         ref_cos = np.cos(2 * np.pi * TARGET_FREQ * t)
            
    #         # Perkalian (Mixing)
    #         mixed_i = data_comb * ref_sin
    #         mixed_q = data_comb * ref_cos

    #         # 4. LAPIS 3: LPF EKSTREM (Sangat Robust)
    #         # Kita ambil rata-rata dari seluruh window (ini LPF paling tajam)
    #         avg_i = np.mean(mixed_i)
    #         avg_q = np.mean(mixed_q)

    #         # Hitung Amplitudo: A = 2 * sqrt(I^2 + Q^2)
    #         # Faktor 2 berasal dari identitas trigonometri sin^2(x)
    #         v_peak_uv = 2 * np.sqrt(avg_i**2 + avg_q**2)
    #         v_peak_volts = v_peak_uv * 1e-6

    #         # 5. Hitung Impedansi
    #         z_kohm = ((v_peak_volts / CURRENT_AMP) - R_SERIES) / 1000.0

    #         # --- VALIDASI KONTAK (HEURISTIC) ---
    #         # Jika sinyal mentok (Rail), kembalikan nilai tinggi
    #         if np.abs(np.mean(data_uv)) > 1000000.0:
    #             return 99999.0

    #         # Jika elektroda mengambang, noise acak tidak akan fase-locked dengan 31.25Hz.
    #         # Namun tegangan total (STD) akan sangat tinggi.
    #         total_std = np.std(data_comb)
    #         if total_std > 800.0 and z_kohm < 100.0:
    #             return 5800.0 + np.random.uniform(-100, 100)

    #         # Clamping
    #         if z_kohm > 6000.0: return 6000.0 + np.random.uniform(-100, 100)
            
    #         return max(0.0, z_kohm)
    def plot_topomap_subtraction_result(
        self,
        parent,
        abs_1,
        abs_2,
        diff_abs,
        band_names,
        d1_label="Eyes open",
        d2_label="Eyes closed"
    ):
        import numpy as np
        import matplotlib.pyplot as plt
        import mne
        from tkinter import ttk

        for widget in parent.winfo_children():
            widget.destroy()

        info = self.analysis_results["topo_info"]
        ch_names = info["ch_names"]
        n_bands = len(band_names)

        # =====================================================
        # Helper skala warna
        # =====================================================
        def safe_vlim(vals, symmetric=False):
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                return 0.0, 1.0

            if symmetric:
                m = float(np.max(np.abs(vals)))
                if m < 1e-12:
                    m = 1.0
                return -m, m

            vmin = float(np.min(vals))
            vmax = float(np.max(vals))

            if np.isclose(vmin, vmax):
                pad = abs(vmin) * 0.05 if abs(vmin) > 1e-12 else 1.0
                return vmin - pad, vmax + pad

            return vmin, vmax

        # =====================================================
        # Buat figure 3 x n_bands = 12 topomap kalau band = 4
        # =====================================================
        fig, axs = plt.subplots(
            3,
            n_bands,
            figsize=(4.2 * n_bands, 12.5),
            squeeze=False
        )

        fig.patch.set_facecolor("#ffffff")
        # fig.suptitle(
        #     f"Absolute Power Topomap Subtraction: {d2_label} - {d1_label}",
        #     fontsize=12,
        #     fontweight="bold",
        #     y=0.985
        # )

        row_labels = [
            f"Eyes open\n{d1_label}",
            f"Eyes closed\n{d2_label}",
            "Selisih\nClosed - Open"
        ]

        # =====================================================
        # Plot 12 topomap
        # Setiap topomap punya vmin/vmax sendiri
        # =====================================================
        for b_idx, band in enumerate(band_names):
            vals1 = np.array(abs_1[band], dtype=float)
            vals2 = np.array(abs_2[band], dtype=float)
            valsd = np.array(diff_abs[band], dtype=float)

            # Skala masing-masing topomap
            vmin1, vmax1 = safe_vlim(vals1, symmetric=False)
            vmin2, vmax2 = safe_vlim(vals2, symmetric=False)

            # Untuk difference, saya sarankan tetap symmetric=True
            # supaya nilai negatif/positif seimbang terhadap nol.
            vmind, vmaxd = safe_vlim(valsd, symmetric=True)

            # =========================
            # Row 1: Durasi 1
            # =========================
            im1, _ = mne.viz.plot_topomap(
                vals1,
                info,
                axes=axs[0, b_idx],
                show=False,
                cmap=LIGHT_JET,
                extrapolate="head",
                vlim=(vmin1, vmax1),
                sphere=0.1
            )

            axs[0, b_idx].set_title(
                f"{band}\n[{vmin1:.3g}, {vmax1:.3g}]",
                fontsize=8,
                fontweight="bold",
                pad=10
            )

            fig.colorbar(im1, ax=axs[0, b_idx], shrink=0.55, fraction=0.035, pad=0.02)

            # =========================
            # Row 2: Durasi 2
            # =========================
            im2, _ = mne.viz.plot_topomap(
                vals2,
                info,
                axes=axs[1, b_idx],
                show=False,
                cmap=LIGHT_JET,
                extrapolate="head",
                vlim=(vmin2, vmax2),
                sphere=0.1
            )

            axs[1, b_idx].set_title(
                f"{band}\n[{vmin2:.3g}, {vmax2:.3g}]",
                fontsize=9,
                fontweight="bold"
            )

            fig.colorbar(im2, ax=axs[1, b_idx], shrink=0.55, fraction=0.035, pad=0.02)

            # =========================
            # Row 3: Difference D2 - D1
            # =========================
            imd, _ = mne.viz.plot_topomap(
                valsd,
                info,
                axes=axs[2, b_idx],
                show=False,
                cmap=LIGHT_JET,
                extrapolate="head",
                vlim=(vmind, vmaxd),
                sphere=0.1
            )

            axs[2, b_idx].set_title(
                f"{band} Δ\n[{vmind:.3g}, {vmaxd:.3g}]",
                fontsize=9,
                fontweight="bold"
            )

            fig.colorbar(imd, ax=axs[2, b_idx], shrink=0.55, fraction=0.035, pad=0.02)

        # =====================================================
        # Label baris
        # =====================================================
        for r in range(3):
            axs[r, 0].set_ylabel(
                row_labels[r],
                fontsize=10,
                fontweight="bold",
                rotation=0,
                labelpad=45,
                va="center"
            )

        fig.subplots_adjust(
            left=0.10,
            right=0.96,
            bottom=0.06,
            top=0.90,
            wspace=0.55,
            hspace=0.80
        )

        # =====================================================
        # Embed figure ke GUI
        # =====================================================
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.BOTH, expand=True)

        self._embed_canvas(fig, top_frame)

        # =====================================================
        # Tabel nilai
        # =====================================================
        table_frame = ttk.LabelFrame(parent, text=" Nilai Topomap per Channel ", padding=6)
        table_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        cols = ["Channel"]
        for band in band_names:
            cols += [f"{band}_D1", f"{band}_D2", f"{band}_D2-D1"]

        tree = ttk.Treeview(
            table_frame,
            columns=cols,
            show="headings",
            height=len(ch_names)
        )

        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=85, anchor="center")

        for ch_i, ch in enumerate(ch_names):
            row = [ch]

            for band in band_names:
                row += [
                    f"{abs_1[band][ch_i]:.5f}",
                    f"{abs_2[band][ch_i]:.5f}",
                    f"{diff_abs[band][ch_i]:.5f}",
                ]

            tree.insert("", "end", values=row)

        sb_y = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb_y.set)

        tree.pack(side="left", fill="x", expand=True)
        sb_y.pack(side="right", fill="y")
            
        
    def start_topomap_subtraction_thread(self):
        import threading

        if not self.diff_topo_files:
            messagebox.showwarning(
                "Topomap Subtraction",
                "Pilih satu atau beberapa file terlebih dahulu."
            )
            return

        threading.Thread(target=self.run_topomap_subtraction, daemon=True).start()


    def run_topomap_subtraction(self):
        import traceback
        import numpy as np

        try:
            self.root.after(0, lambda: self.status_label.config(
                text="STATUS: RUNNING TOPO MAP SUBTRACTION...",
                fg="#f29900"
            ))

            # Ambil durasi 1
            d1_start = float(self.diff1_start_var.get())
            d1_end_str = self.diff1_end_var.get().strip()
            d1_end = float(d1_end_str) if d1_end_str else None

            # Ambil durasi 2
            d2_start = float(self.diff2_start_var.get())
            d2_end_str = self.diff2_end_var.get().strip()
            d2_end = float(d2_end_str) if d2_end_str else None

            if d1_end is None or d2_end is None:
                raise ValueError("Untuk pengurangan topomap, D1 End dan D2 End wajib diisi.")

            if d1_end <= d1_start:
                raise ValueError("Durasi 1 tidak valid: D1 End harus lebih besar dari D1 Start.")

            if d2_end <= d2_start:
                raise ValueError("Durasi 2 tidak valid: D2 End harus lebih besar dari D2 Start.")

            # Proses Durasi 1 dan Durasi 2
            avg_abs_1, band_names, topo_info = self.compute_average_abs_power_for_window(
                self.diff_topo_files,
                d1_start,
                d1_end
            )

            avg_abs_2, band_names_2, topo_info_2 = self.compute_average_abs_power_for_window(
                self.diff_topo_files,
                d2_start,
                d2_end
            )

            if list(band_names) != list(band_names_2):
                raise ValueError("Band names Durasi 1 dan Durasi 2 tidak sama.")

            # Hitung difference: Durasi 2 - Durasi 1
            diff_abs = {}

            for band in band_names:
                diff_abs[band] = np.array(avg_abs_2[band], dtype=float) - np.array(avg_abs_1[band], dtype=float)

            def update_ui():
                if not hasattr(self, "analysis_results") or self.analysis_results is None:
                    self.analysis_results = {}

                self.analysis_results["topo_info"] = topo_info

                self.plot_topomap_subtraction_result(
                    parent=self.diff_topo_plot_frame,
                    abs_1=avg_abs_1,
                    abs_2=avg_abs_2,
                    diff_abs=diff_abs,
                    band_names=band_names,
                    d1_label=f"{d1_start:g}-{d1_end:g}s",
                    d2_label=f"{d2_start:g}-{d2_end:g}s"
                )

                self.status_label.config(
                    text=f"STATUS: TOPO MAP SUBTRACTION DONE — {len(self.diff_topo_files)} files",
                    fg="green"
                )

                messagebox.showinfo(
                    "Topomap Subtraction",
                    "Pengurangan topomap selesai.\n\n"
                    f"Jumlah file: {len(self.diff_topo_files)}\n"
                    f"Eyes Open: {d1_start:g}-{d1_end:g}s\n"
                    f"Eyes Closed: {d2_start:g}-{d2_end:g}s\n"
                    "Arah: Durasi 2 - Durasi 1"
                )

            self.root.after(0, update_ui)

        except Exception as e:
            err = traceback.format_exc()
            print("[TOPO DIFF ERROR]")
            print(err)

            self.root.after(0, lambda: messagebox.showerror(
                "Topomap Subtraction Error",
                f"Gagal membuat pengurangan topomap:\n{e}"
            ))

            self.root.after(0, lambda: self.status_label.config(
                text="STATUS: TOPO MAP SUBTRACTION ERROR",
                fg="red"
            ))
            
    def compute_average_abs_power_for_window(self, file_paths, t_start, t_end):
        """
        Proses beberapa file pada satu window waktu,
        lalu rata-ratakan absolute power per channel dan per band.
        """
        import numpy as np
        import os

        abs_list = []
        first_topo_info = None
        band_names_ref = None

        for file_path in file_paths:
            res, eeg_crop = self.process_one_file_for_average_topomap(
                file_path=file_path,
                t_start=t_start,
                t_end=t_end
            )

            if first_topo_info is None:
                first_topo_info = res["topo_info"]

            abs_power = res["bands"]["abs_power"]
            band_names = res["bands"]["names"]

            if band_names_ref is None:
                band_names_ref = band_names
            else:
                if list(band_names) != list(band_names_ref):
                    raise ValueError(
                        f"Band names tidak konsisten pada file {os.path.basename(file_path)}"
                    )

            abs_list.append(abs_power)

        avg_abs = {}

        for band in band_names_ref:
            avg_abs[band] = np.mean(
                [np.array(abs_power[band], dtype=float) for abs_power in abs_list],
                axis=0
            )

        return avg_abs, band_names_ref, first_topo_info
    
    def select_subtract_topomap_files(self):
        from tkinter import filedialog
        import os

        paths = filedialog.askopenfilenames(
            title="Select EEG files for topomap subtraction",
            filetypes=[
                ("EEG files", "*.txt *.csv"),
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ]
        )

        if not paths:
            return

        self.diff_topo_files = list(paths)
        self.diff_topo_label_var.set(f"{len(self.diff_topo_files)} files selected")

        print("\n[TOPO DIFF] Files selected:")
        for p in self.diff_topo_files:
            print(" ", os.path.basename(p))


    def clear_subtract_topomap_files(self):
        self.diff_topo_files = []
        self.diff_topo_label_var.set("No files selected")

        if hasattr(self, "diff_topo_plot_frame"):
            for widget in self.diff_topo_plot_frame.winfo_children():
                widget.destroy()
                
    def build_topomap_subtraction_tab(self, parent):
        """
        Sub-tab Brain Mapping untuk pengurangan topomap absolute power.

        Mekanisme:
        - pilih satu / beberapa file
        - proses durasi 1
        - proses durasi 2
        - hitung selisih absolute power: Durasi 2 - Durasi 1
        - tampilkan 12 topomap: 4 band x 3 baris
        """
        for widget in parent.winfo_children():
            widget.destroy()

        ctrl = ttk.LabelFrame(parent, text=" TOPO MAP SUBTRACTION — ABSOLUTE POWER ", padding=8)
        ctrl.pack(fill=tk.X, padx=8, pady=6)

        tk.Button(
            ctrl,
            text="📂 Select Files",
            command=self.select_subtract_topomap_files,
            bg="#f8f9fa",
            fg=self.colors["accent"],
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            padx=10
        ).pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl, textvariable=self.diff_topo_label_var, width=22).pack(side=tk.LEFT, padx=5)

        # Durasi 1
        ttk.Label(ctrl, text="D1 Start:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(ctrl, textvariable=self.diff1_start_var, width=6).pack(side=tk.LEFT, padx=2)

        ttk.Label(ctrl, text="D1 End:").pack(side=tk.LEFT, padx=(6, 2))
        ttk.Entry(ctrl, textvariable=self.diff1_end_var, width=6).pack(side=tk.LEFT, padx=2)

        # Durasi 2
        ttk.Label(ctrl, text="D2 Start:").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Entry(ctrl, textvariable=self.diff2_start_var, width=6).pack(side=tk.LEFT, padx=2)

        ttk.Label(ctrl, text="D2 End:").pack(side=tk.LEFT, padx=(6, 2))
        ttk.Entry(ctrl, textvariable=self.diff2_end_var, width=6).pack(side=tk.LEFT, padx=2)

        tk.Button(
            ctrl,
            text="RUN SUBTRACT TOPO",
            command=self.start_topomap_subtraction_thread,
            bg=self.colors["accent"],
            fg="white",
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            padx=10
        ).pack(side=tk.LEFT, padx=8)

        tk.Button(
            ctrl,
            text="Clear",
            command=self.clear_subtract_topomap_files,
            bg="#e8eaed",
            fg=self.colors["text"],
            relief=tk.FLAT,
            font=("Segoe UI", 9),
            padx=8
        ).pack(side=tk.LEFT, padx=4)

        info = (
            "Arah pengurangan: Durasi 2 - Durasi 1. "
            "Nilai positif berarti absolute power lebih tinggi pada Durasi 2."
        )
        ttk.Label(parent, text=info).pack(anchor="w", padx=12, pady=(0, 4))

        self.diff_topo_plot_frame = ttk.Frame(parent)
        self.diff_topo_plot_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        
    # =========================================
    # FUNGSI UTAMA: UPDATE PLOT
    # =========================================
    def update_live_plot(self):
            if self.is_configuring or not self.board:
                self.root.after(100, self.update_live_plot)
                return

            if self.board and self.streaming:
                try:
                    data = self.board.get_board_data()
                    if data.shape[1] > 0:
                        if self.is_recording:
                            self.session_data.append(data)
                        
# SESUDAH:
                        eeg_idx = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)[:NUM_CHANNELS]
                        current_time = time.time()

                        for i in range(NUM_CHANNELS):
                            # Update data buffer
                            new_sig = data[eeg_idx[i]]
                            self.data_buffer[i] = np.roll(self.data_buffer[i], -len(new_sig))
                            self.data_buffer[i, -len(new_sig):] = new_sig
                            
                            ax = self.ax_flat[i] # PAKAI AX_FLAT

                            # Logika Cek Impedansi
                            if self.imp_states[i]:
                                # Update hitungan tiap interval
                                if current_time - self.last_impedance_time[i] > self.IMP_INTERVAL:
                                    raw_chunk = self.data_buffer[i, -FS:]
                                    z_val = calculate_impedance_openbci(raw_chunk)
                                    self.last_known_imp[i] = z_val
                                    self.last_impedance_time[i] = current_time

                                self.lines[i].set_data(self.time_axis, self.data_buffer[i])
                                ax.set_ylim(-200000, 200000)
                            else:
                                # Mode EEG Normal
                                plot_sig = self.data_buffer[i] - np.mean(self.data_buffer[i])
                                self.lines[i].set_data(self.time_axis, plot_sig)
                                
                                # Auto-scale
                                max_val = np.max(np.abs(plot_sig))
                                limit = max(100, max_val * 1.2)
                                ax.set_ylim(-limit, limit)

                            # 3. UPDATE TEKS (Pastikan ini selalu terpanggil)
                            # 3. UPDATE TEKS (Ubah bagian ini)
                            if self.last_known_imp[i] is not None:
                                z_val = self.last_known_imp[i]
                                
                                # Hapus logika "if z_val > 500" yang sebelumnya menulis RAIL
                                # Sekarang kita langsung masukkan angka kilo ohm
                                self.labels_imp_text[i].set_text(f"{z_val:.1f} kΩ")
                                
                                # Kita tetap gunakan warna untuk indikator kualitas
                                if z_val < 750:
                                    color = "lime"      # Bagus (0 - 15 kOhm)
                                elif z_val < 2500:
                                    color = "yellow"    # Cukup (15 - 50 kOhm)
                                else:
                                    color = "red"       # Sangat Buruk / Lepas (> 100 kOhm)
                                
                                self.labels_imp_text[i].set_color(color)

                        # Update Digital (Index 6)
                        new_dig = (data[13] > 0).astype(float)
                        self.data_buffer[NUM_CHANNELS] = np.roll(self.data_buffer[NUM_CHANNELS], -len(new_dig))
                        self.data_buffer[NUM_CHANNELS, -len(new_dig):] = new_dig
                        self.dig_line.set_data(self.time_axis, self.data_buffer[NUM_CHANNELS])

                        self.canvas.draw_idle()

                except Exception as e:
                    # Jika ada error, print agar kita tahu di baris mana
                    print(f"Update Plot Error: {e}")

            self.root.after(100, self.update_live_plot)

    # ANALISIS (Sama seperti sebelumnya)
    # ==============================
    def start_analysis_thread(self):
            """
            Versi Revisi: Mendukung analisis dari data LIVE maupun FILE TXT.
            """
            # 1. Validasi keberadaan data berdasarkan sumbernya
            if self.current_source == "LIVE":
                # Jika sumbernya LIVE, pastikan user sudah merekam data sebelumnya
                if not hasattr(self, 'session_data') or len(self.session_data) == 0:
                    messagebox.showwarning("Analysis Warning", 
                                        "Tidak ada data rekaman live. Silakan Connect dan Record data terlebih dahulu.")
                    return
            
            elif self.current_source == "FILE":
                # Jika sumbernya FILE, pastikan data session_data sudah terisi dari load_txt_file
                if not hasattr(self, 'session_data') or len(self.session_data) == 0:
                    messagebox.showwarning("Analysis Warning", 
                                        "Data dari file belum dimuat dengan benar.")
                    return

            # 2. Membersihkan tampilan grafik Real-time sebelum menampilkan hasil analisis
            # Hapus label impedansi agar tidak menumpuk di grafik analisis nantinya
            self.last_known_imp = [None] * NUM_CHANNELS
            for i in range(NUM_CHANNELS):
                if i < len(self.labels_imp_text):
                    self.labels_imp_text[i].set_text("")
            
            # 3. Update Status UI (Tema Medical Clean)
            # Gunakan warna Orange (#f29900) agar terlihat mencolok di latar putih sebagai tanda 'Proses'
            source_name = "OFFLINE FILE" if self.current_source == "FILE" else "LIVE RECORDING"
            self.status_label.config(
                text=f"STATUS: ANALYZING {source_name}... (Please Wait)", 
                fg="#f29900" 
            )
            
            # Nonaktifkan tombol sementara agar tidak diklik dua kali saat proses berjalan
            self.btn_analyze.config(state=tk.DISABLED, bg="#e8eaed", fg="#9aa0a6")

            # 4. Jalankan pipeline analisis di background thread agar GUI tidak membeku (freeze)
            threading.Thread(target=self.run_analysis_pipeline, daemon=True).start()
            
    def run_analysis_pipeline(self):
        import os
        import tempfile
        import re
        # Import fungsi fitur dari olah_data.py (pastikan file ada di folder yang sama)
        from olah_data import extract_bands, extract_eeg_features, extract_ppg_features

        _tmp_path = None  # Untuk cleanup di finally

        try:
            # 1. VALIDASI file sumber
            if not hasattr(self, 'last_filename') or not self.last_filename:
                raise FileNotFoundError("Path file kosong.")
            if not os.path.exists(self.last_filename):
                raise FileNotFoundError(f"File tidak ditemukan: {self.last_filename}")

            self.root.after(0, lambda: self.status_label.config(
                text="STATUS: PROCESSING...", fg="orange"))

            # 2. Baca file sumber (skip baris komentar %)
            # 2. Baca file sumber
            # Kalau file sudah pernah dimuat dari tombol LOAD FILE, gunakan offline_df
            # agar TXT dan CSV sama-sama masuk sebagai format standar:
            # 8 EEG channel + Digital/PPG.
            if self.current_source == "FILE" and hasattr(self, "offline_df") and self.offline_df is not None:
                raw_df = self.offline_df.copy()
            else:
                loaded = self.read_offline_eeg_file(self.last_filename)

                if isinstance(loaded, tuple):
                    raw_df, detected_format = loaded
                else:
                    raw_df = loaded
                    detected_format = "unknown"
                    
            total_samples = len(raw_df)
            total_duration = total_samples / FS

            # 3. Parse Start / End time dari input GUI
            try:
                t_start = float(self.analysis_start_var.get())
            except (ValueError, AttributeError):
                t_start = 0.0

            try:
                end_str = self.analysis_end_var.get().strip()
                t_end = float(end_str) if end_str else total_duration
            except (ValueError, AttributeError):
                t_end = total_duration

            # Clamp agar tidak keluar batas
            t_start = max(0.0, min(t_start, total_duration - 1.0))
            t_end   = max(t_start + 1.0, min(t_end, total_duration))

            s_start = int(t_start * FS)
            s_end   = int(t_end   * FS)

            print(f"[ANALYSIS] Crop: {t_start:.1f}s – {t_end:.1f}s "
                f"({s_end - s_start} samples dari total {total_samples})")

            # 4. Crop dataframe
            cropped_df = raw_df.iloc[s_start:s_end].reset_index(drop=True)

            # 5. Simpan ke file TXT sementara dengan format yang sama persis
            #    seperti file asli (header komentar % + kolom tab-separated)
            tmp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix='.txt', mode='w', encoding='utf-8')
            _tmp_path = tmp_file.name

            # 5. Simpan ke file TXT sementara — format IDENTIK dengan file asli
            #    eeg_processor_8 pakai skiprows=2, jadi:
            #    Baris 1 = komentar bebas (di-skip)
            #    Baris 2 = komentar bebas (di-skip)
            #    Baris 3 = nama kolom (WAJIB, langsung dibaca sebagai header)
            #    Baris 4+ = data angka
            tmp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix='.txt', mode='w', encoding='utf-8')
            _tmp_path = tmp_file.name

            tmp_file.write(f"% OpenBCI EEG Recording (Cropped {t_start:.1f}s-{t_end:.1f}s)\n")  # baris 1 — di-skip
            tmp_file.write(f"% FS: {FS} Hz\n")                                                    # baris 2 — di-skip
            # Baris 3 = header kolom (TANPA %, langsung terbaca sebagai nama kolom)
            tmp_file.write("\t".join(cropped_df.columns) + "\n")
            tmp_file.close()

            
            with open(_tmp_path, 'a') as f:
                np.savetxt(f, cropped_df.values, delimiter='\t', fmt='%.4f')
                
            skip_ica_value = self.skip_ica_var.get()
            skip_zapline_value = self.skip_zapline_var.get()
            k_thresh = float(self.ica_kurt_var.get())
            bcg_thresh = float(self.ica_bcg_var.get())
            vbr_thresh = float(self.ica_vbr_var.get())

            results = eeg_processor_8.process_eeg_data(
                _tmp_path,
                fs=FS,
                skip_ica=skip_ica_value,
                skip_zapline=skip_zapline_value,
                k_thresh=k_thresh,
                bcg_thresh=bcg_thresh,
                vbr_thresh=vbr_thresh,

                # ========================================================
                # MAICA SPLIT 8CH
                # ========================================================
                maica_n_components=12,
                maica_split_8ch=True,
                maica_group_max_components=12
            )
            # ============================================================
            # TAMBAHAN: HITUNG LZC TOPO MAP DARI EEG CLEAN HASIL ICA
            # ============================================================
            try:
                if "ica" in results and results["ica"].get("data") is not None:
                    eeg_clean_for_lzc = np.asarray(results["ica"]["data"], dtype=float)

                    results["bands"]["lzc_maps"] = self.compute_lzc_maps_from_eeg(
                        eeg_clean_for_lzc,
                        fs=FS
                    )

                    print("[LZC TOPO] lzc_maps berhasil ditambahkan ke results['bands'].")
                    print("[LZC TOPO] keys:", list(results["bands"]["lzc_maps"].keys()))

            except Exception as e:
                print(f"[LZC TOPO] Gagal menghitung LZC maps: {e}")
            # ── PATCH: Hapus edge artifact (spike di ujung) dari hasil ICA ──
            # Spike terjadi karena transien matematis filter/ICA di batas data.
                        # trim_ica_clean  → smooth tepi pada sinyal rekonstruksi (n_ch × n_samp)
                        # trim_ic_components → smooth tepi pada setiap komponen IC untuk plot
            if 'ica' in results:
                if results['ica'].get('data') is not None:
                    results['ica']['data'] = trim_ica_clean(results['ica']['data'])

                if results['ica'].get('after') is not None:
                    results['ica']['after'] = trim_ica_clean(results['ica']['after'])

                if results['ica'].get('ics') is not None:
                    results['ica']['ics'] = trim_ic_components(results['ica']['ics'])

                # ========================================================
                # Tambahan: trim juga IC pada struktur split MAICA
                # ========================================================
                if isinstance(results['ica'].get('ics_split'), dict):
                    for group_key, group_val in results['ica']['ics_split'].items():
                        if isinstance(group_val, dict) and group_val.get('ics') is not None:
                            group_val['ics'] = trim_ic_components(group_val['ics'])
            self.analysis_results = results

            # =====================================================================
            # FORCE DEBUG SNR AWAL (Dipaksa muncul dengan Fallback Perhitungan di GUI)
            # =====================================================================
            print("\n[DEBUG GUI] Memulai pengecekan SNR Awal...")
            
            if 'ica' in results:
                # Cek apakah eeg_processor_8 sudah sukses mengirim nilai avg_snr
                if 'avg_snr' in results['ica']:
                    init_snr = results['ica']['avg_snr']
                    print(f"[DEBUG GUI] SNR ditemukan dari backend.")
                else:
                    # FALLBACK: Jika eeg_processor_8 belum ke-save, kita hitung langsung di GUI
                    print("[DEBUG GUI] Warning: 'avg_snr' tidak ditemukan di file processor. Menghitung mandiri di GUI...")
                    from eeg_processor_8 import calculate_eeg_snr
                    # raw_sig = results['ica']['before']   # Ini adalah subset_clean (setelah Zapline)
                    raw_sig = results['zapline']['raw']  # Ini adalah subset_bpf (sinyal asli sebelum Zapline)
                    clean_sig = results['ica']['after']  # Sinyal sesudah ICA
                    
                    _, init_snr = calculate_eeg_snr(raw_sig, clean_sig)
                    # Masukkan ke dictionary agar sinkron
                    results['ica']['avg_snr'] = init_snr

                # Cetak ke terminal secara paksa (ditambahkan flush=True agar langsung keluar)
                print(f"[INITIAL RUN] SNR Rata-rata Awal: {init_snr:.2f} dB", flush=True)
            else:
                print("[DEBUG GUI] Error: Key 'ica' tidak ditemukan pada output hasil analisis.", flush=True)
            print("=====================================================\n")
            # =====================================================================
            # =============================================================== 

            # === TAMBAHAN: Ekstrak fitur lengkap (EEG + PPG) dari olah_data.py ===
            try:
                # a. Parse label dari nama file (format: Nama<ID>_S<n>_A<n>_D<n>_<G>)
                import re as _re
                basename = os.path.basename(self.last_filename)
                base_no_ext = os.path.splitext(basename)[0]
                m = _re.match(
                    r'^([A-Za-z]+)(\d+)_S(\d+)_A(\d+)_D(\d+)(?:_(M|F))?',
                    base_no_ext
                )
                if m:
                    label_info = {
                        'filename'  : basename,
                        'subject'   : m.group(1),
                        'rec_id'    : int(m.group(2)),
                        'stress'    : int(m.group(3)),
                        'anxiety'   : int(m.group(4)),
                        'depression': int(m.group(5)),
                        'gender'    : m.group(6) if m.group(6) else 'N/A',
                    }
                else:
                    label_info = {
                        'filename'  : basename,
                        'subject'   : base_no_ext,
                        'rec_id'    : 0,
                        'stress'    : None,
                        'anxiety'   : None,
                        'depression': None,
                        'gender'    : 'N/A',
                    }

                # b. Ambil sinyal ICA-clean dari hasil processor
                ica_clean = results['ica']['data']   # shape: (n_ch, n_samp)

                # c. Ekstrak band signals dan EEG features
                band_sig  = extract_bands(ica_clean, FS)
                eeg_feat  = extract_eeg_features(band_sig, FS)

                # d. Ekstrak PPG/HRV features dari kolom digital cropped_df
                if cropped_df.shape[1] > 8:
                    ppg_col = cropped_df.iloc[:, 8].values.astype(float)
                else:
                    ppg_col = np.zeros(len(cropped_df))
                ppg_feat = extract_ppg_features(ppg_col, FS)

                # e. Gabungkan semua ke self.last_features
                self.last_features = {**label_info, **eeg_feat, **ppg_feat}

            except Exception as feat_err:
                print(f"[WARNING] Ekstraksi fitur olah_data gagal: {feat_err}")
                # Fallback: gunakan fitur dari eeg_processor_8 saja
                self.last_features = {
                    'filename': os.path.basename(self.last_filename),
                    **results['features']['dict']
                }
            # === AKHIR TAMBAHAN ===

            # 7. Simpan juga cropped data ke session_data

            # 7. Simpan juga cropped data ke session_data
            #    agar display_last_5s menampilkan rentang yang sama
            self.session_data = cropped_df.iloc[:, :NUM_CHANNELS].values.T

            # 8. Tampilkan hasil
            self.root.after(0, lambda: self.show_visual_results(results))
            self.root.after(0, self.display_last_5s)
            _n_rej_initial = results.get('ica', {}).get('n_rejected', 0)
            _k_init = self.ica_kurt_var.get() if hasattr(self, 'ica_kurt_var') else 2.1
            self.root.after(0, lambda: self.status_label.config(
                text=(f"STATUS: ANALYSIS COMPLETE [{t_start:.1f}s – {t_end:.1f}s] | "
                    f"K≥{_k_init:.1f} | {_n_rej_initial} IC ditolak"),
                fg="green"))

        except FileNotFoundError as fnf_e:
            self.root.after(0, lambda: messagebox.showerror("File Error", str(fnf_e)))
            self.root.after(0, lambda: self.status_label.config(
                text="STATUS: FILE NOT FOUND", fg="red"))

        except Exception as e:
            final_error = str(e)
            print(f"Pipeline Error: {final_error}")
            self.root.after(0, lambda: messagebox.showerror(
                "Analysis Error", f"Terjadi kesalahan:\n{final_error}"))
            self.root.after(0, lambda: self.status_label.config(
                text="STATUS: ERROR IN ANALYSIS", fg="red"))

        finally:
            # Hapus file temp
            if _tmp_path and os.path.exists(_tmp_path):
                try:
                    os.unlink(_tmp_path)
                except Exception:
                    pass
            self.root.after(0, lambda: self.btn_analyze.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.btn_record.config(
                state=tk.NORMAL, text="▶ START RECORDING"))
                    
    def reset_session(self):
        """Reset semua state agar bisa mulai ulang tanpa restart GUI.
        CATATAN: last_features TIDAK direset agar fitur yang sudah diekstrak
        tetap bisa disimpan, atau akan ditimpa oleh analisis berikutnya."""

        # 1. Stop stream kalau masih jalan
        try:
            if self.board and self.streaming:
                self.board.stop_stream()
                time.sleep(1.0) 
        except:
            pass

        # 2. Reset status internal
        self.streaming = False
        self.is_recording = False
        # Batalkan countdown jika sedang berjalan
        if getattr(self, '_countdown_job', None):
            self.root.after_cancel(self._countdown_job)
            self._countdown_job = None
        if hasattr(self, 'countdown_label'):
            self.countdown_label.config(text="")
            
        self.session_data = []
        self.last_filename = ""
        # === PERUBAHAN: last_features SENGAJA tidak direset di sini ===
        # last_features akan di-None-kan secara otomatis setelah berhasil disimpan
        # atau akan ditimpa saat analisis berikutnya dijalankan
        
    def show_visual_results(self, res):
            """Mengaktifkan tab analisis dan mengisi datanya."""
            # 1. Tambahkan "Features" ke dalam list tabs agar frame-nya muncul di notebook
            tabs = [
                ("Zapline", self.zap_tab), 
                ("ICA", self.ica_tab), 
                ("ICA Compare",  self.ica_compare_tab),
                ("Band Signals", self.mif_tab), 
                ("Power Bands", self.band_tab),
                ("BPM Analysis", self.bpm_tab),
                ("Brain Mapping", self.topo_tab),
                ("Features", self.feat_tab),
                ("Saving Feature", self.save_tab),   
                ("Saving Signal",   self.signal_save_tab)  
            ]
            
            
            for name, frame in tabs:
                try: 
                    self.notebook.add(frame, text=f" {name} ")
                except: 
                    pass 

            # 2. Panggil fungsi plotting masing-masing
            # 2. Panggil fungsi plotting masing-masing
            self.plot_multi_zapline(res['zapline'], self.zap_tab)
            self.plot_multi_ica(res['ica'], self.ica_tab)
            self.plot_ica_compare(res['ica'], self.ica_compare_tab)
            self.plot_multi_mif(res['mif'], self.mif_tab)
            self.plot_band_power(res['bands'], self.band_tab)
            self.plot_bpm_analysis(res['bpm_data'], self.bpm_tab)
            self.build_brain_mapping_tab(self.topo_tab, res['bands']) 
            self.display_feature_table(res['features']['table'], self.feat_tab)
            self.build_save_tab(self.save_tab)  
            self.build_signal_save_tab(self.signal_save_tab)  
            
            # Fokuskan ke tab pertama hasil analisis
            self.notebook.select(self.zap_tab)
            
    def build_brain_mapping_tab(self, parent, bands_data=None):
        """
        Membuat tab Brain Mapping dengan:
        1. Panel kontrol average topomap
        2. Area plot topomap
        """
        for widget in parent.winfo_children():
            widget.destroy()

        # =========================
        # CONTROL PANEL
        # =========================
        ctrl = ttk.LabelFrame(parent, text=" AVERAGE TOPO MAP ", padding=8)
        ctrl.pack(fill=tk.X, padx=8, pady=6)

        tk.Button(
            ctrl,
            text="📂 Select Multiple Files",
            command=self.select_average_topomap_files,
            bg="#f8f9fa",
            fg=self.colors["accent"],
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            padx=10
        ).pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl, textvariable=self.avg_topo_label_var, width=28).pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl, text="Start (s):").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Entry(ctrl, textvariable=self.avg_topo_start_var, width=6).pack(side=tk.LEFT, padx=2)

        ttk.Label(ctrl, text="End (s):").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Entry(ctrl, textvariable=self.avg_topo_end_var, width=6).pack(side=tk.LEFT, padx=2)

        tk.Button(
            ctrl,
            text="RUN AVERAGE TOPOMAP",
            command=self.start_average_topomap_thread,
            bg=self.colors["accent"],
            fg="white",
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            padx=10
        ).pack(side=tk.LEFT, padx=8)

        tk.Button(
            ctrl,
            text="Clear",
            command=self.clear_average_topomap_files,
            bg="#e8eaed",
            fg=self.colors["text"],
            relief=tk.FLAT,
            font=("Segoe UI", 9),
            padx=8
        ).pack(side=tk.LEFT, padx=4)

        # =========================
        # PLOT AREA
        # =========================
        self.topo_plot_frame = ttk.Frame(parent)
        self.topo_plot_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Kalau ada data bands dari analisis normal, tampilkan seperti biasa
        if bands_data is not None:
            self.plot_brain_mapping(bands_data, self.topo_plot_frame)

    def select_average_topomap_files(self):
        from tkinter import filedialog
        import os

        paths = filedialog.askopenfilenames(
            title="Select EEG files for average topomap",
            filetypes=[
                ("EEG files", "*.txt *.csv"),
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ]
        )

        if not paths:
            return

        self.avg_topo_files = list(paths)

        self.avg_topo_label_var.set(
            f"{len(self.avg_topo_files)} files selected"
        )

        print("\n[AVERAGE TOPO] Files selected:")
        for p in self.avg_topo_files:
            print(" ", os.path.basename(p))


    def clear_average_topomap_files(self):
        self.avg_topo_files = []
        self.avg_topo_label_var.set("No files selected")
        
    def make_processor_temp_file(self, cropped_df):
        import tempfile
        import numpy as np

        tmp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".txt",
            mode="w",
            encoding="utf-8"
        )

        tmp_path = tmp_file.name

        tmp_file.write("% OpenBCI EEG Recording\n")
        tmp_file.write(f"% FS: {FS} Hz\n")
        tmp_file.write("\t".join(cropped_df.columns) + "\n")
        tmp_file.close()

        with open(tmp_path, "a", encoding="utf-8") as f:
            np.savetxt(f, cropped_df.values, delimiter="\t", fmt="%.8f")

        return tmp_path

    def process_one_file_for_average_topomap(self, file_path, t_start, t_end):
        import os

        # Pakai loader universal yang sama dengan LOAD FILE
        loaded = self.read_offline_eeg_file(file_path)

        # Antisipasi kalau fungsi loader masih ada versi lama yang return 1 nilai
        if isinstance(loaded, tuple):
            raw_df, detected_format = loaded
        else:
            raw_df = loaded
            detected_format = "unknown"

        total_samples = len(raw_df)
        total_duration = total_samples / FS

        if t_end is None:
            t_end_use = total_duration
        else:
            t_end_use = float(t_end)

        t_start_use = float(t_start)

        if t_start_use < 0:
            raise ValueError(f"Start time negatif pada file {os.path.basename(file_path)}")

        if t_end_use <= t_start_use:
            raise ValueError(f"End time harus lebih besar dari start pada file {os.path.basename(file_path)}")

        if t_end_use > total_duration:
            raise ValueError(
                f"File {os.path.basename(file_path)} terlalu pendek.\n"
                f"Durasi file = {total_duration:.2f}s, tetapi diminta sampai {t_end_use:.2f}s."
            )

        s_start = int(round(t_start_use * FS))
        s_end = int(round(t_end_use * FS))

        cropped_df = raw_df.iloc[s_start:s_end].reset_index(drop=True)

        tmp_path = self.make_processor_temp_file(cropped_df)

        try:
            skip_ica_value = self.skip_ica_var.get()
            skip_zapline_value = self.skip_zapline_var.get()

            k_thresh = float(self.ica_kurt_var.get())
            bcg_thresh = float(self.ica_bcg_var.get())
            vbr_thresh = float(self.ica_vbr_var.get())

            # Kalau process_eeg_data kamu sudah menerima threshold
            try:
                res = eeg_processor_8.process_eeg_data(
                    tmp_path,
                    fs=FS,
                    skip_ica=skip_ica_value,
                    skip_zapline=skip_zapline_value,
                    k_thresh=k_thresh,
                    bcg_thresh=bcg_thresh,
                    vbr_thresh=vbr_thresh,

                    # ========================================================
                    # MAICA SPLIT 8CH
                    # ========================================================
                    maica_n_components=12,
                    maica_split_8ch=True,
                    maica_group_max_components=12
                )
            except TypeError:
                # Fallback kalau process_eeg_data belum kamu update dengan threshold
                res = eeg_processor_8.process_eeg_data(
                    tmp_path,
                    fs=FS,
                    skip_ica=skip_ica_value,
                    skip_zapline=skip_zapline_value
                )

        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        # EEG yang dipakai untuk SASI average
        eeg_crop = cropped_df.iloc[:, :NUM_CHANNELS].values.T.astype(float)

        print(
            f"[AVERAGE TOPO] Done: {os.path.basename(file_path)} | "
            f"format={detected_format} | crop={t_start_use:.1f}-{t_end_use:.1f}s"
        )

        return res, eeg_crop


    def compute_topomap_metric_bundle(self, res, eeg_crop):
        from scipy.signal import welch as sp_welch
        import numpy as np

        bands = res["bands"]
        abs_power = bands["abs_power"]
        rel_power = bands["rel_power"]
        band_names = bands["names"]

        # =========================
        # Ratio
        # =========================
        ratio_defs = [
            ("Delta/Theta", "Delta", "Theta"),
            ("Delta/Alpha", "Delta", "Alpha"),
            ("Delta/Beta",  "Delta", "Beta"),
            ("Theta/Alpha", "Theta", "Alpha"),
            ("Theta/Beta",  "Theta", "Beta"),
            ("Alpha/Beta",  "Alpha", "Beta"),
        ]

        ratios = {}

        for label, nb, db in ratio_defs:
            if nb in abs_power and db in abs_power:
                num = np.array(abs_power[nb], dtype=float)
                den = np.array(abs_power[db], dtype=float)
                den = np.where(den == 0, 1e-10, den)
                ratios[label] = num / den

        # =========================
        # Asymmetry
        # =========================
        asym_pairs = [
            ("Fp1-Fp2", 0, 1),
            ("F3-F4",   2, 3),
            ("T3-T4",   4, 5),
            ("P3-P4",   6, 7),
        ]
        ch_to_pair = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3}

        asym_by_band = {}

        for band in band_names:
            p = np.array(abs_power[band], dtype=float)
            asym_topo = np.zeros(NUM_CHANNELS)

            for ch_idx, pair_idx in ch_to_pair.items():
                _, li, ri = asym_pairs[pair_idx]
                L, R = p[li], p[ri]
                denom = L + R
                asym_topo[ch_idx] = (L - R) / denom if denom != 0 else 0.0

            asym_by_band[band] = asym_topo

        # =========================
        # SASI
        # =========================
        freqs, psd_all = sp_welch(
            eeg_crop[:NUM_CHANNELS],
            FS,
            nperseg=min(500, eeg_crop.shape[1]),
            axis=1
        )

        alpha_mask = (freqs >= 8) & (freqs <= 12)

        if np.any(alpha_mask):
            fc = freqs[alpha_mask][np.argmax(psd_all[:, alpha_mask].mean(axis=0))]
            sasi_values = np.array([
                self.compute_sasi(psd_all[ch], freqs, fc)
                for ch in range(NUM_CHANNELS)
            ])
        else:
            sasi_values = np.zeros(NUM_CHANNELS)

        return {
            "abs_power": abs_power,
            "rel_power": rel_power,
            "ratios": ratios,
            "asym_by_band": asym_by_band,
            "sasi_values": sasi_values,
            "band_names": band_names
        }
        
    def average_topomap_metric_bundles(self, bundles):
        import numpy as np

        band_names = bundles[0]["band_names"]

        # =========================
        # Average absolute & relative power
        # =========================
        avg_abs = {}
        avg_rel = {}

        for band in band_names:
            avg_abs[band] = np.mean(
                [np.array(b["abs_power"][band], dtype=float) for b in bundles],
                axis=0
            ).tolist()

            avg_rel[band] = np.mean(
                [np.array(b["rel_power"][band], dtype=float) for b in bundles],
                axis=0
            ).tolist()

        # =========================
        # Average ratio
        # =========================
        ratio_labels = bundles[0]["ratios"].keys()
        avg_ratios = {}

        for label in ratio_labels:
            avg_ratios[label] = np.mean(
                [np.array(b["ratios"][label], dtype=float) for b in bundles],
                axis=0
            )

        # =========================
        # Average asymmetry
        # =========================
        avg_asym = {}

        for band in band_names:
            avg_asym[band] = np.mean(
                [np.array(b["asym_by_band"][band], dtype=float) for b in bundles],
                axis=0
            ).tolist()

        # =========================
        # Average SASI
        # =========================
        avg_sasi = np.mean(
            [np.array(b["sasi_values"], dtype=float) for b in bundles],
            axis=0
        )

        avg_bands = {
            "abs_power": avg_abs,
            "rel_power": avg_rel,
            "names": band_names,
            "ranges": bundles[0].get("ranges", {}),
            "powers": {band: float(np.mean(avg_rel[band])) for band in band_names},

            # Tambahan untuk plot_brain_mapping
            "ratios": avg_ratios,
            "asym_by_band": avg_asym,
            "sasi_values": avg_sasi,
            "is_average": True,
            "n_files": len(bundles)
        }

        return avg_bands

    def start_average_topomap_thread(self):
        import threading

        if not self.avg_topo_files:
            messagebox.showwarning(
                "Average Topomap",
                "Pilih beberapa file terlebih dahulu."
            )
            return

        threading.Thread(target=self.run_average_topomap, daemon=True).start()


    def run_average_topomap(self):
        import os
        import traceback

        try:
            self.root.after(0, lambda: self.status_label.config(
                text="STATUS: RUNNING AVERAGE TOPO MAP...",
                fg="#f29900"
            ))

            try:
                t_start = float(self.avg_topo_start_var.get())
            except Exception:
                t_start = 0.0

            end_str = self.avg_topo_end_var.get().strip()
            t_end = float(end_str) if end_str else None

            bundles = []
            first_topo_info = None

            for idx, file_path in enumerate(self.avg_topo_files):
                res, eeg_crop = self.process_one_file_for_average_topomap(
                    file_path=file_path,
                    t_start=t_start,
                    t_end=t_end
                )

                if first_topo_info is None:
                    first_topo_info = res["topo_info"]

                bundle = self.compute_topomap_metric_bundle(res, eeg_crop)

                # Simpan ranges kalau ada
                bundle["ranges"] = res["bands"].get("ranges", {})

                bundles.append(bundle)

            avg_bands = self.average_topomap_metric_bundles(bundles)

            def update_ui():
                if not hasattr(self, "analysis_results") or self.analysis_results is None:
                    self.analysis_results = {}

                # plot_brain_mapping butuh topo_info dari self.analysis_results
                self.analysis_results["topo_info"] = first_topo_info

                # Tampilkan average ke area plot brain map
                self.plot_brain_mapping(avg_bands, self.topo_plot_frame)

                self.status_label.config(
                    text=f"STATUS: AVERAGE TOPO MAP DONE — {len(self.avg_topo_files)} files",
                    fg="green"
                )

                messagebox.showinfo(
                    "Average Topomap",
                    f"Average topomap selesai.\n\n"
                    f"Jumlah file: {len(self.avg_topo_files)}\n"
                    f"Window: {t_start:.1f}s - {t_end if t_end is not None else 'end'}s"
                )

            self.root.after(0, update_ui)

        except Exception as e:
            err = traceback.format_exc()
            print("[AVERAGE TOPO ERROR]")
            print(err)

            self.root.after(0, lambda: messagebox.showerror(
                "Average Topomap Error",
                f"Gagal membuat average topomap:\n{e}"
            ))

            self.root.after(0, lambda: self.status_label.config(
                text="STATUS: AVERAGE TOPO MAP ERROR",
                fg="red"
            ))
            
            
            
    def plot_bpm_analysis(self, data, parent): # Pastikan namanya 'data'
        for widget in parent.winfo_children():
            widget.destroy()
            
    def plot_bpm_analysis(self, data, parent):
        """Plot BPM analysis dengan background putih dan tanpa visualisasi midpoint peak."""
        for widget in parent.winfo_children():
            widget.destroy()

        # ============================================================
        # FIGURE DASAR PUTIH
        # ============================================================
        fig, axs = plt.subplots(4, 1, figsize=(10, 8))
        fig.patch.set_facecolor("#ffffff")

        # Warna tema light mode
        bg_color = "#ffffff"
        text_color = "#202124"
        grid_color = "#e5e7eb"
        spine_color = "#d0d7de"

        ppg_color = "#00acc1"     # cyan lebih nyaman di background putih
        rr_color = "#f9a825"      # amber, pengganti yellow agar terlihat
        bpm_color = "#00c853"     # hijau

        # ============================================================
        # 1. PPG / DIGITAL PULSE
        # ============================================================
        if "ppg_signal" in data:
            axs[0].plot(
                data["ppg_time"],
                data["ppg_signal"],
                color=ppg_color,
                lw=1.0
            )

            axs[0].set_title(
                "PPG Pulse Signal",
                color=text_color,
                fontsize=11,
                fontweight="bold"
            )
            axs[0].set_ylabel("Binary", color=text_color)

            # ========================================================
            # MIDPOINT PEAK SEMENTARA DIMATIKAN
            # Jangan hapus ppg_peaks dari data, karena RR/BPM masih butuh peak.
            # Yang dimatikan hanya visualisasi titik birunya.
            # ========================================================
            # ppg_peaks = data.get("ppg_peaks", np.array([]))
            # if len(ppg_peaks) > 0:
            #     axs[0].scatter(
            #         data["ppg_time"][ppg_peaks],
            #         data["ppg_signal"][ppg_peaks],
            #         color="blue",
            #         s=30,
            #         zorder=5,
            #         label="Midpoint Peak"
            #     )
            #     axs[0].legend(fontsize=8)

        # ============================================================
        # 2. RR INTERVAL
        # ============================================================
        if "rr_values" in data:
            axs[1].plot(
                data["rr_times"],
                data["rr_values"],
                color=rr_color,
                marker="o",
                markersize=4,
                lw=1.2
            )

            axs[1].set_title(
                "RR Interval (ms)",
                color=text_color,
                fontsize=11,
                fontweight="bold"
            )
            axs[1].set_ylabel("ms", color=text_color)

        # ============================================================
        # 3. INSTANT BPM
        # ============================================================
        if "bpm_values" in data:
            axs[2].plot(
                data["bpm_times"],
                data["bpm_values"],
                color=bpm_color,
                marker="o",
                markersize=4,
                lw=1.2
            )

            axs[2].set_title(
                "Instant BPM",
                color=text_color,
                fontsize=11,
                fontweight="bold"
            )
            axs[2].set_ylabel("BPM", color=text_color)

        # ============================================================
        # 4. SUMMARY BOX
        # ============================================================
        axs[3].axis("off")
        axs[3].set_facecolor(bg_color)

        avg_bpm = data.get("mean_bpm", 0)
        sdnn = data.get("sdnn", 0)
        rmssd = data.get("rmssd", 0)

        info_text = (
            f"Heart Rate Summary\n\n"
            f"Mean BPM : {avg_bpm:.2f}\n"
            f"SDNN     : {sdnn:.2f} ms\n"
            f"RMSSD    : {rmssd:.2f} ms\n"
            f"Status   : {data.get('status', 'N/A')}"
        )

        axs[3].text(
            0.5,
            0.5,
            info_text,
            color=text_color,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(
                facecolor="#ffffff",
                edgecolor=ppg_color,
                linewidth=1.2,
                boxstyle="round,pad=0.6"
            )
        )

        # ============================================================
        # STYLE SEMUA AXIS AGAR BENAR-BENAR LIGHT MODE
        # ============================================================
        for ax in axs[:3]:
            ax.set_facecolor(bg_color)

            ax.grid(
                True,
                linestyle="-",
                linewidth=0.5,
                color=grid_color,
                alpha=0.9
            )

            ax.tick_params(
                axis="both",
                colors=text_color,
                labelsize=8
            )

            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.title.set_color(text_color)

            for spine in ax.spines.values():
                spine.set_color(spine_color)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        self._embed_canvas(fig, parent)
        
        
    def plot_multi_zapline(self, data, parent):
        for widget in parent.winfo_children():
            widget.destroy()

        t, fs = data['t'], 250
        n_ch = data['raw'].shape[0]

        fig, axs = plt.subplots(4, 4, figsize=(18, 10))
        fig.patch.set_facecolor("#ffffff")

        for i in range(4):
            # =========================
            # KOLOM 1: CH1–CH4 (TIME)
            # =========================
            if i < n_ch:
                axs[i, 0].plot(t, data['raw'][i], color='red', alpha=0.4)
                axs[i, 0].plot(t, data['clean'][i], color='green', lw=0.8)
                axs[i, 0].set_ylabel(f"CH {i+1}", fontsize=8)

            # =========================
            # KOLOM 2: CH5–CH8 (TIME)
            # =========================
            ch_idx = i + 4
            if ch_idx < n_ch:
                axs[i, 1].plot(t, data['raw'][ch_idx], color='red', alpha=0.4)
                axs[i, 1].plot(t, data['clean'][ch_idx], color='green', lw=0.8)
                axs[i, 1].set_ylabel(f"CH {ch_idx+1}", fontsize=8)

            # =========================
            # KOLOM 3: CH1–CH4 (PSD)
            # =========================
            if i < n_ch:
                f_raw, p_raw = welch(data['raw'][i], fs, nperseg=fs*2)
                f_cln, p_cln = welch(data['clean'][i], fs, nperseg=fs*2)

                axs[i, 2].semilogy(f_raw, p_raw, color='red', alpha=0.5)
                axs[i, 2].semilogy(f_cln, p_cln, color='green', lw=1)
                axs[i, 2].set_xlim(2, 80)

            # =========================
            # KOLOM 4: CH5–CH8 (PSD)
            # =========================
            if ch_idx < n_ch:
                f_raw, p_raw = welch(data['raw'][ch_idx], fs, nperseg=fs*2)
                f_cln, p_cln = welch(data['clean'][ch_idx], fs, nperseg=fs*2)

                axs[i, 3].semilogy(f_raw, p_raw, color='red', alpha=0.5)
                axs[i, 3].semilogy(f_cln, p_cln, color='green', lw=1)
                axs[i, 3].set_xlim(2, 80)

        # =========================
        # STYLE (BIAR CONSISTENT)
        # =========================
        for row in range(4):
            for col in range(4):
                ax = axs[row, col]
                ax.grid(True, linestyle=':', alpha=0.3)
                ax.tick_params(labelsize=7)

        # TITLE KOLOM
        axs[0, 0].set_title("EEG CH1–CH4")
        axs[0, 1].set_title("EEG CH5–CH8")
        axs[0, 2].set_title("PSD CH1–CH4")
        axs[0, 3].set_title("PSD CH5–CH8")

        plt.tight_layout()
        self._embed_canvas(fig, parent)
    def plot_multi_ica_split(self, data, parent):
        """
        Menampilkan hasil split MAICA dalam dua sub-tab:
        1. Physical 4CH   : Fp1, Fp2, T3, T4
        2. GT Virtual 4CH : F3, F4, P3, P4
        """
        for widget in parent.winfo_children():
            widget.destroy()

        ics_split = data.get("ics_split", {})
        split_stats = data.get("split_stats", {}) or {}

        # ============================================================
        # Header ringkasan
        # ============================================================
        header = ttk.Frame(parent)
        header.pack(fill=tk.X, padx=8, pady=4)

        phys_txt = (
            f"Physical 4CH: "
            f"{split_stats.get('physical_rejected', '-')}/"
            f"{split_stats.get('physical_total', '-')}"
        )

        virt_txt = (
            f"GT Virtual 4CH: "
            f"{split_stats.get('virtual_gt_rejected', '-')}/"
            f"{split_stats.get('virtual_gt_total', '-')}"
        )

        tk.Label(
            header,
            text=f"Split MAICA Mode  |  {phys_txt}  |  {virt_txt}",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors["bg"],
            fg=self.colors["accent"]
        ).pack(side=tk.LEFT, padx=4)

        tk.Label(
            header,
            text="Setiap grup diproses dengan MAICA terpisah, max 12 IC per grup.",
            font=("Segoe UI", 8),
            bg=self.colors["bg"],
            fg=self.colors["text_dim"]
        ).pack(side=tk.LEFT, padx=12)

        # ============================================================
        # Notebook sub-tab
        # ============================================================
        sub_nb = ttk.Notebook(parent)
        sub_nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        group_order = [
            ("physical", " Physical 4CH "),
            ("virtual_gt", " GT Virtual 4CH ")
        ]

        for group_key, tab_title in group_order:
            group = ics_split.get(group_key, {})

            frame = ttk.Frame(sub_nb)
            sub_nb.add(frame, text=tab_title)

            if not isinstance(group, dict) or not group.get("ics"):
                ttk.Label(
                    frame,
                    text=f"Data {tab_title} tidak tersedia.",
                    foreground="gray"
                ).pack(padx=10, pady=10)
                continue

            group_data = {
                "ics": group.get("ics", []),
                "rejected_indices": group.get("rejected_indices", []),
                "n_rejected": group.get("n_rejected", 0),
            }

            # Panggil plot_multi_ica lama untuk isi sub-tab.
            # Karena group_data tidak punya 'ics_split', tidak akan rekursif.
            self.plot_multi_ica(group_data, frame)
    def plot_multi_ica(self, data, parent):
        """Plot SEMUA ICA components dengan scroll + threshold yang bisa diatur."""

        # ============================================================
        # Jika hasil dari eeg_processor sudah split MAICA,
        # tampilkan menjadi sub-tab Physical dan GT Virtual.
        # ============================================================
        if isinstance(data, dict) and isinstance(data.get("ics_split"), dict):
            self.plot_multi_ica_split(data, parent)
            return

        for widget in parent.winfo_children():
            widget.destroy()

        ics = data['ics']
        n_ic = len(ics)

        # ── Frame kontrol threshold (di atas, tidak ikut scroll) ─────
        ctrl_frame = ttk.Frame(parent)
        ctrl_frame.pack(fill=tk.X, padx=8, pady=4)

        tk.Label(ctrl_frame, text="Kurtosis ≥",
                 font=('Segoe UI', 9), bg=self.colors["bg"]).pack(side=tk.LEFT, padx=(4, 2))
        tk.Spinbox(ctrl_frame, from_=0.0, to=50.0, increment=0.5,
                   textvariable=self.ica_kurt_var,
                   width=5, font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=2)


        tk.Label(ctrl_frame, text="  BCG ≥",
                font=('Segoe UI', 9), bg=self.colors["bg"]).pack(side=tk.LEFT, padx=(8, 2))
        tk.Spinbox(ctrl_frame, from_=0.01, to=1.0, increment=0.01,
                textvariable=self.ica_bcg_var,
                width=5, font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=2)

        tk.Label(ctrl_frame, text="  VBR ≥",
                font=('Segoe UI', 9), bg=self.colors["bg"]).pack(side=tk.LEFT, padx=(8, 2))
        tk.Spinbox(ctrl_frame, from_=1.0, to=20.0, increment=0.5,
                textvariable=self.ica_vbr_var,
                width=5, font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=2)

        # Jika sedang mode split, tombol Apply dari sub-tab tetap harus me-refresh
        # parent utama ICA tab, bukan frame kecil di dalam sub-tab.
        target_ica_parent = self.ica_tab if (
            hasattr(self, "analysis_results") and
            isinstance(self.analysis_results, dict) and
            isinstance(self.analysis_results.get("ica", {}).get("ics_split"), dict)
        ) else parent

        tk.Button(
            ctrl_frame,
            text="Apply & Recompute Features",
            command=lambda: self.apply_ica_threshold(target_ica_parent),
            bg=self.colors["accent"],
            fg="white",
            font=('Segoe UI', 9, 'bold'),
            relief=tk.FLAT,
            padx=10
        ).pack(side=tk.LEFT, padx=10)
        rejected_set = set(data.get('rejected_indices') or [])
        n_rejected = len(rejected_set)

        tk.Label(
            ctrl_frame,
            text=f"Total IC: {n_ic}  |  Rejected: {n_rejected}",
            font=('Segoe UI', 8),
            fg=self.colors["text_dim"],
            bg=self.colors["bg"]
        ).pack(side=tk.LEFT, padx=5)
               

        # ── Container scroll ─────────────────────────────────────────
        scroll_container = ttk.Frame(parent)
        scroll_container.pack(fill=tk.BOTH, expand=True)

        # Scrollbar vertikal dan horizontal
        vbar = tk.Scrollbar(scroll_container, orient=tk.VERTICAL)
        hbar = tk.Scrollbar(scroll_container, orient=tk.HORIZONTAL)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)

        canvas_scroll = tk.Canvas(scroll_container,
                                  yscrollcommand=vbar.set,
                                  xscrollcommand=hbar.set,
                                  bg=self.colors["bg"],
                                  highlightthickness=0)
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vbar.config(command=canvas_scroll.yview)
        hbar.config(command=canvas_scroll.xview)

        inner_frame = ttk.Frame(canvas_scroll)
        canvas_scroll.create_window((0, 0), window=inner_frame, anchor='nw')

        def _on_frame_configure(event):
            canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        inner_frame.bind("<Configure>", _on_frame_configure)

        # Scroll vertikal dengan mouse wheel, horizontal dengan Shift+wheel
        def _on_mousewheel(event):
            canvas_scroll.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _on_mousewheel_h(event):
            canvas_scroll.xview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas_scroll.bind_all("<MouseWheel>", _on_mousewheel)
        canvas_scroll.bind_all("<Shift-MouseWheel>", _on_mousewheel_h)

        # ── Baca threshold saat ini ───────────────────────────────────
        k_thresh = self.ica_kurt_var.get()

        # ── Plot SEMUA IC ─────────────────────────────────────────────
        n_rows = int(np.ceil(n_ic / 2))
        fig, axs = plt.subplots(
            n_rows, 4,
            figsize=(40, 2.0 * n_rows),   # sangat lebar → scroll horizontal aktif
            squeeze=False,
            gridspec_kw={'width_ratios': [5, 1, 5, 1]}
        )
        fig.patch.set_facecolor("#ffffff")

        

        for i in range(n_ic):
            row = i % n_rows
            col_offset = 0 if i < n_rows else 2
            ic = ics[i]
            # z = ic.get('z_score', 0.0)
            corr_bcg = ic.get('corr_pulse', 0.0)

            # Gunakan rejected_indices sebagai ground truth jika tersedia
            # Ground truth reject dari processor
            is_rejected = i in rejected_set
            color_sig = '#e53935' if is_rejected else "#174ea6"
            text_col  = '#e53935' if is_rejected else '#2e7d32'
            status    = "✖ REJECTED" if is_rejected else "✔ KEPT"

            # Time domain
            ax_t = axs[row, col_offset]
            ax_t.plot(ic['sig'], lw=0.6, color=color_sig)
            ax_t.set_facecolor("#ffffff")

            vbr_val = ic.get('vbr', 0.0)
            info = (f"IC {i+1}  {status}\n"
                    f"K:{ic['kurt']:.2f} BCG:{corr_bcg:.3f} VBR:{vbr_val:.1f}")

            # info = (f"IC {i+1}  {status}\n"
            #         f"K: {ic['kurt']:.2f}   BCG: {corr_bcg:.3f}") 
            ax_t.set_ylabel(info, color=text_col, fontsize=7,
                            rotation=0, labelpad=36, va='center',
                            multialignment='left')

            # Frequency domain
            ax_f = axs[row, col_offset + 1]
            ax_f.plot(ic['freqs'],
                      10 * np.log10(ic['psd'] + 1e-12),
                      color='indianred', lw=0.8)
            ax_f.set_facecolor("#ffffff")
            ax_f.set_xlim(0, 80)

            for ax in [ax_t, ax_f]:
                ax.grid(True, linestyle=':', color='#cccccc', alpha=0.5)
                ax.tick_params(axis='both', colors='#555555', labelsize=7, pad=1)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#dddddd')

        # Sembunyikan axes yang tidak terpakai (jika n_ic ganjil)
        if n_ic % 2 != 0:
            for c in range(2, 4):
                axs[n_rows - 1, c].axis('off')

        plt.tight_layout(pad=0.3, w_pad=0.1, h_pad=0.3,
                         rect=[0.02, 0.01, 0.99, 0.99])

        # Embed canvas ke dalam inner_frame (bukan parent langsung)
        # SESUDAH:
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

        fig_canvas = FigureCanvasTkAgg(fig, master=inner_frame)
        fig_canvas.draw()

        # Toolbar diletakkan di LUAR inner_frame (scroll), menempel di bawah ctrl_frame
        # agar tidak ikut terbawa scroll dan selalu terlihat
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        toolbar = NavigationToolbar2Tk(fig_canvas, toolbar_frame)
        toolbar.update()

        fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    
    def plot_ica_compare(self, data, parent):
        """Perbandingan sinyal per channel sebelum dan sesudah ICA, overlay transparan."""
        for widget in parent.winfo_children():
            widget.destroy()
        ctrl_frame = ttk.Frame(parent)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=3)

        ttk.Checkbutton(
            ctrl_frame,
            text="Show Before ICA",
            variable=self.show_before_ica_var,
            command=lambda: self.plot_ica_compare(data, parent)
        ).pack(side=tk.LEFT)
        ctrl_frame = ttk.Frame(parent)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=3)

        ttk.Checkbutton(
            ctrl_frame,
            text="Show legend",
            variable=self.show_ica_compare_legend_var,
            command=lambda: self.plot_ica_compare(data, parent)
        ).pack(side=tk.LEFT)
        before = data.get('before')   # shape: (n_ch, n_samp)
        after  = data.get('after')    # shape: (n_ch, n_samp)

        if before is None or after is None:
            ttk.Label(parent, text="Data before/after ICA tidak tersedia.").pack()
            return

        n_ch   = before.shape[0]
        n_samp = min(before.shape[1], after.shape[1])
        t      = np.arange(n_samp) / FS

        MAX_PTS = 6000
        if n_samp > MAX_PTS:
            step   = n_samp // MAX_PTS
            t      = t[::step]
            before = before[:, ::step]
            after  = after[:,  ::step]

        ch_labels = ['Fp1','Fp2','F3','F4','T3','T4','P3','P4']

        fig, axs = plt.subplots(n_ch, 1, figsize=(14, 2.2 * n_ch), sharex=True)
        fig.patch.set_facecolor('#ffffff')

        if n_ch == 1:
            axs = [axs]

        for i in range(n_ch):
            ax = axs[i]
            sig_before = before[i] - np.mean(before[i])
            sig_after  = after[i]  - np.mean(after[i])

            show_before = self.show_before_ica_var.get()

            if show_before:
                ax.plot(t, sig_before, color='#e53935', lw=0.8, alpha=0.35, label='Before ICA')

            ax.plot(t, sig_after, color='#1a73e8', lw=0.9, alpha=0.90, label='After ICA')
            y_max = max(np.max(np.abs(sig_after)) * 1.3, 10.0)
            ax.set_ylim(-y_max, y_max)

            ch_name = ch_labels[i] if i < len(ch_labels) else f"CH{i+1}"
            ax.set_ylabel(ch_name, fontsize=9, fontweight='bold',
                          color='#1a73e8', rotation=0, labelpad=30, va='center')
            ax.set_facecolor('#ffffff')
            ax.grid(True, linestyle=':', color='#e0e0e0', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=7, colors='#555555')

            if i == 0:
                handles = []

                if show_before:
                    handles.append(
                        plt.Line2D([0], [0], color='#e53935', alpha=0.35, lw=1.5, label='Before ICA')
                    )

                handles.append(
                    plt.Line2D([0], [0], color='#1a73e8', alpha=0.90, lw=1.5, label='After ICA')
                )

                ax.legend(loc='upper right', fontsize=8, framealpha=0.7, handles=handles)

        axs[-1].set_xlabel("Time (s)", fontsize=9)
        fig.suptitle("ICA Artifact Removal — Before vs After (per Channel)",
                     fontsize=11, fontweight='bold', color='#202124', y=1.0)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        self._embed_canvas(fig, parent)
        
    def apply_ica_threshold(self, ica_parent):
        if not hasattr(self, 'analysis_results') or self.analysis_results is None:
            messagebox.showwarning("Belum Ada Data",
                                "Jalankan Run Analysis terlebih dahulu.")
            return

        intermediates = self.analysis_results.get('ica_intermediates')
        if intermediates is None:
            messagebox.showwarning(
                "Data Tidak Lengkap",
                "Intermediate ICA tidak tersedia.\n"
                "Pastikan menggunakan eeg_processor_8.py versi terbaru.")
            return

        # Ambil nilai threshold terbaru dari GUI slider Anda
        try:
            k_thresh   = float(self.ica_kurt_var.get())
            bcg_thresh = float(self.ica_bcg_var.get())
            vbr_thresh = float(self.ica_vbr_var.get())
        except Exception:
            k_thresh = 1.6

        print(f"[DEBUG GUI] Threshold terbaca dari GUI -> Kurtosis: {k_thresh}")

        self.status_label.config(
            text=f"STATUS: RECOMPUTING (K≥{k_thresh:.1f} | BCG≥{bcg_thresh:.2f} | VBR≥{vbr_thresh:.1f})...",
            fg="#f29900"
        )
        self.root.update_idletasks()

        # Fungsi pekerja yang akan berjalan di dalam Thread baru
        def _do_recompute():
            try:
                # Kirim threshold terbaru ke eeg_processor_8
                # recon = reconstruct_from_ica(intermediates, k_thresh)
                recon = reconstruct_from_ica(intermediates, k_thresh, bcg_thresh, vbr_thresh)
                
                print(f"[RECOMPUTE SUCCESS] k_thresh={k_thresh}")
                print(f"[RECOMPUTE SUCCESS] n_rejected={recon.get('n_rejected', 'N/A')}")
# ============================================================
# TAMBAHAN: HITUNG ULANG LZC SETELAH RECOMPUTE ICA
                # ============================================================
                try:
                    if recon.get("ica_clean") is not None:
                        recon["bands"]["lzc_maps"] = self.compute_lzc_maps_from_eeg(
                            recon["ica_clean"],
                            fs=FS
                        )

                        print("[LZC TOPO] lzc_maps berhasil dihitung ulang setelah recompute ICA.")

                except Exception as e:
                    print(f"[LZC TOPO] Gagal menghitung ulang LZC saat recompute: {e}")
                # ============================================================
                # Trim juga IC hasil recompute
                # ============================================================
                if recon.get("ics_updated") is not None:
                    recon["ics_updated"] = trim_ic_components(recon["ics_updated"])

                if isinstance(recon.get("ics_split"), dict):
                    for group_key, group_val in recon["ics_split"].items():
                        if isinstance(group_val, dict) and group_val.get("ics") is not None:
                            group_val["ics"] = trim_ic_components(group_val["ics"])
                # Menimpa hasil analisis lama dengan rekonstruksi threshold baru
                self.analysis_results['ica']['data']             = recon['ica_clean']
                self.analysis_results['ica']['after']            = recon['ica_clean']
                self.analysis_results['ica']['ics_updated']      = recon['ics_updated']

                
                self.analysis_results['ica']['ics'] = recon['ics_updated']

                # ============================================================
                # Tambahan untuk split MAICA
                # ============================================================
                if recon.get("ics_split") is not None:
                    self.analysis_results["ica"]["ics_split"] = recon["ics_split"]

                if recon.get("split_stats") is not None:
                    self.analysis_results["ica"]["split_stats"] = recon["split_stats"]

                self.analysis_results['ica']['n_rejected'] = recon['n_rejected']
                self.analysis_results['ica']['rejected_indices'] = recon.get('rejected_indices', [])
                self.analysis_results['bands']                   = recon['bands']
                self.analysis_results['features']                = recon['features']
                self.analysis_results['mif']                     = recon['mif']

                if self.last_features is not None:
                    label_keys = {'filename', 'subject', 'rec_id',
                                'stress', 'anxiety', 'depression',
                                'gender', 'seg_id'}
                    label_part = {k: v for k, v in self.last_features.items()
                                if k in label_keys}
                    self.last_features = {**label_part, **recon['features']['dict']}
                else:
                    import os
                    self.last_features = {
                        'filename': os.path.basename(
                            getattr(self, 'last_filename', 'unknown')),
                        **recon['features']['dict']
                    }

                n_rej = recon['n_rejected']

                # Mengirim perintah update grafik ke Main Thread Tkinter
                def _update_ui():
                    self.plot_multi_ica(self.analysis_results['ica'], ica_parent)
                    self.plot_ica_compare(self.analysis_results['ica'], self.ica_compare_tab)
                    self.plot_multi_mif(self.analysis_results['mif'], self.mif_tab)
                    self.plot_band_power(self.analysis_results['bands'], self.band_tab)
                    self.plot_brain_mapping(self.analysis_results['bands'], self.topo_tab)
                    self.display_feature_table(
                        self.analysis_results['features']['table'], self.feat_tab)
                    self.build_save_tab(self.save_tab)
                    split_stats = self.analysis_results.get("ica", {}).get("split_stats")

                    if isinstance(split_stats, dict):
                        status_text = (
                            f"STATUS: RECOMPUTE DONE — "
                            f"K≥{k_thresh:.1f} | "
                            f"Physical {split_stats.get('physical_rejected', '-')}/"
                            f"{split_stats.get('physical_total', '-')} | "
                            f"GT Virtual {split_stats.get('virtual_gt_rejected', '-')}/"
                            f"{split_stats.get('virtual_gt_total', '-')}"
                        )
                    else:
                        status_text = (
                            f"STATUS: RECOMPUTE DONE — "
                            f"K≥{k_thresh:.1f} | "
                            f"{n_rej} IC ditolak"
                        )

                    self.status_label.config(
                        text=status_text,
                        fg="green"
                    )

                self.root.after(0, _update_ui)

            except Exception as e:
                import traceback
                err_msg = traceback.format_exc()
                print(f"[ERROR] apply_ica_threshold:\n{err_msg}")
                self.root.after(0, lambda: messagebox.showerror(
                    "Recompute Error", f"Gagal rekonstruksi ulang:\n{str(e)}"))
                self.root.after(0, lambda: self.status_label.config(
                    text="STATUS: RECOMPUTE ERROR", fg="red"))

        # Jalankan proses recompute di latar belakang agar GUI tidak hang/freeze
        threading.Thread(target=_do_recompute, daemon=True).start()
            
    def plot_multi_mif(self, data, parent):
            """Plot MIF dengan Time Series yang diperlebar (Rasio 3:1)."""
            for widget in parent.winfo_children(): widget.destroy()
            mimfs = data['mimfs']
            n_mimfs = len(mimfs)
            n_rows = int(np.ceil(n_mimfs / 2))
            
            # Gunakan width_ratios untuk memperlebar kolom 0 dan 2
            fig, axs = plt.subplots(n_rows, 4, figsize=(18, 1.8 * n_rows), squeeze=False,
                                    gridspec_kw={'width_ratios': [3, 1, 3, 1]})
            fig.patch.set_facecolor("#fffefe")

            for i in range(n_mimfs):
                row = i % n_rows
                col_offset = 0 if i < n_rows else 2
                m = mimfs[i]
                
                # --- Kolom Time Domain (WIDE) ---
                ax_t = axs[row, col_offset]
                ax_t.plot(m['sig'], lw=0.8, color='#174ea6')
                ax_t.set_facecolor("#FFFFFF")
                band_name = m.get('name', f"Band {i+1}")
                ax_t.set_ylabel(band_name, color='#174ea6', fontsize=8, rotation=0, labelpad=20, va='center')

                # --- Kolom Freq Domain (COMPACT) ---
                ax_f = axs[row, col_offset + 1]
                ax_f.fill_between(m['f'], m['p'], color='green', alpha=0.4)
                ax_f.set_facecolor("#FFFDFD")
                ax_f.set_xlim(0, 80)

                for ax in [ax_t, ax_f]:
                    ax.grid(True, linestyle=':', color='#333333', alpha=0.3)
                    ax.tick_params(axis='both', colors='#555555', labelsize=7, pad=1)
                    for spine in ax.spines.values(): spine.set_edgecolor('#333333')

            if n_mimfs < (n_rows * 2):
                for c in range(2, 4): axs[n_rows-1, c].axis('off')

            plt.tight_layout(pad=0.1, w_pad=0.05, h_pad=0.1, rect=[0.02, 0.02, 0.99, 0.98])
            self._embed_canvas(fig, parent)
            
    def plot_band_power(self, data, parent):
        """Plot distribusi power band dengan layout 2 kolom."""
        # Bersihkan widget sebelumnya
        for widget in parent.winfo_children(): 
            widget.destroy()

        powers, signals = data['powers'], data['signals']
        ranges, imf_map = data['ranges'], data['imf_map']
        band_names = list(powers.keys())
        num_bands = len(band_names)
        
        # Hitung jumlah baris yang dibutuhkan untuk 2 kolom (setelah bar chart)
        # math.ceil(num_bands / 2)
        rows_needed = (num_bands + 1) // 2 
        
        fig = plt.figure(figsize=(14, 12))
        # Bar chart di baris 0, sinyal mulai dari baris 1
        gs = fig.add_gridspec(rows_needed + 1, 2)
        
        # --- 1. Bar Chart (Spans both columns at the top) ---
        ax_bar = fig.add_subplot(gs[0, :]) 
        ax_bar.bar(band_names, list(powers.values()), color='skyblue', edgecolor='navy')
        ax_bar.set_yscale('log')
        ax_bar.set_title("Relative Power Distribution (Butterworth Band Extraction)", fontsize=12, pad=15)
        ax_bar.set_ylabel("Power (Log)")

        # --- 2. Sinyal Tiap Band (2-Column Layout) ---
        for i, band in enumerate(band_names):
            # Hitung posisi grid: 
            # i // 2 menentukan baris (ditambah 1 karena baris 0 adalah bar chart)
            # i % 2 menentukan kolom (0 atau 1)
            row_idx = (i // 2) + 1
            col_idx = i % 2
            
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            # Ambil rata-rata sinyal antar channel
            mean_sig = np.mean(signals[i], axis=0) 
            n_points = len(mean_sig)
            t_plot = np.linspace(0, n_points / FS, n_points) 
            
            ax.plot(t_plot, mean_sig, lw=0.6, color='#174ea6')
            
            # Metadata Formatting
            hz_range = ranges[band]
            hz_text = f"{hz_range[0]:.1f}-{hz_range[1]:.1f} Hz" if hz_range[0] is not None else "N/A"
            hz_range = ranges[band]
            imf_text = f"{hz_range[0]:.1f}–{hz_range[1]:.1f} Hz" if hz_range[0] is not None else "N/A"
                
            # Styling Subplot
            ax.set_title(f"Band: {band} ({hz_text})", fontsize=10, loc='left')
            ax.text(0.01, 0.05, f"Composition: {imf_text}", 
                    transform=ax.transAxes, fontsize=8, color='darkred', alpha=0.7)
            
            ax.set_ylabel("Amp")
            ax.set_xlabel("Time (s)")
            ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        self._embed_canvas(fig, parent)
        plt.close(fig)
    

    def _embed_canvas(self, fig, parent):
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        
if __name__ == "__main__":
    root = tk.Tk()
    app = EEGAnalysisGUI(root)
    root.mainloop()