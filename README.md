# EEG Processing & Analysis

Aplikasi GUI berbasis Python untuk akuisisi, pemrosesan, dan analisis sinyal EEG
(menggunakan BrainFlow, MNE, SciPy, dan scikit-learn).

## Struktur File

| File | Fungsi |
|---|---|
| `GUI.py` | Program utama (antarmuka Tkinter) |
| `eeg_processor_8.py` | Modul pemrosesan sinyal EEG (filter, ICA, SNR, dll) |
| `olah_data.py` | Ekstraksi fitur EEG & PPG |
| `requirements.txt` | Daftar library yang dibutuhkan |

## Persyaratan

- Python 3.9 - 3.11
- Tkinter (sudah termasuk pada instalasi Python standar Windows)

## Instalasi

```bash
pip install -r requirements.txt
```

## Menjalankan

```bash
python GUI.py
```

Ketiga file `.py` harus berada dalam **satu folder**.

## Catatan

- Sesuaikan port COM board EEG pada kolom input di GUI.
- Dataset (CSV/txt) tidak disertakan di repository ini.
