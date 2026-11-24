# Logbook Eksperimen — Segmentasi Instance pada Lingkungan Indoor Nyata

- **Mata Kuliah:** Visi Komputer — Evaluasi CPMK 3 (ITS)
- **Variabel Wajib (Digit Terakhir NRP = 3):** Jumlah data latih bertingkat **25% / 50% / 100%** (99 / 198 / 395 gambar, 25% ⊂ 50% ⊂ 100%, `seed = 42`)
- **Dataset Latih:** Panoptic Indoor Segmentation v5 (Roboflow, `panopticindoor/panoptic-indoor-segmentation`), 74 kelas dipakai (9 kelas tidak memiliki contoh di train)
- **Model:** YOLO11m-seg (one-stage CNN), Mask R-CNN R50-FPN v2 (two-stage CNN), Mask2Former Swin-T (transformer)
- **Lingkungan Eksekusi:** Kaggle Notebooks, GPU **Tesla T4 ×2** (2 jalur proses terpisah: GPU0 untuk 100%, GPU1 untuk 25% lalu 50%)

---

## 1. Hipotesis Bertanggal

| ID | Tanggal | Hipotesis (sebelum hasil akhir) | Hasil Aktual & Status |
|---|---|---|---|
| **H1** | 2026-11-15 | Penambahan data latih (25% → 50% → 100%) akan menaikkan Mask AP ketiga model; model transformer (Mask2Former) diperkirakan tetap unggul pada data kecil berkat representasi global dan pretraining kuat, sedangkan CNN (YOLO11m-seg, Mask R-CNN) mendapat kenaikan relatif paling besar saat data ditambah. | **Terbukti.** Pada `valid` 25%→100%: YOLO +59,5% (0,170→0,271), Mask R-CNN +41,9% (0,159→0,226), Mask2Former +22,7% (0,271→0,332) namun tetap tertinggi di semua fraksi. Pola sama pada `test_own`: YOLO +92,5% (0,109→0,210), Mask R-CNN +67,7% (0,127→0,214), Mask2Former +16,6% (0,259→0,302). |
| **H2** | 2026-11-15 | Akan terjadi penurunan AP (*domain shift*) dari `valid` ke `test` Roboflow dan `test_own` (foto HP sendiri), dengan penurunan paling tajam pada detektor one-stage (YOLO11m-seg) dibanding two-stage dan transformer. | **Terbukti sebagian (bernuansa).** Penurunan ke `test` tajam pada semua model (sisa AP `test/valid`: YOLO 38,0%, Mask R-CNN 52,4%, Mask2Former 55,9%) karena `test` memiliki gambar >2 MP dan 22,5% instance kelas langka. Ke `test_own`, sisa AP justru lebih tinggi (YOLO 77,5%, Mask2Former 90,8%, Mask R-CNN 94,6%), tetapi YOLO tetap mengalami penurunan relatif terbesar dibanding dua model lain. |
| **H3** | 2026-11-24 | Pada `test_own`, foto kondisi sulit (`oklusi`, `redup`, `silau`, `backlight`) dan kelas tanpa contoh di train (`mouse`, `toothbrush`, `scissors`) akan menurunkan F1 secara drastis; perbaikan praproses (crop/zoom, naikkan resolusi, pencerahan) dapat memulihkan sebagian kegagalan skala/cahaya, tetapi **tidak** bisa memulihkan kelas *unseen*. | **Terbukti.** `oklusi` menjadi kondisi tersulit (AP: YOLO 0,082, Mask R-CNN 0,155, Mask2Former 0,269). Pada `normal_14`, crop/zoom menaikkan deteksi `spoon` (YOLO F1 0,000→0,222; Mask2Former 0,182→0,333) tetapi `toothbrush` tetap 0 di semua model. Pada `redup_07`, pencerahan gamma+CLAHE memulihkan CNN dari F1 0,000→0,400 (YOLO) dan 0,000→0,250 (Mask R-CNN). |

---

## 2. Catatan Proses Kronologis

### Tahap 1 — Persiapan Data & Verifikasi Format (2026-11-15)
- Membuat **Notebook 0** (`00_data_prep.ipynb`) untuk mengunduh dataset Roboflow, membersihkan anotasi, membuat subset bertingkat 25%/50%/100% (`seed=42`), dan mengonversi format COCO ke YOLO-seg.
- **Temuan & keputusan:**
  1. Ditemukan 1 gambar `test` dengan 111 instance `book`, sehingga evaluasi GT-vs-GT dengan batas standar COCO `maxDets=100` menghasilkan 0,9994 (bukan 1,0). Verifikasi format di NB 0 menggunakan `maxDets=10000`, sementara evaluasi model tetap memakai standar COCO `maxDets=100`.
  2. Ditemukan 4 gambar `train`/`valid` berukuran fisik 480×640 sedangkan JSON mencatat 640×480 **tanpa tag rotasi EXIF**. Keempat gambar dibuang agar tidak merusak mask training.
  3. Disusun modul bersama `seg_common.py` (identik di NB 1–3) agar ketiga model dievaluasi dengan `pycocotools` yang persis sama (`segm`, area dari mask, `maxDets=100`, `score >= 0,001`).

### Tahap 2 — Pilot & Training Penuh 3 Model × 3 Fraksi (2026-11-15)
- Menjalankan pilot 2 epoch (target durasi ~2 jam untuk run 100%) untuk menentukan jumlah epoch final:
  - **YOLO11m-seg:** 150 epoch (pilot ~0,7 mnt/epoch).
  - **Mask R-CNN R50-FPN v2:** 70 epoch (pilot 3,3 mnt / 2 epoch).
  - **Mask2Former Swin-T:** 60 epoch (pilot 4,4 mnt / 2 epoch).
- Menjalankan training penuh 9 run (3 model × 3 fraksi) pada 15 Nov 2026 (07:51–12:22 UTC).
- **Kendala:** Tahap prediksi `test` pada Mask2Former gagal (`returncode=1`, CUDA OOM) karena split `test` memiliki gambar hingga 10,3 MP; upsample 100 query mask sekaligus membutuhkan ±1,3 GB/MP.

### Tahap 3 — Perbaikan OOM Mask2Former & Evaluasi Awal (2026-11-17)
- Memperbaiki `instance_inference` di **Notebook 3** agar melakukan upsample mask per potongan `chunk=10` query lalu menyalin mask biner ke CPU (hasil matematis identik, VRAM turun 10×).
- Menjalankan ulang prediksi `valid` dan `test` Mask2Former (17 Nov 2026, 23:10–23:16 UTC) dengan `RUN_TRAINING = False` → sukses tanpa error.
- Membuat **Notebook 4** (`04_perbandingan.ipynb`) untuk perbandingan awal pada `valid` dan `test`.

### Tahap 4 — Pengambilan Foto HP, Anotasi Roboflow, & Evaluasi `test_own` (2026-11-24)
- Mengambil 40 foto lingkungan indoor nyata dengan kamera HP sendiri: `normal` (19), `redup` (11), `oklusi` (5), `backlight` (3), `silau` (1), `sudut` (1).
- Mengekstrak metadata EXIF (tanpa lokasi GPS) ke `test own/exif_test_own.csv`; foto asli 12 MP diabaikan Git (`.gitignore`) demi privasi.
- Menganotasi 40 foto di Roboflow (`sultan-hamdi-jailani/cv_instance_seg` v1): **276 poligon, 35 kelas** (semua termasuk 74 kelas dataset; rata-rata 6,9 anotasi/foto; 3 kelas tanpa contoh di train: `mouse`, `toothbrush`, `scissors`). Dokumentasi di `test own/daftar_objek.md`.
- Memperbarui konfigurasi NB 0 untuk mengunduh `test_own` dan menjalankan ulang NB 0 → NB 1–3 (`RUN_TRAINING = False`, prediksi saja) → NB 4.
- **Kendala kecil:** Save & Run All sempat antre (*queued*) di Kaggle; selain itu run pertama NB 4 masih memakai daftar `CONDITIONS` lama sehingga `redup`/`backlight`/`sudut` masuk kelompok `lainnya`.

### Tahap 5 — Perbaikan NB 4 & Pembuatan NB 5 Demo + 5 Studi Kasus (2026-11-25)
- Menjalankan ulang NB 4 dengan versi `CONDITIONS` terbaru (`hasil/4/cv-is-perbandingan-final-2.ipynb`, `hasil/4/results-2/`) → keenam kondisi `test_own` terpisah rapi, 0 error.
- Membuat **Notebook 5** (`05_demo_dan_studi_kasus.ipynb`) berisi demo langsung 10 menit (inferensi foto baru, ubah threshold, AP satu kelas, penjelasan baris kode) serta 5 eksperimen pembuktian kegagalan (`oklusi_03`, `normal_14`, `oklusi_04`, `redup_07`, `silau_01`).
- **Kendala kecil:** Error `KeyError: 'id'` di sel konfigurasi NB 5 karena `dataset_info.json` memakai kunci `"coco_id"`. Diperbaiki → run penuh di Kaggle (`hasil/5/cv-is-demo.ipynb`) selesai bersih (0 error).

---

## 3. Rekap Waktu GPU (Kaggle Tesla T4 ×2)

Sumber: log `gpu_time_log.csv` pada output `hasil/1`–`hasil/3` dan `hasil/4/results-2/biaya_vs_hasil.csv`.

| Model | Pilot | Train 25% | Train 50% | Train 100% | Subtotal Train | Predict (valid+test) | Predict Ulang (+`test_own`) | Total Jam-GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **YOLO11m-seg** (150 ep) | ~0,02 jam | 0,42 jam | 0,60 jam | 0,91 jam | **1,93 jam** | 0,12 jam | 0,13 jam | **~2,20 jam** |
| **Mask R-CNN R50-FPN v2** (70 ep) | 0,05 jam | 0,59 jam | 0,80 jam | 1,25 jam | **2,65 jam** | 0,14 jam | 0,17 jam | **~3,01 jam** |
| **Mask2Former Swin-T** (60 ep) | 0,07 jam | 0,71 jam | 1,10 jam | 1,84 jam | **3,65 jam** | 0,12* + 0,16 jam | 0,16 jam | **~4,16 jam** |
| **Total** | **0,14 jam** | **1,72 jam** | **2,50 jam** | **4,00 jam** | **8,23 jam** | **0,54 jam** | **0,46 jam** | **~9,37 jam** |

*\*0,12 jam pada 15 Nov gagal OOM sebelum diperbaiki menjadi 0,16 jam pada 17 Nov dengan inferensi per potongan.*
*Notebook 0 dan Notebook 4 dijalankan di CPU; Notebook 5 dijalankan di GPU T4 (~3 menit).*

---

## 4. Riwayat Commit Bertahap

| Hash | Waktu | Ringkasan Commit |
|---|---|---|
| `ecbb4a5` | 2026-11-24 15:11 | Inisialisasi repo: README dan `.gitignore` |
| `8288d59` | 2026-11-24 15:12 | Notebook 0: persiapan data, subset 25/50/100%, konversi COCO→YOLO |
| `b8132c0` | 2026-11-24 15:12 | Notebook 1: YOLO11m-seg, 3 fraksi data, 150 epoch |
| `8298b08` | 2026-11-24 15:12 | Notebook 2: Mask R-CNN R50-FPN v2, 3 fraksi data, 70 epoch |
| `622a8db` | 2026-11-24 15:12 | Notebook 3: Mask2Former Swin-T, 3 fraksi data, 60 epoch |
| `9283c2c` | 2026-11-24 15:12 | Notebook 3: prediksi ulang valid/test dengan inferensi per potongan |
| `649e527` | 2026-11-24 15:12 | Notebook 4: perbandingan model, domain shift, analisis kegagalan (tanpa `test_own`) |
| `6347776` | 2026-11-24 17:32 | `test_own`: 40 foto HP, nama per kondisi, tabel EXIF |
| `8ba8ec7` | 2026-11-24 20:10 | `test_own`: anotasi Roboflow (40 foto, 276 poligon, 35 kelas) dan daftar objek |
| `3002907` | 2026-11-24 20:39 | Konfigurasi `test_own` (Roboflow v1) dan mode prediksi saja untuk Notebook 1–3 |
| `a1f33be` | 2026-11-24 22:11 | Update `.gitignore` |
| *(berikutnya)* | 2026-11-25 | Hasil final NB 0–4 (`test_own`), NB 5 demo & 5 studi kasus, logbook, deklarasi AI |
