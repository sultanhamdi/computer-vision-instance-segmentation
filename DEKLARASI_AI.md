# Deklarasi Penggunaan AI

## 1. Pernyataan Transparansi

Dalam pengerjaan tugas Evaluasi CPMK 3 (Segmentasi Instance pada Lingkungan Indoor Nyata), asisten AI berbasis LLM (**Claude Code / Antigravity**) digunakan sebagai alat bantu pemrograman, debugging, dan penyusunan kerangka analisis. Seluruh keputusan eksperimen, pengambilan dan anotasi dataset uji sendiri (`test_own`), eksekusi di Kaggle, verifikasi angka, serta pertanggungjawaban isi laporan dilakukan secara mandiri oleh pemilik tugas.

### Pembagian Pekerjaan
- **Dikerjakan secara mandiri tanpa AI:**
  - Pengambilan 40 foto indoor nyata menggunakan kamera HP pribadi dengan variasi kondisi (`normal`, `redup`, `oklusi`, `backlight`, `silau`, `sudut`).
  - Anotasi manual 276 poligon (35 kelas) di Roboflow serta ekstraksi bukti EXIF.
  - Eksekusi seluruh notebook di Kaggle (pilot, training 9 model, prediksi ulang, perbandingan, dan demo).
  - Verifikasi visual mask, pemeriksaan konsistensi hasil, dan penarikan kesimpulan akhir.
- **Dibantu AI (dengan verifikasi & revisi):**
  - Penulisan kerangka kode persiapan data (`00_data_prep.ipynb`), modul evaluasi seragam `seg_common.py`, dan skrip training/prediksi 3 arsitektur (`01`–`03`).
  - Pembuatan visualisasi analisis perbandingan, domain shift, dan galeri kegagalan (`04_perbandingan.ipynb`).
  - Pembuatan notebook demo langsung & 5 eksperimen pembuktian studi kasus (`05_demo_dan_studi_kasus.ipynb`).
  - Penyusunan draf logbook dan kerangka laporan berdasarkan angka riil dari output CSV/notebook.

---

## 2. Ringkasan Prompt Utama yang Digunakan

1. **Perancangan pipeline & persiapan data:**
   *"Buat pipeline notebook Kaggle (GPU T4×2) untuk membandingkan YOLO11m-seg, Mask R-CNN R50-FPN v2, dan Mask2Former Swin-T pada dataset Panoptic Indoor Segmentation v5 dengan subset bertingkat 25%/50%/100% dan evaluasi pycocotools yang seragam untuk semua model."*
2. **Debugging & perbaikan memori:**
   *"Prediksi split test pada Mask2Former gagal OOM di GPU T4 karena ada gambar beresolusi ~10 MP. Perbaiki fungsi inferensi agar hemat VRAM tanpa mengubah hasil numerik."*
3. **Evaluasi `test_own` & analisis kegagalan:**
   *"Integrasikan test set foto HP sendiri (`test_own`, 40 foto dari Roboflow) ke Notebook 0–4, evaluasi ulang semua model dengan mode `RUN_TRAINING = False`, dan periksa kondisi sulit per awalan nama file."*
4. **Demo langsung & eksperimen 5 studi kasus:**
   *"Buat Notebook 5 untuk demo 10 menit (inferensi foto baru, ubah threshold, hitung AP satu kelas) dan 5 eksperimen pembuktian kecil pada foto kegagalan `test_own` (crop/zoom, ubah resolusi, pencerahan gamma+CLAHE, redam silau)."*

---

## 3. Kasus Konkret Saran / Kode AI yang Salah (Beserta Perbaikannya)

Sesuai ketentuan tugas, berikut empat kasus nyata di mana saran atau kode AI terbukti salah saat diverifikasi terhadap data/eksekusi di Kaggle:

### Kasus 1: Diagnosis salah "tag rotasi EXIF" pada 4 gambar `train`/`valid`
- **Saran AI yang salah:** Saat ditemukan 4 gambar yang ukuran filenya 480×640 sedangkan dimensi di JSON 640×480, AI langsung mendiagnosis penyebabnya adalah tag rotasi EXIF dan mengusulkan perbaikan dengan `ImageOps.exif_transpose`.
- **Fakta setelah dicek:** Keempat gambar tersebut **tidak memiliki tag EXIF sama sekali**; dimensi di file gambar memang tidak cocok dengan anotasi JSON.
- **Perbaikan:** Keempat gambar bermasalah tersebut dibuang secara eksplisit di Notebook 0 agar tidak merusak poligon saat training.

### Kasus 2: Kode inferensi Mask2Former buatan AI memicu CUDA OOM di GPU T4
- **Kode AI yang salah:** Implementasi awal `instance_inference` di Notebook 3 melakukan `F.interpolate` terhadap seluruh 100 query mask sekaligus ke resolusi asli gambar di GPU. Pada split `test` yang memiliki gambar ±10 MP, operasi tersebut membutuhkan ±1,3 GB VRAM per megapiksel sehingga memicu OOM (`returncode=1` pada run 15 Nov 2026).
- **Perbaikan:** Mengubah `instance_inference` untuk melakukan upsampling per potongan `chunk=10` query lalu menyalin mask biner ke CPU. Pemakaian VRAM turun ~10× dengan hasil AP yang identik secara matematis (run ulang 17 Nov 2026 berhasil, `returncode=0`).

### Kasus 3: Kesimpulan awal AI yang keliru soal "AP `test` datar adalah sifat data"
- **Analisis AI yang salah:** Setelah melihat Mask AP `test` YOLO11m-seg hampir tidak naik saat data ditambah (0,097 → 0,102 → 0,103), AI menyimpulkan bahwa datarnya kurva merupakan sifat inheren split `test` yang tidak bisa diperbaiki oleh penambahan data.
- **Fakta setelah semua model selesai:** Mask R-CNN (`0,086 → 0,102 → 0,118`, naik +37,8%) dan Mask2Former (`0,151 → 0,186 → 0,186`, naik +23,4%) ternyata **naik nyata** di split `test` yang sama. Jadi kesimpulan AI terbantahkan; fenomena datar pada `test` lebih spesifik pada keterbatasan generalisasi one-stage YOLO11m-seg terhadap domain shift `test`.

### Kasus 4: Asumsi nama kunci JSON `"id"` pada Notebook 5 (`KeyError`)
- **Kode AI yang salah:** Saat membuat `05_demo_dan_studi_kasus.ipynb`, AI menuliskan `CAT_NAME = {c["id"]: c["name"] for c in CLASSES}` tanpa memeriksa skema `dataset_info.json` yang dihasilkan Notebook 0.
- **Fakta saat dijalankan di Kaggle:** Sel konfigurasi langsung gagal dengan `KeyError: 'id'` karena `dataset_info.json` menyimpan kunci `"coco_id"` (bukan `"id"`).
- **Perbaikan:** Mengganti `c["id"]` menjadi `c["coco_id"]` di sel 3 Notebook 5; setelah diperbaiki, seluruh 28 sel Notebook 5 berjalan bersih dengan 0 error.
