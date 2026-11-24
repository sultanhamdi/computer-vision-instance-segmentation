# Segmentasi Instance pada Lingkungan Indoor Nyata

Tugas Evaluasi CPMK 3, Visi Komputer ITS. Perbandingan tiga model segmentasi instance dari dua keluarga arsitektur
pada dataset Panoptic Indoor Segmentation v5 (Roboflow), dengan variabel jumlah data latih 25% / 50% / 100%
(digit terakhir NRP = 3), ditambah evaluasi pada test set foto HP sendiri (`test_own`).

| Model | Keluarga | Notebook |
|---|---|---|
| YOLO11m-seg | one-stage, CNN | `notebooks/01_yolo11_seg.ipynb` |
| Mask R-CNN R50-FPN v2 | two-stage, CNN | `notebooks/02_mask_rcnn.ipynb` |
| Mask2Former Swin-T | transformer | `notebooks/03_mask2former.ipynb` |

## Struktur

```
notebooks/   kode (dijalankan di Kaggle, GPU T4 x2)
  00_data_prep.ipynb     unduh dataset, bersihkan anotasi, subset 25/50/100%, konversi COCO -> YOLO, test_own
  01–03_*.ipynb          training 3 fraksi per model + evaluasi pycocotools yang sama untuk semua model
  04_perbandingan.ipynb  perbandingan, domain shift, analisis kegagalan (CPU)
hasil/<n>/   salinan notebook hasil run Kaggle (dengan output) sebagai bukti proses dan log
```

## Urutan menjalankan

1. Notebook 0 → menghasilkan `indoor_seg_data.zip`.
2. Notebook 1–3 (input: output Notebook 0) → pilot dulu, lalu training penuh.
3. Notebook 4 (input: output Notebook 0–3).
4. Setelah test_own siap: Notebook 0 lagi → Notebook 1–3 dengan `RUN_TRAINING = False` → Notebook 4.
