# Daftar Objek test_own (40 foto)

Dihitung dari anotasi Roboflow (`test own/test/_annotations.coco.json`): **40 foto, 276 anotasi poligon, 35 kelas**.
Semua nama kelas ada di 74 kelas dataset Panoptic Indoor Segmentation v5. Ukuran semua foto 639×480 (Fit within 640).
Tanda **\*** = kelas tanpa satu pun contoh di data train (model hampir pasti gagal).

## Per foto

| Foto | Jumlah | Anotasi |
|---|---|---|
| normal_01 | 4 | backpack 1, handbag 2, wall-other-merged 1 |
| redup_01 | 4 | backpack 1, handbag 2, wall-other-merged 1 |
| normal_02 | 8 | door-stuff 1, floor-other-merged 1, laptop 1, light 1, mouse\* 1, table-merged 1, wall-other-merged 2 |
| redup_02 | 8 | door-stuff 1, floor-other-merged 1, laptop 1, light 1, mouse\* 1, table-merged 1, wall-other-merged 2 |
| normal_03 | 6 | bed 1, chair 1, floor-other-merged 1, pillow 2, wall-other-merged 1 |
| redup_03 | 7 | bed 1, chair 1, floor-other-merged 2, pillow 2, wall-other-merged 1 |
| normal_04 | 6 | door-stuff 2, floor-other-merged 1, rug-merged 1, wall-other-merged 2 |
| redup_04 | 6 | door-stuff 2, floor-other-merged 1, rug-merged 1, wall-other-merged 2 |
| normal_05 | 7 | bottle 3, shelf 1, toothbrush\* 2, wall-tile 1 |
| normal_06 | 13 | bottle 3, cup 2, knife 1, pan 2, plate 1, spoon 2, stove-merged 1, wall-tile 1 |
| normal_07 | 13 | bottle 4, cup 1, knife 1, pan 2, plate 1, shelf 1, spoon 1, stove-merged 1, wall-tile 1 |
| redup_05 | 16 | bottle 4, cup 2, food-other-merged 2, knife 1, pan 2, plate 2, spoon 1, stove-merged 1, wall-tile 1 |
| normal_08 | 8 | box 4, floor-other-merged 1, plug 1, table-merged 1, wall-other-merged 1 |
| redup_06 | 7 | cardboard 4, floor-other-merged 1, table-merged 1, wall-other-merged 1 |
| normal_09 | 6 | door-stuff 1, floor-other-merged 1, wall-other-merged 2, window-other 2 |
| backlight_01 | 6 | door-stuff 1, floor-other-merged 1, wall-other-merged 2, window-other 2 |
| normal_10 | 5 | door-stuff 1, laptop 1, mouse\* 1, table-merged 1, wall-other-merged 1 |
| silau_01 | 7 | door-stuff 1, laptop 1, light 1, mouse\* 1, plug 1, table-merged 1, wall-other-merged 1 |
| normal_11 | 10 | ceiling-merged 1, door-stuff 2, floor-other-merged 1, wall-other-merged 5, window-other 1 |
| backlight_02 | 9 | door-stuff 2, floor-other-merged 1, wall-other-merged 5, window-other 1 |
| oklusi_01 | 4 | fork 1, knife 1, spoon 2 |
| redup_07 | 4 | fork 1, knife 1, spoon 2 |
| oklusi_02 | 5 | cabinet-merged 1, chair 1, floor-other-merged 1, pillow 1, wall-other-merged 1 |
| redup_08 | 4 | cabinet-merged 1, floor-other-merged 1, pillow 1, wall-other-merged 1 |
| normal_12 | 6 | floor-other-merged 1, paper 1, scissors\* 2, table-merged 1, wall-other-merged 1 |
| redup_09 | 7 | floor-other-merged 1, paper 1, plug 1, scissors\* 2, table-merged 1, wall-other-merged 1 |
| normal_13 | 7 | floor-other-merged 1, laptop 1, mouse\* 1, paper 1, plug 1, table-merged 1, wall-other-merged 1 |
| redup_10 | 6 | floor-other-merged 1, laptop 1, mouse\* 1, paper 1, table-merged 1, wall-other-merged 1 |
| normal_14 | 6 | fork 1, spoon 2, toothbrush\* 2, wall-tile 1 |
| normal_15 | 9 | bottle 2, fork 1, spoon 2, toothbrush\* 3, wall-tile 1 |
| normal_16 | 5 | backpack 1, cardboard 2, floor-other-merged 1, wall-other-merged 1 |
| normal_17 | 5 | backpack 1, cardboard 2, floor-other-merged 1, wall-other-merged 1 |
| redup_11 | 5 | backpack 1, cardboard 2, floor-other-merged 1, wall-other-merged 1 |
| normal_18 | 4 | floor-other-merged 1, rug-merged 1, wall-other-merged 1, window-other 1 |
| oklusi_03 | 7 | bottle 2, towel 5 |
| oklusi_04 | 13 | bowl 3, cup 3, floor-other-merged 1, plate 1, spoon 2, wall-other-merged 1, wall-tile 2 |
| backlight_03 | 11 | ceiling-merged 1, door-stuff 1, light 1, shelf 1, wall-other-merged 5, window-other 2 |
| normal_19 | 2 | plug 1, wall-other-merged 1 |
| sudut_01 | 3 | floor-other-merged 1, mouse\* 1, wall-other-merged 1 |
| oklusi_05 | 7 | cardboard 4, cup 1, wall-other-merged 1, window-other 1 |

## Total per kelas

| Kelas | Foto | Instance |
|---|---|---|
| wall-other-merged | 31 | 49 |
| floor-other-merged | 24 | 25 |
| bottle | 6 | 18 |
| door-stuff | 11 | 15 |
| cardboard | 5 | 14 |
| spoon | 8 | 14 |
| table-merged | 10 | 10 |
| window-other | 7 | 10 |
| cup | 5 | 9 |
| wall-tile | 7 | 8 |
| mouse\* | 7 | 7 |
| toothbrush\* | 3 | 7 |
| pillow | 4 | 6 |
| laptop | 6 | 6 |
| pan | 3 | 6 |
| plug | 5 | 5 |
| backpack | 5 | 5 |
| towel | 1 | 5 |
| knife | 5 | 5 |
| plate | 4 | 5 |
| box | 1 | 4 |
| light | 4 | 4 |
| fork | 4 | 4 |
| handbag | 2 | 4 |
| paper | 4 | 4 |
| scissors\* | 2 | 4 |
| rug-merged | 3 | 3 |
| chair | 3 | 3 |
| shelf | 3 | 3 |
| stove-merged | 3 | 3 |
| bowl | 1 | 3 |
| bed | 2 | 2 |
| ceiling-merged | 2 | 2 |
| food-other-merged | 1 | 2 |
| cabinet-merged | 2 | 2 |
| **Total (35 kelas)** | **40** | **276** |

Rata-rata 6,9 anotasi per foto (data asli 11–15 per foto).

## Keputusan pemetaan yang dipakai

- Jendela, termasuk jendela krepyak → `window-other` (`window-blind` tidak dipakai).
- Dinding cat → `wall-other-merged`, boleh lebih dari satu poligon per foto (per bidang dinding); dinding keramik → `wall-tile`.
- Lantai keramik → `floor-other-merged`.
- Kardus → `cardboard` (kecuali `normal_08`, lihat di bawah).
- Tas duffel → `backpack`.
- Stopkontak, terminal, adaptor → `plug`.
- Benda di luar 74 kelas (koper, kipas, gayung, bak mandi, sapu, helm, sepatu, router, dumbbell, rak piring, sutil) tidak dianotasi.

## Pasangan foto yang anotasinya berbeda

Pasangan normal/redup dari pemandangan yang sama dipakai untuk membandingkan kondisi cahaya. Perbedaan anotasi di
dalam pasangan membuat perbandingan itu tidak murni. Perbaiki kalau sempat, atau sebutkan di laporan.

| Pasangan | Perbedaan |
|---|---|
| normal_08 ↔ redup_06 | kotak yang sama: `box` 4 vs `cardboard` 4; `plug` hanya di normal_08 |
| oklusi_02 ↔ redup_08 | `chair` tidak dianotasi di redup_08 |
| normal_10 ↔ silau_01 | `light` dan `plug` tidak dianotasi di normal_10 |
| normal_12 ↔ redup_09 | `plug` hanya di redup_09 |
| normal_13 ↔ redup_10 | `plug` hanya di normal_13 |
| normal_11 ↔ backlight_02 | `ceiling-merged` hanya di normal_11 |
| normal_03 ↔ redup_03 | `floor-other-merged` 1 vs 2 poligon |
| normal_06, normal_07 ↔ redup_05 | `food-other-merged` dan `plate` 2 hanya di redup_05; `shelf` hanya di normal_07 |

## Catatan

- `mouse` (7 foto), `toothbrush` (3 foto), dan `scissors` (2 foto) tidak punya contoh di data train. Bahan studi kasus
  kegagalan "kelas tidak pernah dilihat".
- Benda di luar 74 kelas tidak dianotasi; keterbatasan ini dicatat di laporan.
