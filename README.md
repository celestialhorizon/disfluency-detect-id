# disfluency-id

Deteksi filler word dan disfluensi pada ucapan Bahasa Indonesia, sampai ke Edit
Decision List yang bisa dijalankan ffmpeg. Prototipe pendamping proposal
penelitian.

Perkaranya: dua kata identik berdampingan tidak selalu disfluensi.

```
saya saya mau bertanya        -> repetisi, dipotong
anak anak sekarang mandiri    -> reduplikasi gramatikal, dipertahankan
```

Detektor yang memakai asumsi Bahasa Inggris memotong keduanya. Pemisahnya ada
di `reduplication.py`.

## Batas data

Korpus benih `data/corpus/seed_id.txt` ditulis tangan dan penanda waktunya
disintesis, jadi angka di bagian [Hasil](#hasil) menunjukkan pipeline berjalan,
bukan temuan tentang ucapan Bahasa Indonesia.

Korpus di `data/corpus/gpu/` berasal dari rekaman nyata, tetapi seluruh
labelnya masih `O`. Anotasinya belum dikerjakan.

## Menjalankan

Python >= 3.10. Inti paket memakai pustaka standar saja, tanpa pemasangan
apa pun.

```bash
python -m disfluency_id stats        # ringkasan korpus benih
python -m disfluency_id build        # korpus -> data/corpus/seed_id.jsonl
python -m disfluency_id experiment   # perbandingan sistem + ablasi -> reports/
python -m disfluency_id train        # -> models/perceptron.json
```

Detektor pada teks bebas:

```bash
python -m disfluency_id detect --text "eee jadi saya saya mau bilang anak anak itu perlu istirahat"
```

```
eee/FP jadi/DM saya/REP saya mau bilang anak anak itu perlu istirahat

Span disfluensi terdeteksi:
  [FP]   0.250-  0.395s (0.145s) potong  "eee"
  [DM]   0.525-  0.755s (0.230s) simpan  "jadi"
  [REP]   0.885-  1.115s (0.230s) potong  "saya"

Transkrip bersih: jadi saya mau bilang anak anak itu perlu istirahat
```

Alasan tiap putusan reduplikasi, dengan rincian skornya:

```bash
python -m disfluency_id explain --text "anak anak sekarang lebih mandiri"
python -m disfluency_id explain --jsonl data/corpus/gpu/disfluency_1.jsonl
```

Edit Decision List, plus skrip ffmpeg dan transkrip bersihnya:

```bash
python -m disfluency_id edl --jsonl data/corpus/seed_id.jsonl        # -> reports/edl/
python -m disfluency_id render --edl reports/edl/edl.json --out potong.mp4
```

`render` butuh `ffmpeg` di PATH. Perintah lain tidak.

Uji:

```bash
python -m pytest -q          # 274 uji
```

## Transkripsi

Satu-satunya jalur ASR: `nyrahealth/CrisperWhisper` lewat `transformers`.
Modelnya dilatih verbatim, jadi jeda terisi ikut dituliskan; Whisper arus utama
menghapusnya, yakni menghapus objek penelitiannya.

Butuh GPU. Pada CPU tidak praktis.

```bash
pip install -r reqs.txt
python -m disfluency_id ingest --check
```

Rekaman tidak ikut repositori. Yang ikut hanya `data/media/provenans.csv`, dan
berkas yang tidak punya baris di sana tidak akan diunduh:

```bash
python -m disfluency_id media --unduh
python -m disfluency_id ingest --semua                  # transkrip -> data/corpus/gpu/
python -m disfluency_id audit data/corpus/gpu/*.jsonl
```

Transkrip yang sudah ada tidak dibuat ulang, jadi `ingest` yang terputus cukup
diulang perintahnya. Transkrip mentahnya ikut terkomit di
`data/transkrip/crisperwhisper/`, sehingga langkah ASR bisa dilewati:

```bash
python -m disfluency_id ingest \
    --whisper-json data/transkrip/crisperwhisper/disfluency_1.json \
    --out data/corpus/gpu/disfluency_1.jsonl
```

Perintah itu tidak memuat model, tidak butuh `transformers` maupun `torch`.

Setelan yang tidak sembarangan:

| Bendera | Bawaan | Alasan |
|---|---:|---|
| `--jendela` | 30 | rentang degenerasinya paling pendek dibanding 20 dan 15 |
| `--maks-token` | 160 | syarat muat VRAM 16 GB: tanpa batas, puncaknya 13,71 GB pada T4 14,56 GB |

`repetition_penalty` dan `no_repeat_ngram_size` tidak dipakai. Keduanya
menghukum pengulangan, dan pengulangan itulah datanya. Degenerasi ditangani
sesudah bangkitan oleh `cari_loop` dengan ambang >= 4 putaran; rentang yang
dibuang dicatat pada medan `rentang_rusak` di JSON transkrip.

### Mutu transkrip

Diukur `audit` pada `disfluency_1.mp4`, 57,8 detik ujaran spontan:

| Yang diukur | Nilai |
|---|---:|
| Token | 101 |
| Jeda antar-kata terukur | 99 |
| Jeda dilaporkan persis 0,000 s | 96 (97,0%) |
| Jeda terisi (`eee`, `uhm`) | 7 (6,93%) |
| Putusan `audit` | laju filler wajar |

Laju 6,93% jatuh di kisaran kepustakaan untuk ujaran spontan, jadi fillernya
tidak terhapus ASR.

Penanda waktunya lain soal. `_extract_token_timestamps` pada `transformers`
memberi tiap token satu angka lompatan DTW, sehingga akhir kata ke-*i* selalu
sama dengan awal kata ke-*i*+1 dan jedanya nol. Ini sifat jalurnya, bukan cacat
model, dan arahnya merugikan: modul reduplikasi membaca jeda rapat sebagai
bukti satu satuan ucap. Fitur prosodi baru sah dihitung di atas penanda waktu
hasil forced alignment (WhisperX, MFA).

Isi leksikalnya juga rusak di beberapa tempat (`berhaku`, `dianggar`,
`pu blic`). Koreksi manual wajib sebelum transkrip dipakai.

Keluaran `ingest` seluruhnya berlabel `O`. ASR hanya menghasilkan kata dan
waktunya; label disfluensi harus datang dari anotator manusia.

## Tagset

| Label | Arti | Dipotong? |
|---|---|---|
| `O`   | Fluen | tidak |
| `FP`  | Filled pause (`eee`, `emm`, `hmm`, `anu`) | ya |
| `DM`  | Discourse marker (`kayak`, `gitu`, `apa ya`) | hanya mode agresif |
| `REP` | Repetisi, salinan berlebih | ya |
| `RPR` | Reparandum + penanda ralat (`... ke pasar eh ke toko`) | ya |
| `PW`  | Fragmen kata (`peng-`) | ya |

`DM` tidak dipotong secara bawaan: membuang penanda wacana mengubah gaya tutur
orangnya, dan itu keputusan editorial.

## Bobot bukti reduplikasi

Tabel ini dihasilkan `weight_table()`, jangan disunting tangan. Ada uji yang
gagal kalau isinya berbeda dari konstanta penghitungnya.

<!-- TABEL-BOBOT: dihasilkan weight_table(), jangan disunting tangan -->

| Sumber | Kaidah | Bobot |
|---|---|---:|
| Leksikal | `anak-anak` terdaftar reduplikasi gramatikal | +3,0 |
| Leksikal | bentuk yang sudah bereduplikasi diulang lagi | -3,0 |
| Leksikal | filler diulang (`eee eee`) -- filler tidak punya reduplikasi | -2,5 |
| Morfosintaktis | kata fungsi (`saya`, `yang`, `di`) tidak produktif direduplikasi | -3,0 |
| Morfosintaktis | kelas terbuka | +0,5 |
| Prosodi | jeda <= 0,08 dtk (satu satuan ucap) | +1,5 |
| Prosodi | jeda >= 0,18 dtk | -1,5 |
| Prosodi | jeda di antara 0,08 dan 0,18 dtk, diinterpolasi lurus | +1,5 s/d -1,5 |
| Prosodi | salinan pertama terpotong (rasio panjang < 0,6) | -0,8 |
| Prosodi | kedua salinan sama panjang (rasio 0,8-1,25) | +0,4 |
| Struktural | muncul 3x berturut-turut atau lebih | -2,0 |

Zona `|skor| <= 0,75` dianggap ambigu dan putusan bawaan dipertahankan.

<!-- /TABEL-BOBOT -->

Bukti leksikal dan morfosintaktis berbobot lebih besar daripada prosodi. Pada
`disfluency_1.jsonl`, `lu lu` dan `tiba tiba` sama-sama berjeda 0,000 detik
tetapi diputus berlawanan, -1,50 dan +5,00. Pembobotan itu yang menyelamatkan
sistem ketika penanda waktunya rusak.

## Hasil

Korpus benih, validasi silang 5 lipatan. Sekali lagi: perilaku sistem, bukan
temuan empiris.

| Sistem | Akurasi | F1 mikro | F1 span | Pelestarian reduplikasi |
|---|---:|---:|---:|---:|
| naif (asumsi Bahasa Inggris) | 0.9360 | 0.6993 | 0.7419 | 0.8182 |
| aturan (sadar reduplikasi) | 0.9951 | 0.9783 | 0.9746 | 1.0000 |
| perceptron [lex] | 0.9565 | 0.8121 | 0.7339 | 1.0000 |
| perceptron [lex+prosodi] | 0.9705 | 0.8693 | 0.8089 | 0.9818 |
| perceptron [lex+reduplikasi] | 0.9828 | 0.9333 | 0.8945 | 1.0000 |
| perceptron [prosodi+reduplikasi] | 0.9204 | 0.6032 | 0.6927 | 0.9818 |
| perceptron [penuh] | 0.9803 | 0.9172 | 0.8803 | 1.0000 |

Sistem naif memotong 18% token reduplikasi, sementara akurasinya tetap 0.936
karena disfluensi kelas minoritas. Itu sebabnya pelestarian reduplikasi
dilaporkan terpisah dari F1 agregat.

Fitur reduplikasi penyumbang terbesar pada model terpelajar (F1 mikro 0.8121
menjadi 0.9333). Tanpa rumpun leksikal, model jatuh ke 0.6032.

Sistem aturan mengungguli model terpelajar pada korpus sekecil ini karena
aturannya disusun dari pengetahuan linguistik yang sama dengan yang dipakai
melabeli korpus, sedangkan model harus mempelajarinya dari 167 ujaran.

Uji signifikansi tidak dijalankan di sini karena datanya sintetis. Rencananya
ada di bagian 7 `reports/hasil_eksperimen.md` yang dihasilkan `experiment`.

## Struktur

```
disfluency_id/
  schema.py          Token, Utterance, Span, tagset, I/O JSONL
  lexicon.py         pemuatan & pencarian leksikon
  reduplication.py   pemisahan reduplikasi vs repetisi
  corpus.py          parsing anotasi, simulator waktu, pembagian data
  baseline.py        detektor naif & detektor aturan
  features.py        rekayasa fitur + sakelar ablasi
  model.py           structured averaged perceptron + Viterbi
  evaluate.py        P/R/F1 token & span + diagnostik reduplikasi
  experiment.py      perbandingan sistem + ablasi -> laporan
  edl.py             Edit Decision List, ffmpeg, transkrip bersih
  render.py          jalankan EDL -> mp4 terpotong (mode penuh/bukti)
  media.py           registri media: provenans.csv -> unduhan terverifikasi
  asr.py             transkripsi CrisperWhisper
  ingest.py          transkrip -> Utterance
  asr_audit.py       audit mutu transkrip sebelum dipakai
  cli.py             antarmuka baris perintah
data/
  lexicon/                       leksikon filler & reduplikasi (JSON)
  corpus/seed_id.txt             korpus benih beranotasi + pedoman anotasi
  corpus/gpu/                    korpus dari transkrip T4, belum dianotasi
  transkrip/crisperwhisper/      transkrip mentah
  media/provenans.csv            daftar sumber; mp4 tidak ikut repositori
tests/                           274 uji
```

Lintasan bawaan di `cli.py` berpatokan pada letak paketnya, bukan direktori
kerja, jadi klon boleh ditaruh di mana saja tanpa suntingan.

`reports/` dan `models/` dibuat sendiri oleh `experiment` dan `train`.

## Yang belum dikerjakan

- Korpus berlabel masih sintetis. Rekaman nyata sudah ditranskripsi tetapi
  belum dianotasi.
- Penanda waktu ASR belum layak untuk fitur prosodi. Forced alignment wajib
  dijalankan lebih dulu.
- Ambang laju filler ~6% pada `asr_audit.py` belum dirujukkan ke kepustakaan.
- Anotator tunggal, jadi Cohen kappa belum bisa dihitung.
- Belum ada pembanding neural. Antarmuka `fit`/`predict` pada `model.py`
  disiapkan supaya IndoBERT atau CRF bisa dipasang tanpa mengubah evaluasinya.
- Leksikon belum mencakup filler ragam daerah (`lho`, `to`, `mah`, `atuh`).
- Ambang prosodi 0,08 s dan 0,18 s masih perkiraan awal.
- Versi `transformers` dan `torch` tidak dipatok di `reqs.txt`. Versi yang
  terpakai dicetak setiap transkripsi berjalan.

## Etika

`ingest.py` menyentuh konten milik orang lain. Catatan lengkapnya ada di
docstring modul itu: hanya konten publik, audio mentah tidak diredistribusi,
nama diri disamarkan pada transkrip yang dipublikasikan, lisensi tiap sumber
dicatat di `data/media/provenans.csv`.

## Lisensi

MIT, lihat `LICENSE`. Hanya berlaku untuk kode. Rekaman sumber tidak berada di
repositori ini dan tetap tunduk pada ketentuan pemiliknya masing-masing.
