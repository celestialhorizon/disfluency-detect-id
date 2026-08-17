# Prototipe: Deteksi Filler Word & Disfluensi Ucapan Bahasa Indonesia

Bukti konsep yang mendampingi proposal penelitian **"Deteksi Filler Words &
Disfluency pada Ucapan Bahasa Indonesia untuk Automasi Penyuntingan
Video/Podcast"**.

Prototipe ini menjalankan seluruh rantai pipeline data science secara utuh —
dari korpus beranotasi, rekayasa fitur, pemodelan sekuens, evaluasi, sampai
keputusan potong yang siap dieksekusi ffmpeg — sehingga Bab 3 (Metodologi)
dapat ditulis dari sistem yang benar-benar berjalan, bukan dari rancangan di
atas kertas.

---

## ⚠️ Batas tafsir — baca lebih dulu

Korpus benih di `data/corpus/seed_id.txt` **ditulis tangan oleh peneliti**
sebagai contoh fenomena kebahasaan, dan penanda waktunya **disintesis**, bukan
diukur dari audio.

Artinya seluruh angka yang dihasilkan prototipe ini:

- **sahih** sebagai bukti bahwa pipeline berjalan utuh dan rancangan
  eksperimen dapat dieksekusi;
- **tidak sahih** sebagai temuan empiris tentang ucapan Bahasa Indonesia.

Jangan salin angka mana pun dari sini ke bagian "Hasil Penelitian". Yang boleh
disalin adalah **rancangan**-nya: tagset, pedoman anotasi, rumpun fitur,
metrik, dan alur eksperimen.

---

## Persoalan yang diteliti

Pada Bahasa Inggris, dua kata identik yang berdampingan hampir selalu
disfluensi:

```
the the cat        ->  the cat
```

Pada Bahasa Indonesia, bentuk yang **sama persis** bisa merupakan morfologi
produktif yang wajib dipertahankan:

```
anak anak sekarang mandiri   ->  anak-anak (JAMAK, gramatikal)  ✅ pertahankan
saya saya mau bertanya       ->  repetisi disfluen              ✂️ potong
```

Detektor yang memindahkan asumsi Bahasa Inggris apa adanya akan mengubah
`anak-anak` menjadi `anak` — merusak makna kalimat. Inilah celah yang
dijadikan dasar penelitian, dan prototipe ini mengukurnya secara langsung
lewat metrik **laju pelestarian reduplikasi**.

---

## Cara menjalankan

Tanpa pemasangan apa pun. Seluruh inti prototipe memakai pustaka standar
Python (≥ 3.10) saja. Untuk memasangnya di server, lihat
[Pemasangan di server Ubuntu](#pemasangan-di-server-ubuntu).

```bash
cd prototype

python -m disfluency_id stats        # ringkasan korpus
python -m disfluency_id experiment   # perbandingan sistem + ablasi -> reports/
python -m disfluency_id train        # latih model penuh -> models/
```

Menjalankan detektor pada kalimat bebas:

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

Perhatikan `saya saya` dipotong sementara `anak anak` dipertahankan.

Menelusuri **alasan** di balik putusan itu:

```bash
python -m disfluency_id explain --text "anak anak sekarang lebih mandiri"
```

```
--- input-0001 ---
Ujaran: anak anak sekarang lebih mandiri

Token 0-1  'anak anak'  jeda 0.130s
  Putusan   : REDUPLIKASI (pertahankan)
  Skor bukti: +3.90  (keyakinan 1.00)
    - word list: 'anak-anak' is a real reduplicated word
    - word type: 'anak' is a normal word, it can be doubled
    - gap: 0.130s is in between, worth -0.00
    - length: both copies are as long (ratio 1.00)

1 pasangan pada 1 ujaran yang diperiksa.
```

Dengan `--jsonl`, seluruh ujaran ditelusuri, bukan yang pertama saja:

```bash
python -m disfluency_id explain --jsonl data/corpus/gpu/disfluency_1.jsonl
```

Pada rekaman itu `lu lu` dan `tiba tiba` sama-sama berjeda 0,000 detik tetapi
diputus berlawanan — −1,50 dan +5,00. Jeda yang identik, putusan yang
berbeda: yang memutuskan bukti leksikal dan morfosintaktisnya.

Menyusun Edit Decision List:

```bash
python -m disfluency_id build                                  # korpus -> JSONL
python -m disfluency_id edl --jsonl data/corpus/seed_id.jsonl   # -> reports/edl/
```

Menjalankan uji:

```bash
python -m pytest -q          # 274 uji
```

### Memproses rekaman sungguhan

Ini satu-satunya bagian yang butuh pemasangan pustaka, karena transkripsi
tidak mungkin dilakukan dengan pustaka standar saja:

```bash
pip install -r reqs.txt
python -m disfluency_id ingest --check    # pastikan terpasang
```

Catat sumbernya di `data/media/provenans.csv` lebih dulu — berkas yang tidak
punya baris di sana tidak akan terunduh — lalu:

```bash
python -m disfluency_id media --unduh     # provenans.csv -> data/media/
```

Setelah berkasnya ada:

```bash
python -m disfluency_id ingest --semua                         # seluruh provenans
python -m disfluency_id audit  data/corpus/gpu/*.jsonl         # PERIKSA DULU
python -m disfluency_id detect --jsonl data/corpus/gpu/disfluency_1.jsonl
python -m disfluency_id edl    --jsonl data/corpus/gpu/disfluency_1.jsonl \
    --source data/media/disfluency_1.mp4
```

`ingest --semua` mentranskripsikan setiap rekaman yang terdaftar di provenans.
**Transkrip yang sudah ada tidak dibuat ulang**; ia dibaca kembali menjadi
korpus. Transkripsi adalah langkah termahal rangkaian ini, jadi jalankan yang
terputus cukup diulang perintahnya — ia melanjutkan, bukan mengulang dari nol.

Keberadaan transkrip diperiksa **sebelum** keberadaan media: transkrip yang
sudah jadi tidak memerlukan berkas medianya sama sekali. Urutan sebaliknya
membuat artefak termahal rangkaian ini tak terpakai hanya karena artefak
termurahnya belum diunduh ulang.

Untuk satu berkas saja, `--audio` tetap ada:

```bash
python -m disfluency_id ingest --audio data/media/sampel.mp4 --out data/corpus/sampel.jsonl
```

Jalankan `audit` sebelum mempercayai keluaran apa pun di atasnya. Perintah itu
mengukur dua kerusakan khas transkrip otomatis — penanda waktu yang runtuh dan
filler yang dihapus — dan menolak diam bila menemukannya. Rinciannya ada di
bagian [Mutu transkrip ASR](#mutu-transkrip-asr).

Video tidak perlu dikonversi lebih dulu; trek audio diambil langsung dari
wadahnya. Transkrip mentah disimpan di `data/transkrip/` supaya percobaan
berulang tidak perlu menjalankan Whisper lagi.

#### Transkripsi: CrisperWhisper, dan hanya itu

Satu-satunya jalur ASR proyek ini adalah **`nyrahealth/CrisperWhisper`** lewat
`transformers`. Tidak ada model kedua, dan itu keputusan metodologis bukan
kemudahan.

Model Whisper arus utama dilatih dari subtitle web yang sudah dirapikan
penyubtitle, sehingga ia belajar bahwa jeda terisi "bukan teks" dan
menghapusnya — yakni menghapus persis objek penelitian ini. CrisperWhisper
dilatih verbatim: pada `disfluency_1.mp4` ia menuliskan **6,93%** token berupa
jeda terisi, laju yang membuat `audit` meluluskan berkasnya.

Ongkosnya nyata dan harus ikut ditanggung: isi leksikalnya lebih sering rusak
(`berhaku`, `dianggar`, `pu blic`), ia sesekali menerjemahkan ke Inggris walau
`--language id` dipatok, dan bangkitannya bisa degenerasi. Yang terakhir
ditangani `cari_loop` **sesudah** bangkitan, bukan dengan `repetition_penalty`
— lihat di bawah.

**Butuh GPU.** Pada CPU transkripsi CrisperWhisper tidak praktis. Prototipe ini
menjalankannya di T4 gratis Google Colab; lihat
[Ruang kerja di Google Colab](#ruang-kerja-di-google-colab).

```bash
python -m disfluency_id ingest --audio data/media/sampel.mp4 \
    --out data/corpus/sampel.jsonl
```

Setelannya dapat diubah, tetapi bawaannya bukan angka sembarangan:

| Bendera | Bawaan | Alasan |
|---|---:|---|
| `--jendela` | 30 | 30/20/15 diadu; 30 dtk yang rentang degenerasinya paling pendek |
| `--maks-token` | 160 | **syarat muat di VRAM**, bukan penghematan waktu |
| `--tanpa-koreksi-senyap` | mati | senyap dikikis dari ujung kata lewat pengukuran akustik |

`--maks-token` adalah syarat perangkat keras. `return_timestamps="word"`
menahan cross-attention 32 lapisan dekoder untuk **tiap** langkah dekode.
Tanpa batas, bangkitan lari sampai 448 langkah dan puncaknya **13,71 GB** —
gagal pada T4 yang hanya 14,56 GB. Dengan batas 160, puncaknya **10,89 GB**.
Diukur pada kernel bersih, bukan ditaksir.

#### Batas rancangan: dua setelan yang haram di sini

`repetition_penalty` dan `no_repeat_ngram_size` adalah cara baku meredam
degenerasi bangkitan, dan **tidak pernah** disetel di proyek ini. Keduanya
menghukum pengulangan, padahal pengulangan itulah yang diteliti — `lu lu`,
`tiba tiba`, `akhirnya … akhirnya`. Menyetelnya sama dengan menghapus data
sendiri lalu melaporkan datanya tidak ada.

Degenerasi karena itu ditangani **sesudah** bangkitan oleh `cari_loop`, dengan
ambang >= 4 putaran: pengulangan sah yang panjangnya 2 putaran lolos,
sementara putaran runaway 26x tertangkap. Kata yang dibuang **dicatat**
rentangnya pada medan `rentang_rusak` di JSON transkrip — membiarkannya masuk
akan mencemari korpus, membuangnya diam-diam akan menyembunyikan bahwa ada
audio tanpa transkrip sah.

Prinsip umumnya layak dibawa ke tugas lain: **penekan artefak bangkitan baku
tidak boleh dipakai pada tugas yang targetnya berbentuk permukaan sama dengan
artefaknya.**

#### Memisahkan langkah ASR dari langkah sesudahnya

`ingest` juga menerima transkrip yang sudah jadi, sehingga langkah ASR dapat
dijalankan di mesin ber-GPU dan sisanya di mana saja:

```bash
python -m disfluency_id ingest \
    --whisper-json data/transkrip/crisperwhisper/sampel.json \
    --out data/corpus/gpu/sampel.jsonl
```

Perintah itu tidak memanggil model sama sekali, jadi selesai seketika dan
tidak memerlukan `transformers` maupun `torch` terpasang. Inilah jalur yang
dipakai sehari-hari; transkrip mentahnya ikut terkomit di dalam repositori
supaya percobaan berulang tidak perlu mengulang langkah termahalnya.

Satu batas yang perlu dijaga: perangkat keras yang berbeda menghasilkan
transkrip yang **tidak sama persis** meski model dan berkasnya sama. Korpus
yang sebagian dari satu lingkungan dan sebagian dari lingkungan lain karena
itu membawa sumber variasi yang tidak dapat dipisahkan dari temuan —
transkripsikan seluruh berkas di satu lingkungan yang sama, jangan dicampur.

Keluaran `ingest` seluruhnya berlabel `O`. ASR hanya menghasilkan kata dan
waktunya — label disfluensi tetap harus diberikan anotator manusia, sebab
label yang dihasilkan mesin tidak bisa dipakai menilai mesin yang sama.

---

## Ruang kerja di Google Colab

Seluruh prototipe dijalankan di Colab. Kodenya diambil dari GitHub:

    https://github.com/celestialhorizon/disfluency-detect-id

Alurnya searah, dan searahnya disengaja:

    sunting di laptop  ->  git commit & push  ->  sel bagian 4 `git pull` di Colab

Colab adalah **pemakai** kode, bukan tempat kode disunting.

| Lokasi | Isi |
|---|---|
| `/content/disfluency-detect-id/` | klon repositori ini (`AKAR`) |
| `/content/disfluency-detect-id/data/media/` | rekaman, **diunduh**, tidak ada di repositori (`MEDIA`) |
| `/content/simpanan/<cap-waktu>/` | salinan berkas terlacak yang tertimpa saat menarik |

`/content` dihapus habis setiap kali sesi Colab berakhir: menganggur kira-kira
90 menit, paling lama 12 jam, terputus, atau runtime diganti. Apa pun yang
dihasilkan di sana — transkrip, korpus, EDL, mp4 — lenyap bersama runtimenya.
Itu diterima sebagai konsekuensi: hasil yang perlu bertahan dikomit dari
laptop, bukan dari Colab.

### Cara sel penarik bekerja

Sel bagian 4 memakai `git fetch` + `git reset --hard origin/master`, bukan
`git pull`. Alasannya:

- `git pull` **gagal** begitu ada berkas terlacak yang berubah, dan di sini
  transkrip serta korpus memang berubah setiap kali bagian 5 dan 6 dijalankan.
- Yang dikehendaki adalah origin selalu menang untuk kode.
- `reset --hard` **tidak menyentuh berkas tak terlacak**, sehingga rekaman yang
  sudah diunduh ke `data/media/` dan keluaran di `reports/` bertahan lintas
  penarikan. Hanya berkas terlacak yang dikembalikan ke versi origin.

Karena `reset --hard` menghapus perubahan pada berkas terlacak, sel itu
**menyalin dulu** setiap berkas terlacak yang kotor ke
`/content/simpanan/<cap-waktu>/` dan mencetak daftarnya sebelum menimpa. Sebuah
transkrip yang baru dihasilkan di Colab karena itu tidak lenyap diam-diam; ia
tetap harus disalin keluar bila hendak disimpan.

### Notebooknya tidak ada di repositori ini

Notebook kerja hidup di Colab, dan tidak ada mekanisme apa pun yang
menyinkronkannya ke sini. Yang berpindah lewat repositori ini adalah **kode**,
bukan notebooknya.

Bila perlu potret mati untuk lampiran: **File → Download → Download .ipynb**.
Bersihkan keluaran selnya lebih dulu lewat **Edit → Clear all outputs** —
sel transkrip mencetak percakapan lengkap beserta nama orang, dan keluaran sel
ikut tersimpan di dalam berkas `.ipynb`.

### Rekaman sumber tidak berada di repositori

`.gitignore` mengecualikan seluruh berkas media, dan pengecualian itu bukan
soal ukuran melainkan etika: isinya suara orang sungguhan, dan seluruhnya
bahan pihak ketiga. Colab karena itu **mengunduhnya**, bukan menerimanya lewat
klon:

```bash
python -m disfluency_id media --unduh
```

`data/media/provenans.csv` adalah satu-satunya daftar sumber — tidak ada daftar
kedua di notebook maupun di kode. Akibatnya bukan sekadar rapi: **berkas yang
belum punya baris provenans tidak akan terunduh**, sehingga syarat "provenans
diisi sebelum berkas diproses" berlaku dengan sendirinya alih-alih menjadi
aturan yang harus diingat.

Unduhannya diperiksa terhadap `Content-Length`, bukan sekadar `exists()`:
unduhan yang putus di tengah meninggalkan berkas yang **ada** tetapi pendek,
dan pemeriksaan keberadaan saja akan menerimanya sebagai lengkap — lalu ASR
mentranskripsikan potongan itu tanpa memberi tahu siapa pun. Berkas diunduh ke
`.part` dan baru diberi nama aslinya setelah jumlah baitnya cocok.

### Lintasan tidak perlu disunting

Seluruh lintasan bawaan di `cli.py` berpatokan pada
`ROOT = Path(__file__).resolve().parent.parent` — yakni letak paketnya sendiri,
**bukan** direktori kerja. Klon boleh diletakkan di mana pun tanpa satu pun
suntingan lintasan.

### Sebelum notebook dibagikan kepada siapa pun

**Bersihkan keluaran sel lebih dulu.** Keluaran tersimpan di dalam berkas
notebook dan terbaca tanpa menjalankan apa pun; bagian 9 mencetak transkrip
wawancara lengkap dengan nama orang. Membagikan tautannya sama dengan
menerbitkan transkrip itu — bertentangan dengan syarat anonimisasi yang
dipegang proyek ini sendiri.

Berbagi sebagai *Viewer* tetap kebiasaan yang benar, sebab sel sisipan berjalan
dengan kewenangan yang menjalankannya. Notebook tidak mengaitkan Drive sama
sekali, jadi taruhannya sebatas runtime.

Yang perlu diketahui saat menjalankannya:

- Colab menghapus paket **dan seluruh `/content`** setiap sesi mati. Sel
  penarik dan `pip install` karena itu diulang setiap sesi baru, begitu pula
  unduhan medianya.
- `PYTHONDONTWRITEBYTECODE=1` disetel sebelum menjalankan apa pun, dan `pytest`
  dijalankan dengan `-p no:cacheprovider`. Alasannya kebersihan `git status`:
  `__pycache__` yang bertaburan membuat keluaran status sukar dibaca.
- Hanya transkripsi yang membutuhkan GPU. Seluruh perintah lain berjalan pada
  runtime CPU, sebab tak satu pun memakai pustaka pihak ketiga.
- Transkripsi (bagian 5) boleh dilewati: transkripnya sudah ikut terkomit di
  dalam repositori. Ia hanya perlu dijalankan bila transkrip dibuat ulang dari
  mp4.
- Satu langkah tetap menuntut tangan manusia: mengganti jenis runtime ke GPU.

## Pemasangan di server Ubuntu

Bagian ini tetap berlaku bila suatu saat prototipe dipindahkan ke mesin
sendiri; ia bukan lagi jalur yang dipakai sehari-hari.

**Pakai Ubuntu 24.04 bila bisa memilih.** Python bawaannya 3.12, persis versi
yang dipakai menguji seluruh angka di dokumen ini.

Batas versinya berbeda untuk dua bagian prototipe:

| Bagian | Python minimum | Alasan |
|---|---|---|
| Inti + uji | 3.10 | `requires-python` paket dan pytest |
| Transkripsi ASR | **3.10** | mengikuti `transformers`; yang membatasi di sini GPU, bukan versi Python |

Versi `transformers` dan `torch` **sengaja tidak dipatok** (lihat `reqs.txt`),
sehingga yang terpasang mengikuti apa yang tersedia pada mesin itu. Ini
pelemahan keterulangan yang disadari; versinya dicetak setiap kali transkripsi
berjalan supaya tetap tercatat. Bila hasil transkripsi harus sebanding dengan
yang sudah ada, samakan versinya secara manual dan pasang Python 3.12:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa   # hanya perlu di 22.04
sudo apt install -y python3.12 python3.12-venv
```

### 1. Paket sistem

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg tmux
```

`ffmpeg` **wajib**: transkripsi mengekstrak trek audio jadi WAV 16 kHz mono
lebih dulu, sebab pipeline `transformers` tidak dapat disodori `.mp4`
langsung. Satu hal lain juga membutuhkannya: `render` beserta skrip potong yang
dihasilkan `edl` (`reports/edl/*_ffmpeg.sh`).
Bila server hanya dipakai menghitung dan pemotongan videonya dikerjakan di
tempat lain, paket itu boleh dilewati.

### 2. Lingkungan Python

```bash
cd prototype

python3 -m venv .venv          # di 22.04: python3.12 -m venv .venv
source .venv/bin/activate
pip install -r reqs.txt
```

### 3. Verifikasi

```bash
python -m disfluency_id ingest --check    # pustaka opsional
python -m pytest -q                       # 274 uji
python -m disfluency_id stats             # korpus terbaca
```

Ketiganya berdiri sendiri. Bila `ingest --check` melaporkan "belum dipasang"
sementara `pytest` lulus, yang gagal hanya jalur transkripsi — `stats`,
`experiment`, `train`, `detect`, `explain`, `audit`, dan `edl` tetap dapat
dijalankan, sebab tidak satu pun bergantung pada pustaka pihak ketiga.

### 4. Transkripsi menuntut GPU

CrisperWhisper dijalankan lewat `transformers`, dan pada CPU ia tidak praktis.
Server tanpa GPU tetap dapat menjalankan **seluruh sisa prototipe** —
`stats`, `build`, `experiment`, `train`, `detect`, `explain`, `audit`, `edl`,
`render`, dan `ingest --whisper-json` — sebab tak satu pun bergantung pada
pustaka pihak ketiga. Yang tidak bisa hanya `ingest --audio`.

Pola kerjanya karena itu: transkripsikan di mesin ber-GPU, salin JSON-nya,
lanjutkan di mana saja.

**Arahkan cache model ke partisi berkapasitas.** Bawaannya `~/.cache`, dan
bobot CrisperWhisper ~3 GB. Pada VPS dengan partisi root tipis, unduhan gagal
dengan pesan yang menyesatkan — ia menyebut kegagalan penyusunan berkas,
padahal sebabnya ruang cakram.

```bash
echo 'export HF_HOME=/srv/hf-cache' >> ~/.bashrc
```

Tulis ke `~/.bashrc` atau ke unit systemd-nya, bukan sekadar diketik di sesi
berjalan. Proses yang dilepas ke latar tidak selalu mewarisi peubah dari sesi
tempat ia diluncurkan, dan bila tidak mewarisinya ia akan mengunduh ulang
model ~3 GB itu ke lokasi bawaan tanpa memberi tahu.

**Batas VRAM-nya nyata.** `--maks-token 160` adalah syarat muat pada GPU 16 GB,
bukan penghematan waktu. Pada kartu yang lebih besar batas itu boleh dinaikkan;
pada yang lebih kecil ia harus diturunkan, dan cakupan transkripnya ikut
menurun.

**Lepas prosesnya dengan `tmux`:**

```bash
tmux new -s transkrip
python -m disfluency_id ingest --audio data/media/sampel.mp4 \
    --transcript data/transkrip/crisperwhisper/sampel.json \
    --out data/corpus/sampel.jsonl
# lepas: Ctrl-b lalu d      sambung lagi: tmux attach -t transkrip
```

Dengan `nohup`, tambahkan `--no-progress` supaya ringkasannya tidak bercampur
dengan log lain:

```bash
nohup python -m disfluency_id ingest --audio data/media/sampel.mp4 \
    --no-progress --out data/corpus/sampel.jsonl > transkrip.log 2>&1 &
```

Unduhan model hanya terjadi pada jalankan pertama, jadi jalankan sekali secara
interaktif lebih dulu — supaya kegagalan jaringan atau ruang cakram terlihat
langsung, bukan terkubur di dalam log pekerjaan latar.

**Periksa `rentang_rusak` sesudahnya.** Medan itu mencatat rentang degenerasi
yang dibuang, yakni audio tanpa transkrip sah. Bila isinya panjang, turunkan
`--jendela` dan bandingkan; jangan diabaikan.

---

## Tagset

| Label | Arti | Dipotong? |
|---|---|---|
| `O`   | Fluen | tidak |
| `FP`  | Filled pause (`eee`, `emm`, `hmm`, `anu`) | ya |
| `DM`  | Discourse marker filler (`kayak`, `gitu`, `apa ya`) | hanya mode agresif |
| `REP` | Repetition disfluency — salinan berlebih | ya |
| `RPR` | Reparandum + penanda ralat (`... ke pasar eh ke toko`) | ya |
| `PW`  | Partial word — fragmen kata (`peng-`) | ya |

`DM` sengaja dikecualikan dari pemotongan default: membuang penanda wacana
mengubah **gaya tutur** seseorang. Itu keputusan editorial, bukan pembersihan.

---

## Cara membedakan reduplikasi dari repetisi

Empat rumpun bukti digabung menjadi satu skor berbobot, dan setiap putusan
menyimpan alasannya agar dapat diaudit. Tabel di bawah ini **dihasilkan dari
konstanta yang dipakai penghitungnya**, bukan diketik ulang; ada uji yang
gagal kalau keduanya berbeda:

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

Bukti leksikal dan morfosintaktis sengaja diberi bobot lebih besar daripada
prosodi, sehingga **fakta morfologis mengalahkan penanda akustik**: `saya saya`
yang diucapkan tanpa jeda tetap dinilai disfluensi, dan `anak anak` dengan jeda
panjang tetap dipertahankan. Skor di zona `|skor| ≤ 0.75` dinyatakan **ambigu**
dan secara default dipertahankan — memotong reduplikasi merusak makna,
sedangkan menyisakan satu repetisi hanya menyisakan kekasaran gaya.

---

## Hasil (korpus benih, validasi silang 5 lipatan)

| Sistem | Akurasi | F1 mikro | F1 span | Pelestarian reduplikasi |
|---|---:|---:|---:|---:|
| naif (asumsi Bahasa Inggris) | 0.9360 | 0.6993 | 0.7419 | **0.8182** |
| aturan (sadar reduplikasi) | 0.9951 | 0.9783 | 0.9746 | **1.0000** |
| perceptron [lex] | 0.9565 | 0.8121 | 0.7339 | 1.0000 |
| perceptron [lex+prosodi] | 0.9705 | 0.8693 | 0.8089 | 0.9818 |
| perceptron [lex+reduplikasi] | 0.9828 | 0.9333 | 0.8945 | 1.0000 |
| perceptron [prosodi+reduplikasi] | 0.9204 | 0.6032 | 0.6927 | 0.9818 |
| perceptron [penuh] | 0.9803 | 0.9172 | 0.8803 | 1.0000 |

Yang dapat dibaca dari tabel ini (sekali lagi: sebagai perilaku sistem, bukan
temuan empiris):

1. **Sistem naif memotong 18% token reduplikasi.** Akurasinya tampak tinggi
   (0.936) justru karena disfluensi adalah kelas minoritas — inilah alasan
   metrik pelestarian reduplikasi dilaporkan terpisah dari F1 agregat.
2. **Fitur reduplikasi menyumbang paling besar** pada model terpelajar:
   F1 mikro 0.8121 → 0.9333 ketika rumpun fitur itu dinyalakan.
3. **Fitur leksikal tidak tergantikan.** Tanpa rumpun leksikal, model jatuh ke
   0.6032 — bukti prosodi saja tidak cukup.
4. **Sistem aturan mengungguli model terpelajar** pada korpus sekecil ini.
   Ini wajar dan dilaporkan apa adanya: aturan disusun dari pengetahuan
   linguistik yang sama dengan yang dipakai melabeli korpus, sementara model
   harus mempelajarinya dari 167 ujaran. Pada korpus nyata yang jauh lebih
   besar dan lebih berisik, urutan ini diharapkan berbalik — dan itulah salah
   satu hal yang diuji penelitian sesungguhnya.

Uji signifikansi **tidak** dijalankan di sini karena pada data sintetis
hasilnya menyesatkan. Rencana pengujian pada data sungguhan (McNemar
berpasangan, bootstrap berpasangan, Cohen kappa antaranotator) dicantumkan di
bagian 7 laporan `reports/hasil_eksperimen.md`.

---

## Struktur berkas

```
disfluency-detect-id/
├── disfluency_id/
│   ├── schema.py          Token, Utterance, Span, tagset, I/O JSONL
│   ├── lexicon.py         pemuatan & pencarian leksikon
│   ├── reduplication.py   ★ pemisahan reduplikasi vs repetisi
│   ├── corpus.py          parsing anotasi, simulator waktu, pembagian data
│   ├── baseline.py        detektor naif & detektor aturan
│   ├── features.py        rekayasa fitur + sakelar ablasi
│   ├── model.py           structured averaged perceptron + Viterbi
│   ├── evaluate.py        P/R/F1 token & span + diagnostik reduplikasi
│   ├── edl.py             Edit Decision List, ffmpeg, transkrip bersih
│   ├── render.py          menjalankan EDL -> mp4 terpotong (mode penuh/bukti)
│   ├── asr.py             ★ transkripsi CrisperWhisper (satu-satunya jalur ASR)
│   ├── ingest.py          berkas lokal -> transkrip -> Utterance (tanpa unduh)
│   ├── media.py           ★ registri media: provenans.csv -> unduhan terverifikasi
│   ├── asr_audit.py       audit mutu transkrip ASR sebelum dipakai
│   ├── experiment.py      perbandingan sistem + ablasi -> laporan
│   └── cli.py             antarmuka baris perintah
├── data/
│   ├── lexicon/                leksikon filler & sumber daya reduplikasi (JSON)
│   ├── corpus/seed_id.txt      korpus benih beranotasi + pedoman anotasi
│   ├── corpus/gpu/             ★ korpus dari transkrip T4 — dasar anotasi
│   ├── transkrip/crisperwhisper/ ★ transkrip CrisperWhisper -- satu-satunya
│   └── media/                  provenans.csv = daftar sumber — mp4 TIDAK ikut repo
├── tests/                 274 uji
├── reports/               keluaran eksperimen (dihasilkan otomatis)
├── models/                perceptron.json hasil `train`
├── reqs.txt               pustaka pihak ketiga (hanya ASR + uji)
└── pyproject.toml         metadata paket & ketergantungan opsional
```

---

## Mutu transkrip ASR

Diukur pada `disfluency_1.mp4` (57,8 detik ujaran spontan Bahasa Indonesia)
memakai `python -m disfluency_id audit`. Berbeda dengan angka pada bagian
Hasil, **angka di bagian ini pengukuran sungguhan** — datanya rekaman nyata,
bukan korpus benih.

Angkanya keluaran program, dan dapat dihitung ulang kapan saja: transkripnya
ikut repositori, jadi `audit` bisa dijalankan ulang tanpa GPU.

| Yang diukur | **CrisperWhisper** |
|---|---:|
| Ujaran | 2 |
| Token | 101 |
| Jeda antar-kata terukur | 99 |
| Jeda dilaporkan persis 0,000 s | 96 (**97,0%**) |
| Median jeda antar-kata | 0,000 s |
| **Jeda terisi (`eee`, `uhm`)** | **7 (6,93%)** |
| Penanda wacana | 10 |
| Putusan `audit` | laju filler wajar |

Tiga kesimpulan, dan ketiganya membatasi apa yang boleh diklaim dari data ASR.

**Laju jeda terisi itu yang menentukan bahan ini layak diteliti.** 6,93% jatuh
di kisaran kepustakaan untuk ujaran spontan, sehingga `audit` meluluskan
berkasnya alih-alih memvonisnya "diduga dihapus ASR". Laju setinggi itu didapat
karena model transkripsinya dilatih verbatim — model yang dilatih dari subtitle
rapi menuliskan jauh lebih sedikit, sebab ia belajar bahwa jeda terisi bukan
teks. **Inilah alasan proyek ini memakai CrisperWhisper dan tidak menyediakan
pilihan model lain.**

**Penanda waktu runtuh, dan mengganti model tidak menolong.** 97,0% jeda
antar-kata dilaporkan persis nol. Sebabnya struktural, bukan cacat model:
`_extract_token_timestamps` pada `transformers` memberi tiap token **satu**
angka — waktu lompatan DTW-nya — sehingga awal dan akhir kata disusun dari dua
lompatan berurutan dan akhir kata ke-*i* selalu sama dengan awal kata ke-*i*+1.
Ubin rapat itu sifat jalur DTW-nya, jadi model apa pun yang lewat jalur ini
memberi hasil serupa. Akibatnya tetap merugikan dan **berarah**: modul
reduplikasi membaca jeda rapat sebagai bukti "satu satuan ucap", sehingga jeda
nol palsu terkirim ke setiap pasangan kata berulang dan sistem condong
mempertahankan repetisi yang seharusnya dipotong. Menambah data tidak
menghapusnya. Fitur prosodi karena itu hanya sah dihitung di atas penanda waktu
hasil forced alignment (WhisperX, Montreal Forced Aligner) — dan itu prasyarat
kesahihan, bukan saran perbaikan.

**Transkripnya tidak boleh dipakai apa adanya.** Isi leksikalnya rusak di
beberapa tempat (`berhaku`, `dianggar`, `pu blic`), sebagian ucapan dibuang
sebagai rentang degenerasi, dan keduanya masuk korpus tanpa tanda apa pun kalau
tidak diperiksa orang. Koreksi manual transkrip karena itu langkah wajib, bukan
opsional.

### Perbaikan yang lahir dari data ini

Whisper memecah kata bertanda hubung menjadi dua token (`kira-kira` →
` kira` + `-kira`), yang bagi bahasa penanda reduplikasi berakibat serius: satu
kata terhitung dua token, dan batas semu di tengah kata dilaporkan berjeda
0,000 detik lalu terbaca sebagai bukti prosodi. Pada satu berkas, penyatuan
kembali token tersebut menurunkan "pasangan kata berulang" dari 14 menjadi 2 —
**dua belas di antaranya artefak tokenizer**, dan kesepuluh putusan
"reduplikasi" di dalamnya palsu seluruhnya. Lihat `merge_hyphen_continuations`
pada `ingest.py`.

Data ini juga membetulkan cara audit menghitung penanda ralat. Semula ia
menghitung keanggotaan leksikon, sehingga `bukan` terhitung sebagai penanda
ralat padahal sering hanya pengingkar biasa ("jadi memang **bukan** setiap
agama itu"). Detektornya sendiri sudah benar — ia menolaknya, sebab `bukan`
bukan penanda kuat sehingga hanya memicu bila ada kecocokan salinan kasar. Yang
keliru hanya auditnya, yang tidak bisa melihat pembedaan itu karena tersimpan
sebagai konstanta di dalam detektor. Pembedaan tersebut kini berada di leksikon
(`editing_term_kuat`), dibaca kedua modul, dan bentuk ambigu dilaporkan
terpisah persis seperti `filled_pause_ambiguous` — dihitung dan ditampilkan,
tetapi tidak masuk total laju filler.

### Bukti bahwa pembobotannya benar

`lu lu` dan `tiba tiba` sama-sama berjeda 0,000 detik, tetapi diputus
**berlawanan**: yang pertama disfluensi (−1,50, kaidah kata fungsi), yang kedua
reduplikasi (+5,00). Jeda yang identik, putusan yang berbeda — yang menentukan
bukti leksikal dan morfosintaktisnya.

Itu memvalidasi satu keputusan rancangan yang diambil sebelum data nyata masuk:
bukti leksikal/morfosintaktis sengaja diberi bobot lebih besar daripada
prosodi, dan justru itu yang menyelamatkan sistem ketika prosodinya rusak.

---

## Pemetaan ke pipeline data science (untuk Bab 3)

| Tahap CRISP-DM | Berkas | Keadaan pada prototipe |
|---|---|---|
| Business Understanding | `README.md` | Automasi penyuntingan podcast/video |
| Data Collection | `media.py`, `asr.py`, `ingest.py` | Provenans sebagai registri; CrisperWhisper |
| Data Preparation | `corpus.py`, `data/corpus/` | Korpus benih tulis tangan + pedoman anotasi |
| Feature Engineering | `features.py` | 3 rumpun fitur, dapat diablasi |
| Modeling | `model.py`, `baseline.py` | Perceptron terstruktur + 2 garis dasar |
| Evaluation | `evaluate.py`, `experiment.py` | Validasi silang, P/R/F1, metrik reduplikasi |
| Deployment | `edl.py` | EDL JSON/CSV + skrip ffmpeg + transkrip bersih |

---

## Yang belum dikerjakan (jujur, untuk bagian Keterbatasan)

- **Korpus berlabel masih sintetis.** Rekaman nyata sudah masuk dan sudah
  ditranskripsi (lihat [Mutu transkrip ASR](#mutu-transkrip-asr)), tetapi
  belum dianotasi, sehingga seluruh angka pada bagian Hasil tetap berasal
  dari korpus benih dengan penanda waktu sintetis.
- **Penanda waktu ASR belum layak untuk fitur prosodi.** Terukur 92,8%
  jeda bernilai nol. Forced alignment (WhisperX / MFA) wajib dijalankan
  sebelum rumpun fitur prosodi dihitung di atas data nyata.
- **Ambang laju filler belum bersumber.** Pembanding ~6% pada `asr_audit.py`
  dipakai untuk menandai kecurigaan penghapusan dan masih perlu dirujukkan
  ke kepustakaan disfluensi, atau diganti angka dari korpus lisan Indonesia.
- **Anotator tunggal.** Korpus benih dilabeli satu orang, sehingga
  keterandalan antaranotator (Cohen kappa) belum dapat dihitung.
- **Belum ada pembanding neural.** IndoBERT dan CRF belum dipasang;
  antarmuka `fit`/`predict` pada `model.py` sengaja dibuat agar keduanya
  dapat dipasang tanpa mengubah pipeline evaluasi.
- **Leksikon belum menjangkau ragam daerah.** Filler Jawa dan Sunda yang
  lazim menyusup ke ucapan Bahasa Indonesia (`lho`, `to`, `mah`, `atuh`)
  belum tercakup.
- **Ambang prosodi belum dikalibrasi.** Nilai 0,08 s dan 0,18 s adalah
  perkiraan awal yang wajib diukur ulang dari data nyata.

---

## Catatan etika

Modul `ingest.py` menyentuh konten milik orang lain. Catatan etika lengkap ada
pada docstring modul tersebut: hanya konten publik, hormati ketentuan layanan
platform, audio mentah tidak diredistribusi, nama diri disamarkan pada
transkrip yang dipublikasikan, dan lisensi tiap sumber dicatat dalam berkas
provenans.

---

## Lisensi

Kode dalam repositori ini berlisensi MIT — lihat `LICENSE`. Silakan dipakai,
diubah, dan dilanjutkan, dengan mencantumkan pemberitahuan hak cipta.

Lisensi tersebut **hanya berlaku untuk kode**. Rekaman sumber tidak berada di
repositori ini dan tetap tunduk pada ketentuan pemiliknya masing-masing;
status lisensi tiap rekaman dicatat dalam `data/media/provenans.csv`.
# trigger reindex
