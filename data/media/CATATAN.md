# Folder masukan media

Folder ini **kosong di repositori**, dan memang harus begitu. `.gitignore` di
akar proyek mengecualikan seluruh berkas audio/video. Alasannya bukan ukuran
melainkan etika: isinya suara orang sungguhan, dan ketentuan pada docstring
`ingest.py` melarang redistribusi audio mentah.

Format yang bisa dibaca: mp4, mkv, mov, webm, wav, mp3, m4a, flac. Trek audio
diambil langsung dari wadah video, jadi video tidak perlu dikonversi dulu.

## Cara mengisinya

`provenans.csv` di folder ini adalah **satu-satunya daftar sumber**. Tidak ada
daftar kedua di tempat lain — tidak di notebook, tidak di kode.

    python -m disfluency_id media            # lihat daftar dan status lokalnya
    python -m disfluency_id media --unduh    # unduh yang belum ada

Perintahnya sama persis di Colab dan di laptop. Yang sudah lengkap dilewati,
jadi menjalankannya berulang kali aman.

Akibat susunan ini yang paling penting bukan kemudahannya, melainkan
urutannya: **berkas yang belum punya baris provenans tidak akan terunduh.**
Syarat "provenans diisi sebelum berkas diproses" karena itu berlaku dengan
sendirinya, bukan lagi aturan yang harus diingat.

Enam rekaman terdaftar sekarang, seluruhnya 11,93 menit — potongan hasil
suntingan dari 272,2 menit karya asli, yakni **4,38%**-nya:

| Berkas | Kanal | Potongan | Asli | Porsi |
|---|---|---:|---:|---:|
| `disfluency_1.mp4` | Deddy Corbuzier | 57,80 dtk | 4m05s | 23,6% |
| `disfluency_2.mp4` | SUARA BERKELAS | 104,54 dtk | 59m02s | 3,0% |
| `disfluency_3.mp4` | KaisarTV; Adythia Pratama | 161,93 dtk | 84m27s | 3,2% |
| `disfluency_4.mp4` | Fellexandro Ruby; Nago Tejena | 132,77 dtk | 53m19s | 4,2% |
| `disfluency_5.mp4` | Wellspring Conversations | 121,42 dtk | 25m52s | 7,8% |
| `disfluency_6.mp4` | Rory Asyari; Dr Geofakta Razali | 137,57 dtk | 45m28s | 5,0% |

Judul lengkap dan URL YouTube tiap berkas ada di `provenans.csv`.

Dua kolom tautan di sana berbeda maksud dan tidak boleh disatukan. **Asal
kanonis** (`tautan_sumber`) adalah halaman YouTube tempat karya itu terbit: itu
yang dikutip, dan itu yang menentukan lisensinya. **Cermin unduh**
(`tautan_unduh`) cuma tempat rangkaian ini mengambil potongannya. Halaman
YouTube tidak mengembalikan mp4 bila diambil langsung, jadi keduanya memang
tidak bisa saling menggantikan.

Kedua durasi juga disimpan terpisah supaya **porsi kutipan dapat dihitung,
bukan sekadar dinyatakan** — itulah ukuran yang menimbang penggunaan wajar,
dan `python -m disfluency_id media` mencetaknya tiap kali dijalankan.

Durasi potongan dibaca dari atom `mvhd` berkas mp4 dan **belum dipastikan
ffprobe**.

Unggahan pertama `disfluency_5.mp4` bitratenya 0,29 Mbps dan trek audionya
ternyata **senyap digital** (−91,0 dB rata-rata *dan* puncak); ASR
menghasilkan 4 token halusinasi darinya. Berkasnya diekspor ulang pada
2026-08-16 dan sekarang berbunyi (−19,6 dB rata-rata, −0,9 dB puncak, 2,38
Mbps). Pelajarannya tetap berlaku untuk berkas baru mana pun: **ukur dulu
tingkat audionya sebelum hasil transkripsinya dipakai** — ASR tidak memberi
tanda apa pun saat ia mengarang dari senyap.

**Rekaman yang disalin manual** cukup dikosongkan kolom `tautan_sumber`-nya.
Barisnya tetap wajib ada; yang dilewati hanya pengunduhannya.

## Mengapa unduhan diperiksa panjangnya

Unduhan yang putus di tengah meninggalkan berkas yang *ada* tetapi pendek, dan
pemeriksaan `exists()` akan menerimanya sebagai lengkap — lalu ASR
mentranskripsikan potongan itu tanpa memberi tahu siapa pun bahwa rekamannya
terpotong. Karena itu berkas diunduh ke `.part` lebih dulu, dan baru diberi
nama aslinya setelah jumlah baitnya cocok dengan `Content-Length`.

Bila `.part` tertinggal, unduhannya gagal. Berkas itu sengaja tidak dihapus:
ia menunjukkan sampai berapa bait yang tiba. Jalankan ulang perintahnya.

## Cara memprosesnya

    python -m disfluency_id ingest --audio data/media/<nama-berkas>.mp4 \
        --out data/corpus/<nama>.jsonl

Perintah itu mentranskripsikan berkas dengan penanda waktu tingkat kata, lalu
menyimpannya sebagai korpus JSONL. **Seluruh token keluar berlabel `O`** —
transkripsi hanya menghasilkan kata dan waktunya; pelabelan disfluensi adalah
pekerjaan anotator manusia. Itu batas yang disengaja: kalau label disfluensi
ikut dihasilkan mesin, tidak ada lagi acuan kebenaran untuk mengukur mesin itu.

Setelah JSONL jadi:

    python -m disfluency_id detect  --jsonl data/corpus/<nama>.jsonl
    python -m disfluency_id explain --jsonl data/corpus/<nama>.jsonl
    python -m disfluency_id edl     --jsonl data/corpus/<nama>.jsonl \
        --source data/media/<nama-berkas>.mp4

## Berkas di sini tidak ikut dipublikasikan

Audio dan video mentah tidak dilampirkan ke dokumen penelitian. Yang
dipublikasikan cukup transkrip beranotasi beserta penanda waktunya.

**Keenam berkas bahan pihak ketiga: potongan podcast dari kanal YouTube.**
Tidak ada satu pun rekaman peneliti sendiri di sini, sehingga kewajiban lisensi
dan atribusi berlaku penuh untuk seluruhnya.

**Keenamnya berlisensi Standar YouTube** — diperiksa satu per satu pada halaman
videonya, tidak satu pun Creative Commons. Lisensi itu **tidak memberi hak
redistribusi**. Yang ada karena itu bukan izin, melainkan sandaran penggunaan
wajar: kutipan pendek dari karya panjang, untuk penelitian non-komersial,
dengan atribusi penuh. Kolom `izin` di `provenans.csv` menyatakan persis itu
dan tidak mengaku lebih.

Porsi kutipan yang kecil — 4,38% keseluruhan — adalah bagian penting dari
sandaran tersebut, dan karena itu disimpan sebagai angka yang dapat dihitung
ulang, bukan klaim.

> **Cermin CDN pada kolom `tautan_unduh` masih terbuka untuk umum.** Berkas di
> sana dapat diunduh siapa pun yang tahu tautannya, tanpa autentikasi — dan
> dengan lisensi Standar YouTube, itu berarti potongan karya orang lain
> tersedia untuk umum dari penyimpanan peneliti. Menyediakannya untuk diunduh
> siapa saja adalah hal yang berbeda dari memakainya untuk meneliti, dan
> sandaran penggunaan wajar menutupi yang kedua saja.
>
> Penyelesaiannya tidak menuntut penyuntingan ulang berkas apa pun: **jadikan
> bucket-nya privat lalu pakai URL bertanda tangan.** Isi `tautan_unduh`
> diganti URL bertanda tangan itu, dan seluruh rangkaian berjalan seperti
> sekarang.
>
> Selama cerminnya masih terbuka, jangan cantumkan tautan CDN-nya di dokumen
> yang diserahkan; kutip URL YouTube di `tautan_sumber`.

Bahan publik yang dikutip **tidak** disamarkan namanya. Aturan penyamaran pada
docstring `ingest.py` ditulis untuk rekaman privat; pada karya publik yang
dirujuk dengan URL-nya, menyamarkan penutur justru membuat provenansnya tidak
dapat diperiksa. Yang tetap harus dijaga adalah data pribadi pihak lain yang
kebetulan tersebut **di dalam** percakapan.

## Transkrip

Hasil transkripsi disimpan di `data/transkrip/crisperwhisper/`. Cuma ada satu
model, dan itu disengaja — alasannya ada di docstring `disfluency_id/asr.py`.

Medan `rentang_rusak` di dalam JSON mencatat rentang degenerasi bangkitan yang
dibuang. Itu audio **tanpa transkrip sah**, dan anotator harus tahu di mana
lubangnya. Periksa medan itu sebelum memakai transkripnya.

## Tempat penyimpanan bobot model

`transformers` mengunduh bobot ke cache HuggingFace, secara bawaan di
`~/.cache/huggingface` (`%USERPROFILE%\.cache\huggingface` di Windows) pada
cakram sistem. CrisperWhisper butuh sekitar 3 GB. Bila cakram sistem sempit,
unduhan gagal di tengah jalan dengan pesan yang menyebut kegagalan penyusunan
berkas — mudah disalahartikan sebagai kerusakan unduhan padahal sebabnya ruang
cakram.

Arahkan cache ke cakram yang lapang lewat `HF_HOME` sebelum menjalankan
transkripsi:

    # PowerShell
    $env:HF_HOME = "E:\hf-cache"

    # bash
    export HF_HOME=/e/hf-cache

Satu peubah itu cukup: ia memindahkan bobot model sekaligus cache potongan
`xet` yang dipakai saat mengunduh — mengarahkan bobotnya saja tidak memadai,
sebab penyusunan berkas sementara tetap menulis ke cakram sistem.

Di Colab hal ini tidak perlu diurus: bobotnya diunduh ulang tiap sesi, sebab
`/content` dihapus habis setiap kali sesi berakhir.
