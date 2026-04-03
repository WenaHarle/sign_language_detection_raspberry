# Sign Language Detection for Raspberry Pi

Project ini berisi kolektor dan recognizer gesture tangan berbasis MediaPipe untuk:

- `single hand`
- `two hand`

Fitur utama:

- pencatatan sudut jari
- pencatatan orientasi telapak tangan
- dukungan gesture satu tangan dan dua tangan secara terpisah
- file dataset terpisah agar mode single-hand dan two-hand tidak tercampur

## Struktur File

- [collector_SINGLE_HAND.py](collector_SINGLE_HAND.py): rekam gesture satu tangan ke `gestures.json`
- [recog_single_hand.py](recog_single_hand.py): recognizer satu tangan
- [collector_TWO_HAND.py](collector_TWO_HAND.py): rekam gesture dua tangan ke `gestures_two_hand.json`
- [recog_two_hand.py](recog_two_hand.py): recognizer dua tangan
- [gestures.json](gestures.json): dataset gesture satu tangan
- [gestures_two_hand.json](gestures_two_hand.json): dataset gesture dua tangan
- `hand_landmarker.task`: model MediaPipe Hand Landmarker
- [requirements.txt](requirements.txt): dependency Python

## Setup Cepat di PC / Laptop

Buat virtual environment lalu install dependency:

```bash
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Download model `hand_landmarker.task` lalu taruh di folder project:

```bash
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Cara Menjalankan

### Single Hand

Collector:

```bash
python .\collector_SINGLE_HAND.py
```

Recognizer:

```bash
python .\recog_single_hand.py
```

### Two Hand

Collector:

```bash
python .\collector_TWO_HAND.py
```

Recognizer:

```bash
python .\recog_two_hand.py
```

## File Penyimpanan Gesture

Mode satu tangan dan dua tangan disimpan terpisah:

- single-hand -> `gestures.json`
- two-hand -> `gestures_two_hand.json`

Ini penting supaya data training dan recognizer tidak saling tercampur.

## Catatan Pemakaian

- untuk mode single-hand, sistem membaca bentuk jari dan orientasi telapak
- untuk mode two-hand, sistem membaca:
  - bentuk tangan kiri
  - bentuk tangan kanan
  - orientasi masing-masing tangan
  - relasi posisi antar kedua tangan
- kalau file JSON kosong atau belum ada, program akan otomatis memakai data kosong
- kalau webcam gagal dibuka, program akan menampilkan pesan error yang lebih jelas

## Setup Raspberry Pi

Bagian ini dipertahankan karena penting untuk Raspberry Pi, terutama saat perlu build Python 3.11 sendiri.

## 1. Install build dependencies

```bash
sudo apt update
sudo apt install -y \
build-essential wget curl git \
libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
libsqlite3-dev libffi-dev libncursesw5-dev xz-utils \
tk-dev libxml2-dev libxmlsec1-dev liblzma-dev \
libgdbm-dev libnss3-dev uuid-dev
```

## 2. Download Python 3.11 source

```bash
cd /usr/src
sudo wget https://www.python.org/ftp/python/3.11.10/Python-3.11.10.tgz
sudo tar xzf Python-3.11.10.tgz
cd Python-3.11.10
```

## 3. Build and install Python 3.11

```bash
sudo ./configure --enable-optimizations
sudo make -j4
sudo make altinstall
```

## 4. Verify installation

```bash
python3.11 --version
```

## 5. Create virtual environment

```bash
python3.11 -m venv ~/myenv
source ~/myenv/bin/activate
```

## 6. Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

## 7. Install core libraries

### NumPy

```bash
pip install numpy
```

### OpenCV

Coba install dari pip dulu:

```bash
pip install opencv-python
```

Kalau gagal:

```bash
sudo apt install python3-opencv
```

## 8. Install MediaPipe

Coba dulu:

```bash
pip install mediapipe
```

Kalau gagal di Raspberry Pi:

```bash
pip install mediapipe-rpi4
```

## Saran Workflow

1. rekam gesture dengan collector
2. cek isi file JSON hasil simpan
3. jalankan recognizer
4. kalau hasil masih kurang stabil, rekam ulang gesture dengan posisi tangan yang konsisten

