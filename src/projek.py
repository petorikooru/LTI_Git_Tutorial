import cv2
import numpy as np
from os import listdir, path, mkdir
import os
import sys

# --- KONFIGURASI DAN INICIALISASI GLOBAL ---
# Klasifier Haar Cascade untuk deteksi wajah. Menggunakan path standar OpenCV.
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mapping Label (ID numerik) ke Nama Orang. Akan diisi saat pelatihan.
NAMES = {}

# Folder utama untuk semua dataset wajah
DATA_DIR = './faces/'
if not path.exists(DATA_DIR):
    try:
        mkdir(DATA_DIR)
    except OSError as e:
        print(f"ERROR: Gagal membuat direktori {DATA_DIR}. Pastikan Anda memiliki izin tulis.")
        sys.exit(1)

# Variabel global untuk menyimpan model yang sudah dilatih
trained_model = None

# --- FUNGSI DETEKSI WAJAH DAN EKSTRAKSI (DATA COLLECTION) ---
def face_extractor(img):
    """
    Mendeteksi wajah menggunakan Haar Cascade dan mengembalikan wajah yang sudah di-crop.
    """
    # Pastikan frame tidak kosong
    if img is None:
        return None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Parameter deteksi: skala faktor 1.3, min Neighbors 5
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None
        
    # Ambil wajah pertama yang terdeteksi
    (x,y,w,h) = faces[0]
    cropped_face = img[y:y+h, x:x+w]
    return cropped_face

# --- BAGIAN 1: PENGAMBILAN DATA BARU ---
def collect_new_user_data():
    """
    Meminta nama dan mengambil 100 sampel wajah untuk pengguna baru.
    """
    person_name = input("Masukkan NAMA Anda (Contoh: Budi, Sella): ").strip()
    if not person_name:
        print("Nama tidak boleh kosong.")
        return

    # Buat direktori khusus untuk pengguna ini
    user_data_path = path.join(DATA_DIR, person_name)
    if not path.exists(user_data_path):
        os.makedirs(user_data_path)
    
    print(f"\n--- Memulai pengambilan sampel untuk {person_name} ---")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Gagal membuka kamera. Periksa apakah kamera sedang digunakan oleh aplikasi lain.")
        return
        
    count = 0
    MAX_SAMPLES = 100 # Batasi jumlah sampel yang diambil

    while True:
        ret, frame = cap.read()
        if not ret:
             print("Gagal membaca frame dari kamera.")
             break
             
        face = face_extractor(frame)
        
        if face is not None:
            count += 1
            
            # Resize dan konversi ke Grayscale (standar untuk LBPH)
            face = cv2.resize(face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # Simpan file di folder nama orang
            file_name_path = path.join(user_data_path, f"{count}.jpg")
            cv2.imwrite(file_name_path, face)
            
            # Tampilkan di jendela
            cv2.putText(face, str(count), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1)
            cv2.imshow(f'Data Collection: {person_name}', face)
        else:
            cv2.putText(frame, "Wajah tidak ditemukan", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
            cv2.imshow(f'Data Collection: {person_name}', frame)
        
        # Selesai saat menekan Enter (13) atau sampel mencapai 100
        if cv2.waitKey(1) == 13 or count == MAX_SAMPLES:
            break

    cap.release()
    cv2.destroyAllWindows()    
    print(f"\nPengambilan sampel untuk {person_name} Selesai. Total: {count} sampel.")


# --- BAGIAN 2: PELATIHAN MODEL (TRAINING) ---
def train_model():
    """
    Melatih model LBPH dengan semua data yang ada di folder 'faces/'.
    """
    global NAMES
    Training_Data, Labels = [], []
    current_label_id = 0
    NAMES = {}

    print("\n--- Memulai Pelatihan Model ---")
    
    # Iterasi melalui setiap folder (setiap orang) di DATA_DIR
    for name in listdir(DATA_DIR):
        person_dir = path.join(DATA_DIR, name)
        
        # Pastikan itu adalah folder
        if path.isdir(person_dir):
            NAMES[current_label_id] = name # Simpan mapping ID ke Nama
            
            # Iterasi melalui semua file gambar di folder orang tersebut
            for filename in listdir(person_dir):
                if filename.endswith(".jpg"):
                    image_path = path.join(person_dir, filename)
                    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    
                    if images is not None:
                        # Kumpulkan data pelatihan dan label ID
                        Training_Data.append(np.asarray(images, dtype=np.uint8))
                        Labels.append(current_label_id)
            
            current_label_id += 1

    if not Training_Data:
        print("ERROR: Tidak ada data wajah ditemukan untuk dilatih! Silakan ambil sampel terlebih dahulu (Opsi 1).")
        return None
        
    # Konversi ke NumPy Array
    Labels = np.asarray(Labels, dtype=np.int32)
    
    # Inisialisasi dan latih model LBPH
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    
    print("Model berhasil dilatih.")
    print(f"Mapping Label: {NAMES}")
    return model

# --- BAGIAN 3: PENGUJIAN DAN PENGENALAN REAL-TIME (MODIFIKASI: HANYA NAMA & CONFIDENCE) ---
def run_face_recognition(model):
    """
    Menjalankan pengenalan wajah real-time menggunakan model yang sudah dilatih, 
    hanya menampilkan Nama dan Confidence.
    """
    if model is None:
        print("Model belum dilatih atau pelatihan gagal. Silakan latih model terlebih dahulu.")
        return

    print("\n--- Memulai Pengenalan Wajah Real-Time ---")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Gagal membuka kamera. Pastikan kamera berfungsi.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi wajah di frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            # Gambar kotak
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            
            # Ekstrak ROI (Region of Interest) dan resize untuk prediksi
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (200, 200))

            try:
                # Prediksi menggunakan model
                label_id, confidence = model.predict(roi_gray)
                
                # Menggunakan mapping NAMES global untuk mendapatkan nama
                predicted_name = NAMES.get(label_id, "Unknown") 
                
                # Hitung 'kepercayaan' (semakin rendah confidence LBPH, semakin yakin)
                # Ambang batas 400 adalah nilai umum untuk normalisasi
                current_confidence = 100 * (1 - (confidence) / 400) 
                
                # Tentukan warna teks berdasarkan Confidence
                text_color = (0, 0, 255) # Merah (Unknown/rendah)
                if current_confidence > 75:
                    text_color = (0, 255, 0) # Hijau (High confidence)
                elif current_confidence > 50:
                     text_color = (0, 255, 255) # Kuning (Medium confidence)
                
                display_string = f"{predicted_name} | {int(current_confidence)}% Confident"
                
                # Tampilkan Nama dan Confidence di atas kotak wajah
                cv2.putText(frame, display_string, (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, text_color, 2)

            except:
                # Teks jika prediksi gagal
                cv2.putText(frame, "Deteksi Gagal", (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 2)

        cv2.imshow('Face Recognition System', frame)
        
        if cv2.waitKey(1) == 13: # Tekan Enter (13) untuk keluar
            break

    cap.release()
    cv2.destroyAllWindows()

# --- ALUR UTAMA PROGRAM ---
def main_menu():
    global trained_model
    
    # Coba latih model segera setelah program dimulai jika sudah ada data
    try:
        if any(path.isdir(path.join(DATA_DIR, d)) for d in listdir(DATA_DIR)):
            print("Mencoba memuat model dari data yang sudah ada...")
            trained_model = train_model()
    except Exception as e:
        print(f"Gagal memuat model awal: {e}")

    while True:
        print("\n==============================")
        print("  SISTEM PENGENALAN WAJAH LBPH")
        print("==============================")
        print("1. Ambil Data Wajah Baru (Tambah Orang)")
        print("2. Latih Ulang Model (Wajib Setelah Data Baru!)")
        print("3. Jalankan Pengenalan Wajah")
        print("4. Keluar")
        
        choice = input("Pilih Opsi (1-4): ")
        
        if choice == '1':
            collect_new_user_data()
        elif choice == '2':
            trained_model = train_model()
        elif choice == '3':
            if trained_model is None:
                print("Model belum dilatih atau pelatihan gagal. Silakan pilih opsi 2 terlebih dahulu.")
            else:
                run_face_recognition(trained_model)
        elif choice == '4':
            print("Program dihentikan.")
            break
        else:
            print("Pilihan tidak valid. Silakan coba lagi.")

# Jalankan Menu Utama
if __name__ == "__main__":
    main_menu()
