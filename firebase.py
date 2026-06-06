# Gerekli kütüphaneleri içe aktarın
import firebase_admin
from firebase_admin import credentials, firestore
from ultralytics import YOLO
import datetime
import os
from pathlib import Path

# --- FİREBASE BAĞLANTI AYARLARI ---
# !!! --- DÜZELTİLMESİ GEREKEN SATIR BURASIYDI --- !!!
# Firebase'den indirdiğiniz özel anahtar dosyasının GERÇEK YOLU
cred = credentials.Certificate(
    r"C:\Users\ACER\Desktop\AgroByte\Yangın_Modulu\FireAi\agrobytevision-587c6-firebase-adminsdk-fbsvc-385622b046.json")

# Storage'a artık ihtiyacımız olmadığı için bucket yapılandırmasını kaldırdık.
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- MODEL VE AYARLAR ---
# Modelinizin tam yolu
model_path = r"C:\Users\ACER\Desktop\AgroByte\Yangın_Modulu\FireAi\runs\train\fire_detection\weights\best.pt"

# YOLO modelini yükle
try:
    model = YOLO(model_path)
    print("YOLO modeli başarıyla yüklendi.")
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")
    exit()


# Fonksiyonun adı ve işlevi güncellendi
def detect_fire_and_save_data(image_path):
    """
    Belirtilen resim yolunda yangın tespiti yapar ve tespit varsa
    verilerini Firebase Firestore'a kaydeder.
    """
    if not os.path.exists(image_path):
        print(f"Hata: Belirtilen resim yolu bulunamadı -> {image_path}")
        return

    print(f"\n'{Path(image_path).name}' resmi işleniyor...")

    # Model ile tahmin yap
    results = model.predict(source=image_path, imgsz=640,
                            verbose=False)  # Terminali temiz tutmak için verbose=False eklendi
    result = results[0]

    # Eğer en az bir nesne tespit edildiyse
    if len(result.boxes) > 0:
        print(f"{len(result.boxes)} adet nesne tespit edildi! Veriler Firestore'a kaydediliyor...")

        # Firestore'a kaydedilecek verileri hazırlayalım
        detected_objects = []
        for box in result.boxes:
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            class_name = model.names[class_id]

            # Tespit edilen her nesnenin bilgisini bir listeye ekle
            detected_objects.append({
                'class_name': class_name,
                'confidence': f"{confidence:.2f}"
            })
            print(f"  - Tespit: Sınıf={class_name}, Güven={confidence:.2f}")

        # Zaman damgasını belge ID'si olarak kullan
        timestamp_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        try:
            # Firestore'a kayıt ekle
            doc_ref = db.collection('detections').document(timestamp_id)
            doc_ref.set({
                'timestamp': firestore.SERVER_TIMESTAMP,
                'location': 'Kamera 1 - Sera A',  # Örnek konum bilgisi
                'detection_count': len(detected_objects),
                'detections': detected_objects  # Tespit edilen nesnelerin listesi
            })
            print("Tespit bilgisi başarıyla Firestore'a kaydedildi.")

        except Exception as e:
            print(f"Firestore'a kayıt sırasında bir hata oluştu: {e}")
    else:
        print("İşlenen resimde yangın tespit edilmedi.")


# --- ANA PROGRAM ---
if __name__ == "__main__":
    # Test edilecek resmin tam yolu
    test_image_path = r'C:\Users\ACER\Desktop\AgroByte\Yangın_Modulu\FireAi\test_foto2.jpg'

    # Ana fonksiyonu çağır
    detect_fire_and_save_data(test_image_path)