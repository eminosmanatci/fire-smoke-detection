from ultralytics import YOLO
import os
from pathlib import Path

# --- DEĞİŞKENLERİ BURADAN AYARLAYABİLİRSİNİZ ---

# 1. Modelinizin tam yolu
model_path = r"C:/Users/ACER/Desktop/AgroByte/Yangın_Modulu/FireAi/runs/train/fire_detection/weights/best.pt"

# 2. Test edilecek resmin tam yolu
# Dosya adını test_foto5.jpg veya test_foto4.jpeg olarak değiştirebilirsiniz.
source_image_path = r"C:/Users/ACER/Desktop/AgroByte/Yangın_Modulu/FireAi/test_foto2.jpg"

# 3. Sonuçların kaydedileceği klasörün tam yolu
save_directory = r"C:/Users/ACER/Desktop/AgroByte/Yangın_Modulu/FireAi/Sonuclar"

# 4. Modelin eğitildiği resim boyutu (genişlik ve yükseklik)
img_size = 640

# --- KOD BAŞLANGICI ---

# Çıktı klasörünün var olup olmadığını kontrol et, yoksa oluştur.
os.makedirs(save_directory, exist_ok=True)

# YOLO'nun 'project' ve 'name' parametrelerini doğru şekilde kullanabilmek için
# kaydetme yolunu ana klasör ve alt klasör olarak ikiye ayırıyoruz.
save_path_obj = Path(save_directory)
project_dir = str(save_path_obj.parent)
name_dir = save_path_obj.name

# YOLO modelini yükle
model = YOLO(model_path)

# Resim üzerinde tahmin yap ve sonucu belirtilen klasöre kaydet
results = model.predict(
    source=source_image_path,
    imgsz=img_size,
    save=True,
    project=project_dir,
    name=name_dir,
    exist_ok=True,
    show=True
)

# --- YENİ: Sonuçları terminale yazdırmak için eklenen bölüm ---

# predict() bir liste döndürür, tek resim işlediğimiz için ilk elemanı ([0]) alıyoruz.
result = results[0]

# Tespit edilen nesne sayısı
detection_count = len(result.boxes)
print(f"\n--- Tespit Sonuçları ---")
print(f"Toplam {detection_count} adet nesne tespit edildi.")

# Eğer en az bir nesne tespit edildiyse, detayları yazdır.
if detection_count > 0:
    print("\nDetaylar:")
    # Tespit edilen her bir kutu (nesne) için döngü oluştur.
    for box in result.boxes:
        # Güven skorunu al (örn: 0.92)
        confidence = box.conf.item()
        # Sınıf ID'sini al (örn: 0)
        class_id = int(box.cls.item())
        # Modelin bildiği sınıf isimlerinden ID'ye karşılık gelen ismi al (örn: 'fire')
        class_name = model.names[class_id]

        # Güven skorunu yüzde olarak formatla ve ekrana yazdır.
        print(f"  - Sınıf: {class_name}, Güven Oranı: {confidence:.2f} ({confidence:.0%})")

# İşlem sürelerini al ve milisaniye (ms) cinsinden yazdır.
speed = result.speed  # speed bir sözlüktür: {'preprocess': ..., 'inference': ..., 'postprocess': ...}
total_time = speed['preprocess'] + speed['inference'] + speed['postprocess']

print("\n--- İşlem Süresi ---")
print(f"Ön işleme (preprocess): {speed['preprocess']:.2f} ms")
print(f"Model çıkarımı (inference): {speed['inference']:.2f} ms")
print(f"Son işleme (postprocess): {speed['postprocess']:.2f} ms")
print(f"Toplam Süre: {total_time:.2f} ms")

# --- YENİ BÖLÜM SONU ---


# Kullanıcıya genel bilgi ver
output_file_name = Path(source_image_path).name
final_save_path = os.path.join(save_directory, output_file_name)

print("\nİşlem başarıyla tamamlandı!")
print(f"Giriş resmi {img_size}x{img_size} boyutuna ayarlanarak işlendi.")
print(f"Tespit sonuçları içeren resim şu yola kaydedildi: {final_save_path}")