from ultralytics import YOLO
import torch

def main():
    # GPU kullanılabilirliği kontrol
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Eğitim başlatılıyor... Kullanılan cihaz: {device}")

    # Önceden eğitilmiş YOLOv8n modeli yükle
    model = YOLO("yolov8n.pt")

    # Modeli eğit
    model.train(
        data="data.yaml",       # Dataset yaml dosyası
        epochs=25,              # Eğitim epoch sayısı
        batch=16,               # Batch size
        imgsz=640,              # Görsel boyutu
        device=device,          # GPU/CPU seçimi
        project="runs/train",   # runs/train klasörüne kaydet
        name="fire_detection",  # Model ismi
        patience=10             # Early stopping
    )

    print("✅ Eğitim tamamlandı. Sonuçlar 'runs/train/fire_detection' içinde.")

if __name__ == "__main__":
    main()


# yolo task=detect mode=predict model="C:/Users/ACER/Desktop/AgroByte/Yangın_Modulu/FireAi/runs/train/fire_detection/weights/best.pt" source="test_video1.mp4" show=True
yolo task=detect mode=predict model="C:/Users/ACER/Desktop/AgroByte/Yangın_Modulu/FireAi/runs/train/fire_detection/weights/best.pt" source="C:/Users/ACER/Desktop/AgroByte/Yangın_Modulu/FireAi/test_foto5.jpg" show=True

