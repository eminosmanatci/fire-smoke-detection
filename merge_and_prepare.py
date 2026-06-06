import os
import shutil
import random
import argparse

def merge_datasets(raw_dir, out_dir, val_ratio=0.1, seed=42):
    random.seed(seed)

    # Çıktı klasörlerini oluştur
    for split in ["train", "val"]:
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    # Tüm dataset klasörlerini al
    datasets = [os.path.join(raw_dir, d) for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    print(f"Found datasets: {datasets}")

    all_images = []

    for dataset in datasets:
        # dataset altındaki tüm alt klasörleri kontrol et (train/test/val)
        for subfolder in os.listdir(dataset):
            subfolder_path = os.path.join(dataset, subfolder)
            img_dir = os.path.join(subfolder_path, "images")
            lbl_dir = os.path.join(subfolder_path, "labels")

            if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
                continue  # Eğer images veya labels yoksa atla

            for img_file in os.listdir(img_dir):
                if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(img_dir, img_file)
                    lbl_path = os.path.join(lbl_dir, img_file.rsplit(".",1)[0] + ".txt")

                    if not os.path.exists(lbl_path):
                        # Label yoksa boş txt oluştur
                        open(lbl_path, 'w').close()

                    # Prefix ile isim çakışmasını önle
                    prefix = os.path.basename(dataset) + "_" + subfolder + "_"
                    new_img_name = prefix + img_file
                    new_lbl_name = prefix + img_file.rsplit(".",1)[0] + ".txt"

                    all_images.append((img_path, lbl_path, new_img_name, new_lbl_name))

    print(f"Total images found: {len(all_images)}")

    # Shuffle ve train/val ayır
    random.shuffle(all_images)
    val_count = int(len(all_images) * val_ratio)

    val_set = all_images[:val_count]
    train_set = all_images[val_count:]

    # Kopyala
    for img_path, lbl_path, new_img_name, new_lbl_name in train_set:
        shutil.copy(img_path, os.path.join(out_dir, "images", "train", new_img_name))
        shutil.copy(lbl_path, os.path.join(out_dir, "labels", "train", new_lbl_name))

    for img_path, lbl_path, new_img_name, new_lbl_name in val_set:
        shutil.copy(img_path, os.path.join(out_dir, "images", "val", new_img_name))
        shutil.copy(lbl_path, os.path.join(out_dir, "labels", "val", new_lbl_name))

    print(f"Train images: {len(train_set)}, Val images: {len(val_set)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="datasets/raw", help="Raw dataset folder")
    parser.add_argument("--out_dir", type=str, default="datasets/merged", help="Output merged dataset folder")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    merge_datasets(args.raw_dir, args.out_dir, args.val_ratio, args.seed)

if __name__ == "__main__":
    main()


