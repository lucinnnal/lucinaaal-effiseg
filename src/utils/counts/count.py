import os
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
import pandas as pd

class PixelCounter:
    def __init__(self, root_dir, save_csv_path="class_pixel_counts.csv", class_range=(0,19), device='cpu'):
        """
        root_dir: str, Cityscapes GT 폴더 경로 (예: gtFine/train)
        save_csv_path: str, 클래스별 픽셀 수 + weight를 저장할 CSV 파일 경로
        class_range: tuple, 클래스 인덱스 범위 (포함)
        device: str or torch.device, tensor 연산 디바이스
        """
        self.root_dir = root_dir
        self.save_csv_path = save_csv_path
        self.class_range = class_range
        self.device = device

        self.class_pixel_counts = defaultdict(int)
        self.class_weights = None

    def count_pixels(self):
        print(f"Counting pixels in {self.root_dir}...")
        for city in os.listdir(self.root_dir):
            city_path = os.path.join(self.root_dir, city)
            if not os.path.isdir(city_path):
                continue

            for filename in os.listdir(city_path):
                if not filename.endswith("_labelIds.png"):
                    continue

                filepath = os.path.join(city_path, filename)
                mask = np.array(Image.open(filepath))

                unique_classes, counts = np.unique(mask, return_counts=True)
                for cls, count in zip(unique_classes, counts):
                    if self.class_range[0] <= cls <= self.class_range[1]:
                        self.class_pixel_counts[cls] += count
        print("Counting done.")

    def load_pixel_counts_from_csv(self):
        if not os.path.exists(self.save_csv_path):
            raise FileNotFoundError(f"{self.save_csv_path} not found.")
        
        df = pd.read_csv(self.save_csv_path)
        self.class_pixel_counts = {int(row['class_id']): int(row['pixel_count']) for _, row in df.iterrows()}
        print(f"픽셀 수 로드 완료: {self.save_csv_path}")

    def save_pixel_counts_to_csv(self):
        class_ids = list(range(self.class_range[0], self.class_range[1]+1))
        pixel_counts = [self.class_pixel_counts.get(cls, 0) for cls in class_ids]

        df = pd.DataFrame({
            "class_id": class_ids,
            "pixel_count": pixel_counts
        })
        df.to_csv(self.save_csv_path, index=False)
        print(f"픽셀 수 저장 완료: {self.save_csv_path}")


    def compute_class_weights(self, beta=0.9):
        """
        Effective Number 기반 클래스별 weight 계산
        """
        counts_list = [self.class_pixel_counts.get(cls, 0) for cls in range(self.class_range[0], self.class_range[1]+1)]
        counts_tensor = torch.tensor(counts_list, dtype=torch.float32, device=self.device)

        effective_num = 1.0 - torch.pow(beta, counts_tensor)
        effective_num = torch.where(effective_num == 0, torch.tensor(1e-8, device=self.device), effective_num)
        weights = (1.0 - beta) / effective_num
        print("Class counts:", counts_tensor.tolist())
        print("Effective nums:", effective_num.tolist())
        print("Raw weights:", weights.tolist())
        weights = (1.0 - beta) / effective_num
        log_weights = torch.log(weights + 1e-8)
        norm_weights = log_weights / log_weights.sum() * len(log_weights)

        # weights = weights / weights.sum() * len(weights)  # normalize sum to num_classes
        self.class_weights = weights.cpu().numpy()  # numpy array for saving in CSV
        print("Computed class weights based on Effective Number of Samples.")
        return self.class_weights

    def save_as_csv(self):
        """
        클래스별 픽셀 수와 weight를 CSV로 저장
        컬럼: class_id, pixel_count, weight
        """
        class_ids = list(range(self.class_range[0], self.class_range[1]+1))
        pixel_counts = [self.class_pixel_counts.get(cls, 0) for cls in class_ids]
        weights = self.class_weights if self.class_weights is not None else [None]*len(class_ids)

        df = pd.DataFrame({
            "class_id": class_ids,
            "pixel_count": pixel_counts,
            "weight": weights
        })
        df.to_csv(self.save_csv_path, index=False)
        print(f"Saved class pixel counts and weights to CSV: {self.save_csv_path}")

    def load_from_csv(self):
        """
        CSV로부터 픽셀 수와 weight 불러오기
        """
        df = pd.read_csv(self.save_csv_path)
        self.class_pixel_counts = {row['class_id']: row['pixel_count'] for _, row in df.iterrows()}
        if 'weight' in df.columns:
            self.class_weights = torch.tensor(df['weight'].values, dtype=torch.float32, device=self.device)
        print(f"Loaded class pixel counts and weights from CSV: {self.save_csv_path}")

    def run(self, beta=0.9999, force_recompute=False):
        if os.path.exists(self.save_csv_path) and not force_recompute:
            self.load_pixel_counts_from_csv()
        else:
            self.count_pixels()
            self.save_pixel_counts_to_csv()

        self.compute_class_weights(beta)
        return self.class_weights

if __name__ == "__main__":
    ROOT_DIR = "/home/urp_jwl/.vscode-server/data/Effiseg/src/data/cityscapes/gtFine/train"
    pc = PixelCounter(ROOT_DIR, save_csv_path="pixel_counts.csv")

    # 첫 실행 또는 강제 재계산
    #weights = pc.run(beta=0.9999, force_recompute=True)

    # 이후 빠른 실행 (픽셀 수 재카운트 없음)
    weights = pc.run(beta=0.9)

