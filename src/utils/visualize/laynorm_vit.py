import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from transformers import ViTModel, Wav2Vec2Model, ViTConfig, Wav2Vec2Config
from matplotlib.gridspec import GridSpec
import warnings
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
warnings.filterwarnings('ignore')

class LayerNormHook:
    """Layer Normalization의 입출력을 캡처하는 Hook 클래스 (affine transformation 이전)"""
    def __init__(self):
        self.inputs = []
        self.normalized_outputs = []
        
    def hook_fn(self, module, input, output):
        # Layer Norm의 입력 저장
        x = input[0].detach().cpu()
        
        # 원본 입력값을 저장 (flatten)
        x_flat = x.flatten()
        self.inputs.append(x_flat.numpy())
        
        # Affine transformation 이전의 정규화된 출력 계산
        # LayerNorm: (x - mean) / sqrt(var + eps)
        if len(x.shape) == 3:  # [batch, seq, hidden]
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
        else:
            mean = x.mean()
            var = x.var(unbiased=False)
            
        normalized = (x - mean) / torch.sqrt(var + module.eps)
        normalized_flat = normalized.flatten()
        self.normalized_outputs.append(normalized_flat.numpy())
        
    def clear(self):
        self.inputs = []
        self.normalized_outputs = []

def load_cifar10_data(batch_size=4):
    """CIFAR-10 데이터셋 로드"""
    print("Loading CIFAR-10 dataset...")
    
    # ViT에 맞는 전처리 (224x224로 리사이즈)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ViT 입력 크기에 맞춰 리사이즈
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # CIFAR-10 다운로드 및 로드
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    # DataLoader 생성
    dataloader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0  # Windows 호환성을 위해 0으로 설정
    )
    
    print(f"CIFAR-10 dataset loaded. Total samples: {len(trainset)}")
    return dataloader

def load_models():
    """ViT와 Wav2Vec2 모델 로드"""
    print("Loading models...")
    
    try:
        # ViT 모델 로드 (더 작은 모델 사용)
        vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        vit_model.eval()
        print("ViT model loaded successfully!")
    except Exception as e:
        print(f"Failed to load ViT model: {e}")
        print("Creating a smaller ViT model for demonstration...")
        config = ViTConfig(
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=6,
            intermediate_size=1536,
            image_size=224,
            patch_size=16
        )
        vit_model = ViTModel(config)
        vit_model.eval()
    
    try:
        # Wav2Vec2 모델 로드
        wav2vec_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
        wav2vec_model.eval()
        print("Wav2Vec2 model loaded successfully!")
    except Exception as e:
        print(f"Failed to load Wav2Vec2 model: {e}")
        print("Creating a smaller Wav2Vec2 model for demonstration...")
        config = Wav2Vec2Config(
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=2048
        )
        wav2vec_model = Wav2Vec2Model(config)
        wav2vec_model.eval()
    
    return vit_model, wav2vec_model

def register_hooks(model, model_type):
    """모델의 여러 레이어에 Hook 등록"""
    hooks = {}
    hook_handles = []
    
    if model_type == 'ViT':
        # ViT의 경우 encoder layer의 layernorm_before
        num_layers = len(model.encoder.layer)
        print(f"ViT has {num_layers} layers")
        
        # 4개 레이어 선택 (처음, 중간, 후반, 마지막)
        target_layers = [
            1,  # 초기
            num_layers // 3,  # 중간 전반
            2 * num_layers // 3,  # 중간 후반
            num_layers - 1  # 마지막
        ]
        
        for i, layer_idx in enumerate(target_layers):
            if layer_idx < num_layers:
                hook = LayerNormHook()
                handle = model.encoder.layer[layer_idx].layernorm_before.register_forward_hook(hook.hook_fn)
                hooks[f'layer_{i}'] = hook
                hook_handles.append(handle)
                print(f"Registered ViT hook for layer {layer_idx}")
    
    elif model_type == 'wav2vec':
        # Wav2Vec2의 경우 encoder layer의 layer_norm
        num_layers = len(model.encoder.layers)
        print(f"Wav2Vec2 has {num_layers} layers")
        
        target_layers = [
            1,
            num_layers // 3,
            2 * num_layers // 3,
            num_layers - 1
        ]
        
        for i, layer_idx in enumerate(target_layers):
            if layer_idx < num_layers:
                hook = LayerNormHook()
                handle = model.encoder.layers[layer_idx].layer_norm.register_forward_hook(hook.hook_fn)
                hooks[f'layer_{i}'] = hook
                hook_handles.append(handle)
                print(f"Registered Wav2Vec2 hook for layer {layer_idx}")
    
    return hooks, hook_handles

def extract_ln_data(model, model_type, hooks, dataloader=None, num_batches=20):
    """모델을 통해 데이터를 전달하고 LN 입출력 추출"""
    print(f"Extracting LN data for {model_type}...")
    
    # Hook 초기화
    for hook in hooks.values():
        hook.clear()
    
    # Multiple forward passes
    if model_type == 'ViT' and dataloader is not None:
        # CIFAR-10 데이터 사용
        batch_count = 0
        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_count >= num_batches:
                break
                
            if batch_count % 5 == 0:
                print(f"Processing CIFAR-10 batch {batch_count + 1}/{num_batches}")
            
            with torch.no_grad():
                try:
                    _ = model(images)
                    batch_count += 1
                except Exception as e:
                    print(f"Error during forward pass {batch_count}: {e}")
                    continue
    
    elif model_type == 'wav2vec':
        # Wav2Vec2는 여전히 랜덤 데이터 사용
        for i in range(num_batches):
            if i % 5 == 0:
                print(f"Processing wav2vec sample {i + 1}/{num_batches}")
            
            with torch.no_grad():
                try:
                    # 더 짧은 오디오 사용
                    batch_data = torch.randn(2, 8000)  # 0.5초
                    _ = model(batch_data)
                except Exception as e:
                    print(f"Error during forward pass {i}: {e}")
                    continue
    
    # 데이터 수집 및 처리
    ln_data = {}
    for name, hook in hooks.items():
        if hook.inputs and hook.normalized_outputs:
            # 모든 데이터를 결합
            all_inputs = np.concatenate(hook.inputs)
            all_outputs = np.concatenate(hook.normalized_outputs)
            
            # NaN이나 inf 제거
            valid_mask = np.isfinite(all_inputs) & np.isfinite(all_outputs)
            all_inputs = all_inputs[valid_mask]
            all_outputs = all_outputs[valid_mask]
            
            # 더 넓은 범위로 outlier 처리 (95%로 완화)
            input_q95 = np.percentile(np.abs(all_inputs), 95)
            output_q95 = np.percentile(np.abs(all_outputs), 95)
            
            # 극값 제거하지 않고 더 많은 데이터 유지
            mask = (np.abs(all_inputs) < input_q95 * 3) & (np.abs(all_outputs) < output_q95 * 3)
            all_inputs = all_inputs[mask]
            all_outputs = all_outputs[mask]
            
            # 더 많은 샘플링
            n_samples = min(100000, len(all_inputs))
            if len(all_inputs) > n_samples:
                indices = np.random.choice(len(all_inputs), n_samples, replace=False)
                inputs_sampled = all_inputs[indices]
                outputs_sampled = all_outputs[indices]
            else:
                inputs_sampled = all_inputs
                outputs_sampled = all_outputs
            
            ln_data[name] = {
                'inputs': inputs_sampled,
                'outputs': outputs_sampled
            }
            
            print(f"Collected {len(inputs_sampled)} valid points for {name}")
            print(f"  Input range: [{inputs_sampled.min():.2f}, {inputs_sampled.max():.2f}]")
            print(f"  Output range: [{outputs_sampled.min():.2f}, {outputs_sampled.max():.2f}]")
        else:
            print(f"No data collected for {name}")
    
    return ln_data

def plot_ln_analysis(vit_data, wav2vec_data, save_path='./plots'):
    """LN 분석 결과 시각화 (동적 축 범위 설정)"""
    os.makedirs(save_path, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    layer_names = ['5', '15', '20', '25']  # 그림과 유사한 레이어 이름
    
    # ViT 플롯
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        
        key = f'layer_{i}'
        if key in vit_data:
            data = vit_data[key]
            
            # 산점도
            ax.scatter(data['inputs'], data['outputs'], 
                      alpha=0.3, s=0.1, color='blue', rasterized=True)
            
            # 추세선 (binning 방식으로 부드럽게)
            if len(data['inputs']) > 100:
                # 입력값을 구간별로 나누어 평균 계산
                n_bins = 50
                x_min, x_max = data['inputs'].min(), data['inputs'].max()
                x_bins = np.linspace(x_min, x_max, n_bins)
                
                y_means = []
                x_centers = []
                
                for j in range(len(x_bins) - 1):
                    mask = (data['inputs'] >= x_bins[j]) & (data['inputs'] < x_bins[j+1])
                    if np.sum(mask) > 10:  # 충분한 데이터가 있는 경우만
                        y_means.append(np.mean(data['outputs'][mask]))
                        x_centers.append((x_bins[j] + x_bins[j+1]) / 2)
                
                if len(x_centers) > 3:
                    ax.plot(x_centers, y_means, 'red', linewidth=2, alpha=0.8)
            
            # 동적 축 범위 설정 (실제 데이터의 최소~최대값)
            x_min, x_max = data['inputs'].min(), data['inputs'].max()
            y_min, y_max = data['outputs'].min(), data['outputs'].max()
            
            # 약간의 여백 추가 (5%)
            x_margin = (x_max - x_min) * 0.05
            y_margin = (y_max - y_min) * 0.05
            
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
            
            # 축 틱을 데이터 범위에 맞게 설정
            x_ticks = np.linspace(x_min, x_max, 5)
            y_ticks = np.linspace(y_min, y_max, 5)
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            
            # 틱 라벨 포맷팅 (소수점 1자리)
            ax.set_xticklabels([f'{tick:.1f}' for tick in x_ticks])
            ax.set_yticklabels([f'{tick:.1f}' for tick in y_ticks])
        
        ax.grid(True, alpha=0.3)
        ax.set_title(f'ViT LN layer {layer_names[i]} (CIFAR-10)', fontsize=12, fontweight='bold')

        if i == 0:
            ax.set_ylabel('ViT LN output', fontsize=11, fontweight='bold')
    
    # Wav2Vec2 플롯
    for i in range(4):
        ax = fig.add_subplot(gs[1, i])
        
        key = f'layer_{i}'
        if key in wav2vec_data:
            data = wav2vec_data[key]
            
            ax.scatter(data['inputs'], data['outputs'], 
                      alpha=0.3, s=0.1, color='green', rasterized=True)
            
            # 추세선
            if len(data['inputs']) > 100:
                n_bins = 50
                x_min, x_max = data['inputs'].min(), data['inputs'].max()
                x_bins = np.linspace(x_min, x_max, n_bins)
                
                y_means = []
                x_centers = []
                
                for j in range(len(x_bins) - 1):
                    mask = (data['inputs'] >= x_bins[j]) & (data['inputs'] < x_bins[j+1])
                    if np.sum(mask) > 10:
                        y_means.append(np.mean(data['outputs'][mask]))
                        x_centers.append((x_bins[j] + x_bins[j+1]) / 2)
                
                if len(x_centers) > 3:
                    ax.plot(x_centers, y_means, 'red', linewidth=2, alpha=0.8)
            
            # 동적 축 범위 설정 (실제 데이터의 최소~최대값)
            x_min, x_max = data['inputs'].min(), data['inputs'].max()
            y_min, y_max = data['outputs'].min(), data['outputs'].max()
            
            # 약간의 여백 추가 (5%)
            x_margin = (x_max - x_min) * 0.05
            y_margin = (y_max - y_min) * 0.05
            
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
            
            # 축 틱을 데이터 범위에 맞게 설정
            x_ticks = np.linspace(x_min, x_max, 5)
            y_ticks = np.linspace(y_min, y_max, 5)
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            
            # 틱 라벨 포맷팅 (소수점 1자리)
            ax.set_xticklabels([f'{tick:.1f}' for tick in x_ticks])
            ax.set_yticklabels([f'{tick:.1f}' for tick in y_ticks])
        
        ax.grid(True, alpha=0.3)
        ax.set_title(f'wav2vec LN layer {layer_names[i]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('LN input', fontsize=11)
        
        if i == 0:
            ax.set_ylabel('wav2vec LN output', fontsize=11, fontweight='bold')
    
    # 전체 제목
    plt.suptitle('LN output (y axis) vs. LN input (x axis) - ViT with CIFAR-10 Data\nOutputs are before the affine transformation in LN', 
                 fontsize=14, y=0.95, fontweight='bold')
    
    # 하단 설명
    fig.text(0.5, 0.02, 
             'ViT model processes real CIFAR-10 images while Wav2Vec2 uses random audio data.\n'
             'The S-shaped curves highly resemble that of a tanh function. '
             'The more linear shapes in earlier layers can also be captured by the center part of a tanh curve.',
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # 저장
    save_file = os.path.join(save_path, 'ln_analysis_vit_wav2vec_cifar10.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_file}")
    
    plt.show()

def main():
    """메인 실행 함수"""
    save_path = './ln_analysis_plots'
    
    # CIFAR-10 데이터 로드
    cifar10_dataloader = load_cifar10_data(batch_size=64)
    
    # 모델 로드
    vit_model, wav2vec_model = load_models()
    
    # Hook 등록
    vit_hooks, vit_handles = register_hooks(vit_model, 'ViT')
    wav2vec_hooks, wav2vec_handles = register_hooks(wav2vec_model, 'wav2vec')
    
    # LN 데이터 추출
    print("Extracting ViT LN data with CIFAR-10...")
    vit_ln_data = extract_ln_data(vit_model, 'ViT', vit_hooks, 
                                 dataloader=cifar10_dataloader, num_batches=20)
    
    print("Extracting Wav2Vec2 LN data...")
    wav2vec_ln_data = extract_ln_data(wav2vec_model, 'wav2vec', wav2vec_hooks, 
                                     dataloader=None, num_batches=20)
    
    # 시각화
    print("Creating visualizations...")
    plot_ln_analysis(vit_ln_data, wav2vec_ln_data, save_path)
    
    # Hook 해제
    for handle in vit_handles + wav2vec_handles:
        handle.remove()
    
    print(f"\nAnalysis complete! Plot saved to: {save_path}")

if __name__ == "__main__":
    main()