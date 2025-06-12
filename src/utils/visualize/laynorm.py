import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from configs.train_arguements import get_arguments
from src.data.get_dataset import get_dataset
from src.models.get_model import get_model

# 전역 리스트에 LayerNorm 입출력 저장
ln_inputs_outputs = []

# 1. LayerNorm Hook 등록 함수
def register_ln_hooks(model):
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            input_tensor = input[0].detach().cpu().flatten().numpy()
            output_tensor = output.detach().cpu().flatten().numpy()
            ln_inputs_outputs.append((name, input_tensor, output_tensor))
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            hooks.append(module.register_forward_hook(get_hook(name)))

    if not hooks:
        print("⚠️ No LayerNorm modules found.")
    return hooks

# 2. 시각화 함수 - 모든 LayerNorm 레이어 포함
def plot_ln_io(ln_inputs_outputs, max_layers=None):
    if not ln_inputs_outputs:
        print("⚠️ No LayerNorm data found!")
        return
    
    print(f"📊 Found {len(ln_inputs_outputs)} LayerNorm layers:")
    for name, _, _ in ln_inputs_outputs:
        print(f"  - {name}")
    
    # max_layers가 지정되지 않으면 모든 레이어 시각화
    n = len(ln_inputs_outputs) if max_layers is None else min(len(ln_inputs_outputs), max_layers)
    
    # 그리드 레이아웃 계산
    cols = min(4, n)
    rows = (n + cols - 1) // cols if cols > 0 else 1

    plt.figure(figsize=(cols * 4, rows * 3))
    
    for i in range(n):
        name, x, y = ln_inputs_outputs[i]
        
        # 데이터 샘플링 (너무 많은 점은 성능 저하)
        max_points = 10000000
        if len(x) > max_points:
            indices = np.random.choice(len(x), max_points, replace=False)
            x_sampled = x[indices]
            y_sampled = y[indices]
        else:
            x_sampled = x
            y_sampled = y
        
        plt.subplot(rows, cols, i + 1)
        
        # 산점도 그리기
        plt.scatter(x_sampled, y_sampled, s=0.2, alpha=0.5, color='blue')
        
        # 추세선 추가 (binning 방식)
        if len(x_sampled) > 100:
            n_bins = 50
            x_min, x_max = x_sampled.min(), x_sampled.max()
            x_bins = np.linspace(x_min, x_max, n_bins)
            
            y_means = []
            x_centers = []
            
            for j in range(len(x_bins) - 1):
                mask = (x_sampled >= x_bins[j]) & (x_sampled < x_bins[j+1])
                if np.sum(mask) > 5:  # 충분한 데이터가 있는 경우만
                    y_means.append(np.mean(y_sampled[mask]))
                    x_centers.append((x_bins[j] + x_bins[j+1]) / 2)
            
            if len(x_centers) > 3:
                plt.plot(x_centers, y_means, 'red', linewidth=2, alpha=0.8)
        
        # 동적 축 범위 설정 (실제 데이터의 최소~최대값)
        x_min, x_max = x_sampled.min(), x_sampled.max()
        y_min, y_max = y_sampled.min(), y_sampled.max()
        
        # 약간의 여백 추가 (5%)
        x_margin = (x_max - x_min) * 0.05 if x_max != x_min else 0.1
        y_margin = (y_max - y_min) * 0.05 if y_max != y_min else 0.1
        
        plt.xlim(x_min - x_margin, x_max + x_margin)
        plt.ylim(y_min - y_margin, y_max + y_margin)
        
        # 축 틱을 데이터 범위에 맞게 설정
        if x_max != x_min:
            x_ticks = np.linspace(x_min, x_max, 5)
            plt.xticks(x_ticks, [f'{tick:.1f}' for tick in x_ticks])
        
        if y_max != y_min:
            y_ticks = np.linspace(y_min, y_max, 5)
            plt.yticks(y_ticks, [f'{tick:.1f}' for tick in y_ticks])
        
        # 레이어 이름을 더 간결하게 표시
        display_name = name.split('.')[-2:] if '.' in name else name
        display_name = '.'.join(display_name) if isinstance(display_name, list) else display_name
        
        plt.title(display_name, fontsize=8)
        plt.xlabel("LN input", fontsize=8)
        plt.ylabel("LN output", fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 데이터 범위 정보 출력
        print(f"Layer {name}:")
        print(f"  Input range: [{x_min:.2f}, {x_max:.2f}]")
        print(f"  Output range: [{y_min:.2f}, {y_max:.2f}]")
        print(f"  Data points: {len(x_sampled)}")
    
    plt.suptitle(f"All LayerNorm Layers - Input vs Output\n({n} layers visualized)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("all_layernorm_plot.png", dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved as all_layernorm_plot.png")
    plt.show()

# 3. 레이어별 세부 분석 함수 추가
def analyze_ln_statistics(ln_inputs_outputs):
    """각 LayerNorm 레이어의 통계 분석"""
    print("\n📈 LayerNorm Layer Statistics:")
    print("=" * 80)
    
    for i, (name, x, y) in enumerate(ln_inputs_outputs):
        print(f"\n[{i+1:2d}] {name}")
        print(f"     Input  - Mean: {np.mean(x):7.3f}, Std: {np.std(x):7.3f}, Range: [{np.min(x):7.3f}, {np.max(x):7.3f}]")
        print(f"     Output - Mean: {np.mean(y):7.3f}, Std: {np.std(y):7.3f}, Range: [{np.min(y):7.3f}, {np.max(y):7.3f}]")
        
        # 정규화 효과 측정
        input_var = np.var(x)
        output_var = np.var(y)
        var_reduction = (input_var - output_var) / input_var * 100 if input_var > 0 else 0
        print(f"     Variance reduction: {var_reduction:5.1f}%")

# 4. 가중치 로드 함수
def load_pretrained_weights(model, checkpoint_path):
    """
    사전 훈련된 가중치를 모델에 로드합니다.
    """
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        return model
    
    try:
        print(f"📂 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # checkpoint 구조 확인
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 모델에 가중치 로드
        model.load_state_dict(state_dict, strict=False)
        print("✅ Pretrained weights loaded successfully!")
        
        # 로드된 정보 출력
        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                print(f"📊 Epoch: {checkpoint['epoch']}")
            if 'best_miou' in checkpoint:
                print(f"📊 Best mIoU: {checkpoint['best_miou']:.4f}")
                
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print("Continuing with randomly initialized weights...")
    
    return model

# 5. 통합 실행 함수
def profile_ln_behavior_in_transformer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")

    # 이미지 1개 샘플 불러오기
    train_dataset, _ = get_dataset(args)
    sample = train_dataset[0]
    image = sample['pixel_values'] if isinstance(sample, dict) else sample

    # Transform 적용
    if not isinstance(image, torch.Tensor):
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
        ])
        image = transform(image)

    input_tensor = image.unsqueeze(0).to(device) if image.dim() == 3 else image.to(device)
    print(f"📥 Input tensor shape: {input_tensor.shape}")

    # 모델 로딩
    model = get_model(args).to(device)
    
    # 사전 훈련된 가중치 로드
    checkpoint_path = "/home/urp_jwl/.vscode-server/data/EffiSeg-lucinaaal/ckpt/segformerb2_teacher_cityscapes.pth"
    model = load_pretrained_weights(model, checkpoint_path)
    
    model.eval()
    
    # Hook 등록
    hooks = register_ln_hooks(model)
    print(f"🔗 Registered {len(hooks)} LayerNorm hooks")

    # Forward pass (hook 작동)
    print("🔄 Running forward pass...")
    with torch.no_grad():
        output = model(input_tensor)
        print(f"📤 Output shape: {output.shape if hasattr(output, 'shape') else type(output)}")

    # Hook 제거
    for h in hooks:
        h.remove()

    print(f"📊 Collected {len(ln_inputs_outputs)} LayerNorm input-output pairs")
    
    # 통계 분석
    if ln_inputs_outputs:
        analyze_ln_statistics(ln_inputs_outputs)
        
        # 시각화 (모든 레이어)
        plot_ln_io(ln_inputs_outputs)
    else:
        print("⚠️ No LayerNorm data collected for visualization")

# 6. 명령행 인자 파싱
def parse_args():
    parser = argparse.ArgumentParser(description='LayerNorm Analysis with Pretrained Weights')
    parser.add_argument('--model', type=str, default='SegformerB2', 
                       help='Model name (default: SegformerB2)')
    parser.add_argument('--checkpoint', type=str, 
                       default='/home/urp_jwl/.vscode-server/data/EffiSeg-lucinaaal/ckpt/segformerb2_teacher_cityscapes.pth',
                       help='Path to checkpoint file')
    parser.add_argument('--max_layers', type=int, default=None,
                       help='Maximum number of layers to visualize (default: all)')
    
    return parser.parse_args()

# 7. 실행
if __name__ == '__main__':
    # 명령행 인자 파싱
    cmd_args = parse_args()
    
    # 기본 설정 가져오기
    args = get_arguments()
    
    # 명령행 인자로 모델 이름 오버라이드
    if hasattr(args, 'model'):
        args.model = cmd_args.model
    elif hasattr(args, 'model_name'):
        args.model_name = cmd_args.model
    else:
        # args에 model 속성이 없으면 추가
        setattr(args, 'model', cmd_args.model)
    
    print(f"🚀 Starting LayerNorm analysis with model: {cmd_args.model}")
    print(f"📁 Checkpoint path: {cmd_args.checkpoint}")
    if cmd_args.max_layers:
        print(f"🔢 Max layers to visualize: {cmd_args.max_layers}")
    else:
        print("🔢 Visualizing all LayerNorm layers")
    
    # 전역 리스트 초기화 (재실행 시를 위해)
    ln_inputs_outputs.clear()
    
    profile_ln_behavior_in_transformer(args)