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

# 1. Mlp 관련 LayerNorm Hook 등록 함수
def register_mlp_ln_hooks(model):
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            input_tensor = input[0].detach().cpu().flatten().numpy()
            output_tensor = output.detach().cpu().flatten().numpy()
            ln_inputs_outputs.append((name, input_tensor, output_tensor))
        return hook

    # Mlp 관련 LayerNorm만 필터링
    mlp_ln_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            # Mlp와 관련된 LayerNorm인지 확인
            # Block 내부의 norm2가 Mlp 앞에 오는 LayerNorm입니다
            if 'norm2' in name or 'mlp' in name.lower():
                hooks.append(module.register_forward_hook(get_hook(name)))
                mlp_ln_count += 1
                print(f"🔗 Registered hook for Mlp LayerNorm: {name}")

    if mlp_ln_count == 0:
        print("⚠️ No Mlp-related LayerNorm modules found.")
        print("📋 Available LayerNorm modules:")
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                print(f"  - {name}")
    else:
        print(f"✅ Total Mlp LayerNorm hooks registered: {mlp_ln_count}")
    
    return hooks

# 2. 시각화 함수 (Mlp LayerNorm 전용)
def plot_mlp_ln_io(ln_inputs_outputs, max_layers=16):
    if not ln_inputs_outputs:
        print("⚠️ No Mlp LayerNorm data to plot")
        return
    
    n = min(len(ln_inputs_outputs), max_layers)
    cols = 4
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(cols * 4, rows * 3))
    for i in range(n):
        name, x, y = ln_inputs_outputs[i]
        plt.subplot(rows, cols, i + 1)
        
        # 샘플링: 너무 많은 점이 있으면 일부만 선택
        max_points = 500000
        if len(x) > max_points:
            indices = np.random.choice(len(x), max_points, replace=False)
            x_sample, y_sample = x[indices], y[indices]
        else:
            x_sample, y_sample = x, y
        
        plt.scatter(x_sample, y_sample, s=0.5, alpha=0.6, color='red')
        plt.title(f"Mlp {name}", fontsize=10)
        plt.xlabel("LN input")
        plt.ylabel("LN output")
        plt.grid(True, alpha=0.3)
        
        # 통계 정보 표시
        plt.text(0.05, 0.95, f'Input: μ={x.mean():.3f}, σ={x.std():.3f}\nOutput: μ={y.mean():.3f}, σ={y.std():.3f}', 
                transform=plt.gca().transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle("Mlp LayerNorm: Output vs Input Analysis", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("mlp_ln_io_plot.png", dpi=300, bbox_inches='tight')
    print("✅ Mlp LayerNorm plot saved as mlp_ln_io_plot.png")
    plt.show()

# 3. 가중치 로드 함수
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

# 4. Mlp LayerNorm 분석 함수
def analyze_mlp_ln_behavior(args):
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
    
    # Mlp LayerNorm Hook 등록
    hooks = register_mlp_ln_hooks(model)

    # Forward pass (hook 작동)
    print("🔄 Running forward pass for Mlp LayerNorm analysis...")
    with torch.no_grad():
        output = model(input_tensor)
        print(f"📤 Output shape: {output.shape if hasattr(output, 'shape') else type(output)}")

    # Hook 제거
    for h in hooks:
        h.remove()

    print(f"📊 Collected {len(ln_inputs_outputs)} Mlp LayerNorm input-output pairs")
    
    
    # 시각화
    if ln_inputs_outputs:
        plot_mlp_ln_io(ln_inputs_outputs)
    else:
        print("⚠️ No Mlp LayerNorm data collected for visualization")

# 5. 명령행 인자 파싱
def parse_args():
    parser = argparse.ArgumentParser(description='Mlp LayerNorm Analysis with Pretrained Weights')
    parser.add_argument('--model', type=str, default='SegformerB2', 
                       help='Model name (default: SegformerB2)')
    parser.add_argument('--checkpoint', type=str, 
                       default='/home/urp_jwl/.vscode-server/data/EffiSeg-lucinaaal/ckpt/segformerb2_teacher_cityscapes.pth',
                       help='Path to checkpoint file')
    
    return parser.parse_args()

# 6. 실행
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
    
    print(f"🚀 Starting Mlp LayerNorm analysis with model: {cmd_args.model}")
    print(f"📁 Checkpoint path: {cmd_args.checkpoint}")
    
    # 전역 리스트 초기화 (재실행 시를 위해)
    ln_inputs_outputs.clear()
    
    analyze_mlp_ln_behavior(args)