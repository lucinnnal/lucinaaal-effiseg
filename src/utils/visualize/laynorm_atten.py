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
        if isinstance(module, nn.LayerNorm) and 'attn' in name:
            hooks.append(module.register_forward_hook(get_hook(name)))
            print(f"🔗 Registered hook for: {name}")

    if not hooks:
        print("⚠️ No LayerNorm modules containing 'attn' found.")
        print("📋 Available LayerNorm modules:")
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                print(f"   - {name}")
    return hooks

# 2. 시각화 함수
def plot_ln_io(ln_inputs_outputs, max_layers=12):
    n = min(len(ln_inputs_outputs), max_layers)
    cols = 4
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(cols * 4, rows * 3))
    for i in range(n):
        name, x, y = ln_inputs_outputs[i]
        plt.subplot(rows, cols, i + 1)
        plt.scatter(x[:100000000000], y[:100000000000], s=0.2, alpha=0.5, color='blue')  # 최대 1000개 점만
        plt.title(name, fontsize=8)
        plt.xlabel("LN input")
        plt.ylabel("LN output")
    plt.suptitle("LayerNorm output vs. input (attn layers only)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("ln_io_plot_attn_layers.png", dpi=300)
    print("✅ Plot saved as ln_io_plot_attn_layers.png")
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

# 4. 통합 실행 함수
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
    
    # 시각화
    if ln_inputs_outputs:
        plot_ln_io(ln_inputs_outputs)
    else:
        print("⚠️ No LayerNorm data collected for visualization")

# 5. 명령행 인자 파싱
def parse_args():
    parser = argparse.ArgumentParser(description='LayerNorm Analysis with Pretrained Weights')
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
    
    print(f"🚀 Starting LayerNorm analysis with model: {cmd_args.model}")
    print(f"📁 Checkpoint path: {cmd_args.checkpoint}")
    
    # 전역 리스트 초기화 (재실행 시를 위해)
    ln_inputs_outputs.clear()
    
    profile_ln_behavior_in_transformer(args)