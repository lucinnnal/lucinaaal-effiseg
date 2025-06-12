import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Class IOUs
data = {
    'Baseline': [0.40537, 0.89775, 0.38665, 0.5054, 0.61489, 0.51943, 0.57216, 0.9218,
                 0.76158, 0.96742, 0.42449, 0.47425, 0.67761, 0.31166, 0.41971,
                 0.89132, 0.62304, 0.87894, 0.63301],
    'KD': [0.50985, 0.90512, 0.64523, 0.65375, 0.64461, 0.55659, 0.59747, 0.91996,
             0.79396, 0.97326, 0.46247, 0.52543, 0.69328, 0.47496, 0.49725,
             0.90925, 0.64808, 0.89486, 0.66246],
    'Dice-Focal Loss': [0.30795, 0.8943, 0.38665, 0.31183, 0.61517, 0.53784, 0.57167, 0.9167,
             0.75117, 0.96583, 0.4314, 0.46934, 0.68213, 0.38551, 0.45461,
             0.89049, 0.61329, 0.87269, 0.64389],
    'DyT': [0.3509, 0.89715, 0.50346, 0.26896, 0.60238, 0.51475, 0.58125, 0.91997,
             0.76173, 0.96734, 0.41127, 0.48433, 0.67363, 0.30398, 0.37921,
             0.89471, 0.64243, 0.87624, 0.6194]
}

categories = ['wall', 'vegetation', 'truck', 'train', 'traffic_sign', 'traffic_light',
              'terrain', 'sky', 'sidewalk', 'road', 'rider', 'pole', 'person',
              'motorcycle', 'fence', 'car', 'bus', 'building', 'bicycle']

# color define for each class
category_colors = [
    '#8B4513',  # wall - brown
    '#228B22',  # vegetation - green  
    '#FF6600',  # truck - orange
    '#000080',  # train - navy blue
    '#FFD700',  # traffic_sign - gold
    '#FF69B4',  # traffic_light - hot pink
    '#9ACD32',  # terrain - yellow green
    '#87CEEB',  # sky - sky blue
    '#FF1493',  # sidewalk - deep pink
    '#696969',  # road - dim gray
    '#DC143C',  # rider - crimson
    '#CD853F',  # pole - peru
    '#9370DB',  # person - medium purple
    '#32CD32',  # motorcycle - lime green
    '#A0522D',  # fence - sienna
    '#FF0000',  # car - red
    '#4169E1',  # bus - royal blue
    '#8A2BE2',  # building - blue violet
    '#00CED1'   # bicycle - dark turquoise
]

def create_radar_chart_with_colored_segments(data, categories, figsize=(12, 10)):
    # 각도 계산
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # 플롯 설정
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    # 각 카테고리별로 가장 바깥쪽 테두리만 색칠
    for i in range(N):
        # 각 구역의 시작과 끝 각도
        start_angle = angles[i] - (2 * pi / N) / 2
        end_angle = angles[i] + (2 * pi / N) / 2

        # 바깥쪽 도넛 구간의 반지름 범위 (예: 0.95~1.0)
        r_outer = 1.0
        r_inner = 0.92

        theta = np.linspace(start_angle, end_angle, 100)
        r1 = np.full_like(theta, r_outer)
        r2 = np.full_like(theta, r_inner)

        # 도넛형 구역을 채우기
        ax.fill_between(theta, r2, r1, color=category_colors[i], alpha=0.9, linewidth=0)
    
    # 데이터 플롯
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # baseline: blue, cskd: orange, diceloss: green, tanh: red
    markers = ['o', 's', 'D', '^']  # baseline: circle, cskd: square, diceloss: diamond, tanh: triangle

    for i, (label, values) in enumerate(data.items()):
        plot_values = values + values[:1]
        ax.plot(angles, plot_values, marker=markers[i], linewidth=2.5, 
               label=label, color=colors[i], markersize=6, markeredgewidth=1.5,
               markeredgecolor='white')
        ax.fill(angles, plot_values, alpha=0.1, color=colors[i])
    
    # 카테고리 라벨 설정
    ax.set_xticks([])  # 기존 라벨 제거

    for angle, label in zip(angles[:-1], categories):
        deg = np.degrees(angle) % 360
        # 좌우(오른쪽: 0±15도, 왼쪽: 180±15도)는 더 멀리, 나머지는 더 가깝게
        if (deg < 20 or deg > 340) or (160 < deg < 200):
            radius = 1.12
        else:
            radius = 1.09
        ax.text(
            angle, radius, label,
            fontsize=10, fontweight='bold',
            ha='center', va='center'
        )
    
    # 반지름 축 설정
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    
    # 범례
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)
    
    plt.tight_layout()
    return fig, ax

def create_comparison_plot():
    """
    색상 구역이 있는 비교 레이더 차트 생성
    """
    fig, ax = create_radar_chart_with_colored_segments(
        data, categories)
    return fig

def print_data_table():
    """
    원본 데이터를 테이블 형식으로 출력
    """
    print("Original Data Table:")
    print("-" * 100)
    print(f"{'Category':<15} {'Baseline':<10} {'CSKD':<10} {'Color':<15}")
    print("-" * 100)
    
    for i, category in enumerate(categories):
        baseline_val = data['Baseline'][i]
        cskd_val = data['KD'][i]
        color = category_colors[i]
        print(f"{category:<15} {baseline_val:<10.5f} {cskd_val:<10.5f} {color:<15}")
    
    print("-" * 100)

# 실행 예제
if __name__ == "__main__":
    print_data_table()
    fig = create_comparison_plot()
    fig.savefig('./cls_ious.png', dpi=300, bbox_inches='tight')
    plt.show()