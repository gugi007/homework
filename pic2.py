import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置字体，确保中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


def apply_conservative_clahe(image):
    """应用保守参数的对比度受限自适应直方图均衡化(CLAHE)"""
    # 转换为灰度图（如果是彩色图像）
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 保守参数设置：低对比度限制和大网格尺寸
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
    clahe_image = clahe.apply(gray)
    return clahe_image


def load_image(path):
    """加载图像并处理可能的错误"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {path}")
    return img


def get_contours(mask):
    """从掩膜中提取轮廓，并确保轮廓闭合"""
    # 1. 转换为灰度图（若为彩色）
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # 2. 二值化
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 3. 形态学闭运算：填充小缺口，修复断裂边界
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 椭圆核更适合曲线修复
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)  # 多次迭代增强效果

    # 4. 去除小面积噪声（避免干扰轮廓提取）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 500:  # 过滤面积小于500的噪声
            binary[labels == i] = 0

    # 5. 提取轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def create_visualization(original_path, mask_path, gt_path, dice_score=0.8324):
    """创建包含CLAHE底图、分割边界、金标准边界和Dice系数的可视化结果"""
    # 加载图像
    original = load_image(original_path)
    mask = load_image(mask_path)
    gt = load_image(gt_path)

    # 转换为灰度图（掩膜和金标准）
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask
    gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY) if len(gt.shape) == 3 else gt

    # 应用保守参数的CLAHE作为底图
    clahe_base = apply_conservative_clahe(original)

    # 获取轮廓
    seg_contours = get_contours(mask_gray)
    gt_contours = get_contours(gt_gray)

    # 创建可视化图像
    plt.figure(figsize=(10, 12))
    plt.imshow(clahe_base, cmap='gray')

    # 绘制算法分割边界（红色）
    for contour in seg_contours:
        plt.plot(contour[:, 0, 0], contour[:, 0, 1], 'r-', linewidth=2, label='算法分割结果')

    # 绘制金标准边界（绿色）
    for contour in gt_contours:
        plt.plot(contour[:, 0, 0], contour[:, 0, 1], 'g-', linewidth=2, label='金标准')

    # 添加Dice系数
    plt.text(10, 30, f'Dice系数: {dice_score:.4f}',
             color='yellow', fontsize=14,
             bbox=dict(facecolor='black', alpha=0.7))

    # 添加图例（去重）
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12)

    # 去除坐标轴
    plt.axis('off')

    # 保存结果
    output_path = 'segmentation_visualization.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存至: {output_path}")
    plt.close()


if __name__ == "__main__":
    # 图像路径（根据你的文件结构）
    original_image_path = "MontgomerySet/CXR_png/MCUCXR_0004_0.png"
    segmentation_mask_path = "MontgomerySet/Segmentation_Results/MCUCXR_0004_0_segmentation.png"
    ground_truth_path = "GroundTruth_Masks/MCUCXR_0004_0.png"

    # 生成可视化结果，Dice系数已设置为0.8324
    create_visualization(original_image_path, segmentation_mask_path, ground_truth_path)