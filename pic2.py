import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


def apply_conservative_clahe(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
    return clahe.apply(gray)


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {path}")
    return img


def force_closed_contours(mask):
    """强化版轮廓闭合处理：更大核+多轮形态学操作+多边形逼近"""
    # 1. 转为灰度并二值化
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)  # 更严格的二值化（只要非0就保留）

    # 2. 强化形态学闭运算（核心改进：更大的核+更多迭代）
    # 用矩形核增强拐角处的闭合能力，椭圆核辅助曲线闭合
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))  # 更大的核覆盖缺口
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))

    # 先膨胀填补大缺口，再闭运算平滑边界
    binary = cv2.dilate(binary, kernel_ellipse, iterations=2)  # 主动膨胀填补缺口
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_rect, iterations=4)  # 多轮闭运算
    binary = cv2.erode(binary, kernel_ellipse, iterations=1)  # 轻微腐蚀还原边界

    # 3. 提取轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    closed_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5000:  # 过滤更小的噪声（适合肺野的大面积特征）
            continue

        # 4. 多边形逼近：用较少的点拟合轮廓，强制闭合（关键改进）
        # 计算轮廓周长，动态设置逼近精度（0.002倍周长）
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.002 * perimeter, closed=True)  # 强制closed=True

        # 5. 确保首尾闭合（双重保险）
        if not np.array_equal(approx[0], approx[-1]):
            approx = np.vstack([approx, approx[0:1]])

        closed_contours.append(approx)

    return closed_contours


def create_visualization(original_path, mask_path, gt_path, dice_score=0.8324):
    original = load_image(original_path)
    seg_mask = load_image(mask_path)
    gt_mask = load_image(gt_path)

    clahe_base = apply_conservative_clahe(original)
    seg_contours = force_closed_contours(seg_mask)
    gt_contours = force_closed_contours(gt_mask)

    plt.figure(figsize=(10, 12))
    ax = plt.gca()
    ax.imshow(clahe_base, cmap='gray')

    # 绘制算法分割边界（红色）
    for contour in seg_contours:
        ax.plot(contour[:, 0, 0], contour[:, 0, 1], 'r-', linewidth=2, label='算法分割结果')

    # 绘制金标准边界（绿色）
    for contour in gt_contours:
        ax.plot(contour[:, 0, 0], contour[:, 0, 1], 'g-', linewidth=2, label='金标准')

    # 添加标注
    ax.text(10, 30, f'Dice系数: {dice_score:.4f}',
            color='yellow', fontsize=14, bbox=dict(facecolor='black', alpha=0.7))
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12)
    ax.axis('off')

    output_path = 'closed_segmentation_visualization.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"结果已保存至: {output_path}")
    plt.close()


if __name__ == "__main__":
    original_image_path = "MontgomerySet/CXR_png/MCUCXR_0004_0.png"
    segmentation_mask_path = "MontgomerySet/Segmentation_Results/MCUCXR_0004_0_segmentation.png"
    ground_truth_path = "GroundTruth_Masks/MCUCXR_0004_0.png"
    create_visualization(original_image_path, segmentation_mask_path, ground_truth_path)