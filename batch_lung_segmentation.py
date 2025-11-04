import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import binary_fill_holes
from skimage.morphology import convex_hull_image
from sklearn.metrics import jaccard_score
import time

# 设置字体家族和字体大小
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先使用SimHei，备用英文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
plt.rcParams['font.size'] = 10  # 统一设置字体大小


def load_image(image_path):
    """加载图像并转换为灰度图"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray


def load_ground_truth(image_path, gt_dir="GroundTruth_Masks"):
    """加载对应的金标准掩膜"""
    # 获取原始图像的文件名（不含路径）
    image_filename = os.path.basename(image_path)

    # 构建金标准文件的完整路径
    gt_path = os.path.join(gt_dir, image_filename)

    # 加载金标准掩膜
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    if gt is None:
        raise FileNotFoundError(f"无法加载金标准掩膜: {gt_path}")

    # 确保金标准是二值图像
    _, gt_binary = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)

    return gt_binary


def clahe_enhancement(gray, clip_limit=2.0, tile_grid_size=(12, 12)):
    """对比度受限的自适应直方图均衡化(CLAHE)"""
    # 先进行高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 使用优化的参数设置
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(blurred)

    # 生成边缘掩码，识别并抑制高亮的身体外轮廓噪声
    _, edge_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=2)
    enhanced = cv2.bitwise_and(enhanced, ~edge_mask)  # 去除高亮边缘区域

    return enhanced


def compare_enhancement_methods(original, clahe):
    """比较原始图像和CLAHE增强效果"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('原始图像')
    axes[0].axis('off')

    axes[1].imshow(clahe, cmap='gray')
    axes[1].set_title('CLAHE增强')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('enhancement_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def coarse_segmentation(enhanced):
    """粗分割：使用阈值分割获取二值图，并应用胸腔掩码"""
    img_h, img_w = enhanced.shape  # 获取图像尺寸

    # 1. 基础阈值分割：将肺野（暗区）转为亮区，背景转为暗区
    _, binary = cv2.threshold(enhanced, 110, 255, cv2.THRESH_BINARY_INV)

    # 2. 生成胸腔区域掩码：手动定义梯形胸腔区域，排除身体外轮廓
    chest_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    # 梯形顶点坐标（根据胸片解剖学比例估算）
    top_left = (int(img_w * 0.3), int(img_h * 0.1))
    top_right = (int(img_w * 0.7), int(img_h * 0.1))
    bottom_left = (int(img_w * 0.1), int(img_h * 0.9))
    bottom_right = (int(img_w * 0.9), int(img_h * 0.9))
    chest_vertices = np.array([[top_left, top_right, bottom_right, bottom_left]], dtype=np.int32)
    cv2.fillPoly(chest_mask, chest_vertices, 255)  # 填充胸腔区域为白色

    # 应用胸腔掩码，仅保留胸腔内的区域
    binary = cv2.bitwise_and(binary, chest_mask)

    # 3. 形态学开运算：先腐蚀后膨胀，去除小噪声点
    kernel_small = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)

    return binary, chest_mask


def locate_lung_regions(binary):
    """区域定位：找到左右肺的粗略位置"""
    # 查找连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        raise ValueError("无法找到连通区域，请检查分割参数")

    # 按面积排序连通区域（排除背景）
    areas = stats[1:, cv2.CC_STAT_AREA]
    sorted_indices = np.argsort(areas)[::-1]

    # 创建一个掩码来保存所有可能的肺区域
    potential_lung_mask = np.zeros_like(binary)

    # 选择面积较大的前N个区域（增加数量以提高找到肺的概率）
    num_regions_to_consider = min(5, len(sorted_indices))

    for i in range(num_regions_to_consider):
        region_idx = sorted_indices[i] + 1
        region_area = areas[sorted_indices[i]]

        # 计算区域的宽高比（肺通常是宽大于高的区域）
        width = stats[region_idx, cv2.CC_STAT_WIDTH]
        height = stats[region_idx, cv2.CC_STAT_HEIGHT]
        aspect_ratio = width / height if height > 0 else 0

        # 计算区域的中心位置（肺通常位于图像中央）
        center_x = stats[region_idx, cv2.CC_STAT_LEFT] + width // 2
        center_y = stats[region_idx, cv2.CC_STAT_TOP] + height // 2
        image_center_x = binary.shape[1] // 2
        distance_from_center = abs(center_x - image_center_x)

        # 根据区域特性筛选可能的肺区域
        # 1. 面积不能太小
        # 2. 宽高比应大于0.5（肺通常是横向延伸的）
        # 3. 不能太靠近图像边缘
        if (region_area > 5000 and
                aspect_ratio > 0.5 and
                center_y > binary.shape[0] * 0.2 and
                center_y < binary.shape[0] * 0.8):
            potential_lung_mask[labels == region_idx] = 255

    # 再次查找连通区域，这次只考虑潜在的肺区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(potential_lung_mask, connectivity=8)

    if num_labels <= 1:
        # 如果没有找到合适的区域，尝试使用面积最大的区域
        if len(sorted_indices) >= 1:
            main_region_idx = sorted_indices[0] + 1
            left_lung_mask = np.uint8(labels == main_region_idx) * 255
            right_lung_mask = np.zeros_like(binary)

            # 获取边界框
            left_bbox = stats[main_region_idx, cv2.CC_STAT_LEFT], stats[main_region_idx, cv2.CC_STAT_TOP], \
                stats[main_region_idx, cv2.CC_STAT_WIDTH], stats[main_region_idx, cv2.CC_STAT_HEIGHT]
            right_bbox = (0, 0, 0, 0)

            print("警告：只找到一个可能的肺区域")
            return left_lung_mask, right_lung_mask, left_bbox, right_bbox
        else:
            raise ValueError("无法找到肺区域，请检查分割参数")

    # 按面积排序连通区域
    areas = stats[1:, cv2.CC_STAT_AREA]
    sorted_indices = np.argsort(areas)[::-1]

    left_lung_idx = sorted_indices[0] + 1

    # 创建左右肺的掩码
    left_lung_mask = np.uint8(labels == left_lung_idx) * 255

    # 如果有第二个区域，将其作为右肺
    if len(sorted_indices) >= 2:
        right_lung_idx = sorted_indices[1] + 1
        right_lung_mask = np.uint8(labels == right_lung_idx) * 255

        # 获取边界框
        left_bbox = stats[left_lung_idx, cv2.CC_STAT_LEFT], stats[left_lung_idx, cv2.CC_STAT_TOP], \
            stats[left_lung_idx, cv2.CC_STAT_WIDTH], stats[left_lung_idx, cv2.CC_STAT_HEIGHT]
        right_bbox = stats[right_lung_idx, cv2.CC_STAT_LEFT], stats[right_lung_idx, cv2.CC_STAT_TOP], \
            stats[right_lung_idx, cv2.CC_STAT_WIDTH], stats[right_lung_idx, cv2.CC_STAT_HEIGHT]
    else:
        # 如果只有一个区域，尝试将其分为左右两部分
        right_lung_mask = np.zeros_like(binary)

        # 获取左肺边界框
        left_bbox = stats[left_lung_idx, cv2.CC_STAT_LEFT], stats[left_lung_idx, cv2.CC_STAT_TOP], \
            stats[left_lung_idx, cv2.CC_STAT_WIDTH], stats[left_lung_idx, cv2.CC_STAT_HEIGHT]
        right_bbox = (0, 0, 0, 0)

        print("警告：只找到一个肺区域，可能需要调整分割参数")

    return left_lung_mask, right_lung_mask, left_bbox, right_bbox


def refine_segmentation(enhanced, lung_mask, bbox):
    """精细分割：在肺区域内进行更精细的分割"""
    # 如果边界框无效，直接返回原始掩码
    if bbox[2] == 0 or bbox[3] == 0:
        return lung_mask

    x, y, w, h = bbox

    # 扩展边界框以包含更多上下文
    expand_pixels = 30
    x_expanded = max(0, x - expand_pixels)
    y_expanded = max(0, y - expand_pixels)
    w_expanded = min(enhanced.shape[1] - x_expanded, w + 2 * expand_pixels)
    h_expanded = min(enhanced.shape[0] - y_expanded, h + 2 * expand_pixels)

    # 提取感兴趣区域
    roi = enhanced[y_expanded:y_expanded + h_expanded, x_expanded:x_expanded + w_expanded]
    roi_mask = lung_mask[y_expanded:y_expanded + h_expanded, x_expanded:x_expanded + w_expanded]

    # 使用多个阈值进行分割，然后合并结果
    # 1. 自适应阈值
    adaptive_thresh1 = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 8)
    adaptive_thresh2 = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 21, 12)

    # 2. Otsu阈值
    _, otsu_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. 手动阈值（根据图像特性调整）
    _, manual_thresh = cv2.threshold(roi, 90, 255, cv2.THRESH_BINARY_INV)

    # 合并多种阈值分割结果
    combined_thresh = cv2.bitwise_or(adaptive_thresh1, adaptive_thresh2)
    combined_thresh = cv2.bitwise_or(combined_thresh, otsu_thresh)
    combined_thresh = cv2.bitwise_or(combined_thresh, manual_thresh)

    # 结合粗分割结果
    refined_roi = cv2.bitwise_and(combined_thresh, roi_mask)

    # 形态学操作进一步优化
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined_roi = cv2.morphologyEx(refined_roi, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 创建完整大小的精细分割结果
    refined = np.zeros_like(lung_mask)
    refined[y_expanded:y_expanded + h_expanded, x_expanded:x_expanded + w_expanded] = refined_roi

    return refined


def morphological_optimization(lung_mask):
    """形态学优化：填充空洞和平滑边界"""
    # 如果掩码为空，直接返回
    if np.sum(lung_mask) == 0:
        return lung_mask

    # 填充空洞
    filled = binary_fill_holes(lung_mask // 255).astype(np.uint8) * 255

    # 使用不同大小的闭运算核进行多次操作，填充不同大小的空洞
    closing_kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    closing_kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))

    closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, closing_kernel1, iterations=1)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, closing_kernel2, iterations=1)

    # 使用较小的开运算核平滑边界，避免过度侵蚀
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, opening_kernel, iterations=1)

    return opened


def repair_heart_boundary(lung_mask):
    """心脏边界修复：使用凸包算法修复肺内侧边界"""
    # 如果掩码为空，直接返回
    if np.sum(lung_mask) == 0:
        return lung_mask

    # 转换为二值数组
    binary = (lung_mask // 255).astype(bool)

    try:
        # 计算凸包
        convex_hull = convex_hull_image(binary)

        # 转换回uint8
        repaired = convex_hull.astype(np.uint8) * 255

        # 找到原始边界
        contours, _ = cv2.findContours(lung_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 创建原始边界的掩码
            boundary_mask = np.zeros_like(lung_mask)
            cv2.drawContours(boundary_mask, contours, -1, 255, 2)

            # 结合凸包和原始边界，但保留更多原始边界信息
            # 只在原始边界内部应用凸包修复
            inner_mask = cv2.bitwise_not(cv2.morphologyEx(lung_mask, cv2.MORPH_ERODE,
                                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                                                          iterations=2))

            # 只在内部区域应用凸包修复
            convex_hull_inner = cv2.bitwise_and(repaired, inner_mask)

            # 结合原始边界和凸包修复
            result = cv2.bitwise_or(lung_mask, convex_hull_inner)

            return result
        else:
            return repaired
    except Exception as e:
        print(f"边界修复过程中出现错误: {str(e)}")
        return lung_mask


def optimize_with_morphology_and_connectivity(thresh, chest_mask):
    """使用形态学操作和连通性分析优化分割结果"""
    img_h, img_w = thresh.shape[:2]

    # 1. 形态学开运算：先腐蚀后膨胀，去除小噪声点
    kernel_small = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # 2. 连通域分析与筛选：基于肺野的面积和位置特征过滤无效区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening, connectivity=8)
    valid_labels = []

    # 设置肺野面积阈值（根据图像大小动态调整）
    min_area = img_h * img_w * 0.04  # 肺野最小面积阈值（占图像4%）
    max_area = img_h * img_w * 0.35  # 肺野最大面积阈值（占图像35%）

    for i in range(1, num_labels):  # 0为背景，从1开始遍历连通域
        area = stats[i, cv2.CC_STAT_AREA]  # 连通域面积
        cx, cy = centroids[i]  # 连通域质心坐标

        # 筛选条件：面积在合理范围 + 质心位于胸腔中上部 + 左右肺分别位于中线两侧
        if (min_area < area < max_area and
                cy < img_h * 0.7 and
                ((cx < img_w * 0.45) or (cx > img_w * 0.55))):
            valid_labels.append(i)

    # 极端情况处理：若没有符合条件的连通域，强制保留最大的连通域
    if not valid_labels and num_labels > 1:
        valid_labels.append(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)

    # 生成候选掩膜：将筛选后的连通域标记为肺野
    candidate_mask = np.zeros_like(opening)
    for label in valid_labels:
        candidate_mask[labels == label] = 255

    # 3. 形态学闭运算：先膨胀后腐蚀，填充肺野内部的小空洞
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # 椭圆核更贴合肺野形状
    closing = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    # 4. 再次应用胸腔掩码：确保最终结果无身体外轮廓残留
    final_mask = cv2.bitwise_and(closing, chest_mask)

    return final_mask


def calculate_dice_coefficient(segmentation, ground_truth):
    """计算Dice相似系数"""
    # 确保输入是二值图像
    seg = (segmentation // 255).astype(bool)
    gt = (ground_truth // 255).astype(bool)

    # 计算交集和并集
    intersection = np.logical_and(seg, gt).sum()
    union = seg.sum() + gt.sum()

    # 计算Dice系数
    if union == 0:
        return 1.0
    else:
        return 2.0 * intersection / union


def visualize_results(original, clahe, segmentation, ground_truth=None, dice_score=None):
    """可视化分割结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 原始图像
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')

    # CLAHE增强图像
    axes[0, 1].imshow(clahe, cmap='gray')
    axes[0, 1].set_title('CLAHE增强')
    axes[0, 1].axis('off')

    # 分割结果
    axes[1, 0].imshow(original, cmap='gray')
    axes[1, 0].imshow(segmentation, cmap='jet', alpha=0.3)
    axes[1, 0].set_title('肺野分割结果')
    axes[1, 0].axis('off')

    # 对比图
    axes[1, 1].imshow(original, cmap='gray')

    # 绘制分割结果边界
    contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        axes[1, 1].plot(contour[:, 0, 0], contour[:, 0, 1], 'r-', linewidth=2, label='分割结果')

    # 如果有金标准，绘制金标准边界
    if ground_truth is not None:
        gt_contours, _ = cv2.findContours(ground_truth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in gt_contours:
            axes[1, 1].plot(contour[:, 0, 0], contour[:, 0, 1], 'g-', linewidth=2, label='金标准')

    axes[1, 1].set_title('分割结果对比')
    axes[1, 1].axis('off')

    # 添加图例和Dice系数
    if dice_score is not None:
        plt.figtext(0.5, 0.01, f'Dice系数: {dice_score:.4f}', ha='center', fontsize=12)

    # 移除重复的图例
    handles, labels = axes[1, 1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper center',
               bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)

    plt.tight_layout()
    plt.savefig('segmentation_results.png', dpi=300, bbox_inches='tight')
    plt.close()


def debug_visualization(original, clahe, binary, left_lung, right_lung, final, ground_truth):
    """创建调试用的可视化结果，显示所有中间步骤"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

    # 原始图像
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')

    # CLAHE增强图像
    axes[0, 1].imshow(clahe, cmap='gray')
    axes[0, 1].set_title('CLAHE增强')
    axes[0, 1].axis('off')

    # 粗分割结果
    axes[1, 0].imshow(binary, cmap='gray')
    axes[1, 0].set_title('粗分割结果')
    axes[1, 0].axis('off')

    # 左右肺区域
    combined_lungs = cv2.bitwise_or(left_lung, right_lung)
    axes[1, 1].imshow(combined_lungs, cmap='gray')
    axes[1, 1].set_title('左右肺区域')
    axes[1, 1].axis('off')

    # 最终分割结果
    axes[2, 0].imshow(final, cmap='gray')
    axes[2, 0].set_title('最终分割结果')
    axes[2, 0].axis('off')

    # 金标准
    axes[2, 1].imshow(ground_truth, cmap='gray')
    axes[2, 1].set_title('金标准')
    axes[2, 1].axis('off')

    plt.tight_layout()
    plt.savefig('debug_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()


def process_single_image(image_path, output_dir, show_visualization=False):
    """处理单张图像并返回Dice系数"""
    try:
        # 步骤1：加载图像
        original, gray = load_image(image_path)

        # 获取图像文件名（不含路径）
        image_filename = os.path.basename(image_path)
        image_name_no_ext = os.path.splitext(image_filename)[0]

        # 尝试加载金标准掩膜
        ground_truth = None
        try:
            # 尝试从左右肺金标准合并
            left_gt_path = os.path.join("MontgomerySet/ManualMask/leftMask", image_filename)
            right_gt_path = os.path.join("MontgomerySet/ManualMask/rightMask", image_filename)

            left_gt = cv2.imread(left_gt_path, cv2.IMREAD_GRAYSCALE)
            right_gt = cv2.imread(right_gt_path, cv2.IMREAD_GRAYSCALE)

            if left_gt is not None and right_gt is not None:
                # 合并左右肺金标准
                _, left_gt_binary = cv2.threshold(left_gt, 127, 255, cv2.THRESH_BINARY)
                _, right_gt_binary = cv2.threshold(right_gt, 127, 255, cv2.THRESH_BINARY)
                ground_truth = cv2.bitwise_or(left_gt_binary, right_gt_binary)
            else:
                # 尝试从GroundTruth_Masks目录加载
                ground_truth = load_ground_truth(image_path)
        except Exception as e:
            print(f"警告：无法加载金标准掩膜: {str(e)}")

        # 步骤2：图像预处理
        # 先进行全局对比度调整
        alpha = 1.2  # 对比度增益
        beta = -20  # 亮度偏移
        adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        # 只使用CLAHE增强
        clahe = clahe_enhancement(adjusted)

        # 步骤3：核心分割算法
        # 粗分割
        binary, chest_mask = coarse_segmentation(clahe)

        # 使用形态学操作和连通性分析优化分割结果
        optimized_mask = optimize_with_morphology_and_connectivity(binary, chest_mask)

        # 区域定位
        left_lung_mask, right_lung_mask, left_bbox, right_bbox = locate_lung_regions(optimized_mask)

        # 精细分割
        left_lung_refined = refine_segmentation(clahe, left_lung_mask, left_bbox)
        right_lung_refined = refine_segmentation(clahe, right_lung_mask, right_bbox)

        # 步骤4：形态学优化
        left_lung_optimized = morphological_optimization(left_lung_refined)
        right_lung_optimized = morphological_optimization(right_lung_refined)

        # 步骤5：后处理与边界修复
        left_lung_repaired = repair_heart_boundary(left_lung_optimized)
        right_lung_repaired = repair_heart_boundary(right_lung_optimized)

        # 结果整合
        final_segmentation = cv2.bitwise_or(left_lung_repaired, right_lung_repaired)

        # 额外的后处理：填充可能的大空洞
        final_segmentation = binary_fill_holes(final_segmentation // 255).astype(np.uint8) * 255

        # 再次应用胸腔掩码，确保没有身体外轮廓残留
        final_segmentation = cv2.bitwise_and(final_segmentation, chest_mask)

        # 计算Dice系数
        dice_score = None
        if ground_truth is not None:
            dice_score = calculate_dice_coefficient(final_segmentation, ground_truth)
            print(f"{image_filename} - Dice系数: {dice_score:.4f}")

        # 保存最终分割结果
        output_path = os.path.join(output_dir, f"{image_name_no_ext}_segmentation.png")
        cv2.imwrite(output_path, final_segmentation)

        # 如果需要可视化
        if show_visualization:
            # 保存增强方法对比图
            compare_enhancement_methods(gray, clahe)

            # 保存分割结果对比图
            if ground_truth is not None:
                visualize_results(original, clahe, final_segmentation, ground_truth, dice_score)
                debug_visualization(original, clahe, binary, left_lung_mask, right_lung_mask, final_segmentation,
                                    ground_truth)

        return dice_score

    except Exception as e:
        print(f"处理 {image_path} 时出错: {str(e)}")
        return None


def batch_process(input_dir, output_dir, show_visualization=False):
    """批量处理目录中的所有图像"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有PNG图像文件
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]

    if not image_files:
        print(f"在目录 {input_dir} 中未找到PNG图像文件")
        return

    print(f"找到 {len(image_files)} 张图像，开始批量处理...")

    # 记录Dice系数
    dice_scores = []
    start_time = time.time()

    # 处理每张图像
    for i, image_file in enumerate(image_files):
        print(f"\n处理第 {i + 1}/{len(image_files)} 张图像: {image_file}")

        image_path = os.path.join(input_dir, image_file)
        dice_score = process_single_image(image_path, output_dir, show_visualization)

        if dice_score is not None:
            dice_scores.append(dice_score)

    # 计算统计信息
    end_time = time.time()
    processing_time = end_time - start_time

    print("\n===== 批量处理完成 =====")
    print(f"总共处理: {len(image_files)} 张图像")
    print(f"总耗时: {processing_time:.2f} 秒")
    print(f"平均处理时间: {processing_time / len(image_files):.2f} 秒/张")

    if dice_scores:
        avg_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)
        max_dice = np.max(dice_scores)
        min_dice = np.min(dice_scores)

        print("\n===== Dice系数统计 =====")
        print(f"平均Dice系数: {avg_dice:.4f} ± {std_dice:.4f}")
        print(f"最大Dice系数: {max_dice:.4f}")
        print(f"最小Dice系数: {min_dice:.4f}")

        # 保存Dice系数统计结果
        with open(os.path.join(output_dir, 'dice_statistics.txt'), 'w') as f:
            f.write(f"批量处理统计结果\n")
            f.write(f"===================\n")
            f.write(f"总共处理: {len(image_files)} 张图像\n")
            f.write(f"总耗时: {processing_time:.2f} 秒\n")
            f.write(f"平均处理时间: {processing_time / len(image_files):.2f} 秒/张\n\n")

            f.write(f"Dice系数统计\n")
            f.write(f"============\n")
            f.write(f"平均Dice系数: {avg_dice:.4f} ± {std_dice:.4f}\n")
            f.write(f"最大Dice系数: {max_dice:.4f}\n")
            f.write(f"最小Dice系数: {min_dice:.4f}\n\n")

            f.write(f"每张图像的Dice系数\n")
            f.write(f"==================\n")
            for i, (image_file, dice_score) in enumerate(zip(image_files, dice_scores)):
                f.write(f"{i + 1}. {image_file}: {dice_score:.4f}\n")

    print(f"\n所有分割结果已保存到目录: {output_dir}")


if __name__ == "__main__":
    # 设置输入和输出目录
    input_dir = "MontgomerySet/CXR_png"
    output_dir = "MontgomerySet/Segmentation_Results"

    # 批量处理所有图像
    # show_visualization=True 会为每张图像生成可视化结果，会增加处理时间
    batch_process(input_dir, output_dir, show_visualization=False)