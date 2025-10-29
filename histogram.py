import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# 设置字体家族和字体大小
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先使用SimHei，备用英文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
plt.rcParams['font.size'] = 10  # 可选：统一设置字体大小

def apply_histogram_equalization(image):
    """应用普通直方图均衡化"""
    # 如果是彩色图像，转换为灰度图
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用直方图均衡化
    equalized = cv2.equalizeHist(image)
    return equalized


def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    """应用对比度受限的自适应直方图均衡化(CLAHE)"""
    # 如果是彩色图像，转换为灰度图
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    # 应用CLAHE
    clahe_equalized = clahe.apply(image)
    return clahe_equalized


def compare_histogram_equalizations(image_path):
    """对比普通直方图均衡化和CLAHE的效果"""

    # 读取图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"无法读取图像: {image_path}")
        return

    # 转换为灰度图（如果原本不是）
    if len(original_image.shape) == 3:
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = original_image.copy()

    print(f"图像尺寸: {gray_image.shape}")
    print(f"图像数据类型: {gray_image.dtype}")
    print(f"像素值范围: [{gray_image.min()}, {gray_image.max()}]")

    # 应用普通直方图均衡化
    he_image = apply_histogram_equalization(gray_image)

    # 应用CLAHE（尝试不同的参数）
    # CLAHE 标准
    # clipLimit（对比度限制阈值） 防止噪声过大
    # grid_size(网格大小) 较小的网格（如 (4,4)）能增强细微纹理但可能产生块效应；较大的网格（如 (16,16)）处理更平滑但局部细节保留较少
    clahe_image_standard = apply_clahe(gray_image, clip_limit=2.0, grid_size=(8, 8))
    # CLAHE 激进
    clahe_image_aggressive = apply_clahe(gray_image, clip_limit=4.0, grid_size=(4, 4))
    # CLAHE 保守
    clahe_image_conservative = apply_clahe(gray_image, clip_limit=1.0, grid_size=(16, 16))

    # 计算直方图
    hist_original = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    hist_he = cv2.calcHist([he_image], [0], None, [256], [0, 256])
    hist_clahe = cv2.calcHist([clahe_image_standard], [0], None, [256], [0, 256])

    # 创建对比可视化
    # 创建一个有着三行四列子图的大图
    # 给大图一个title 直方图均衡化方法对比 - 胸部X光片分析
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('直方图均衡化方法对比 - 胸部X光片分析', fontsize=16, fontweight='bold')

    # 原图
    axes[0, 0].imshow(gray_image, cmap='gray')
    axes[0, 0].set_title('原始图像')
    #axex[0,0].axis('off')，关闭坐标轴
    axes[0, 0].axis('off')

    #原始图像直方图
    axes[1, 0].plot(hist_original, color='black')
    axes[1, 0].set_title('原始图像直方图')
    axes[1, 0].set_xlabel('像素值')
    axes[1, 0].set_ylabel('频数')
    axes[1, 0].grid(True, alpha=0.3)

    # 普通直方图均衡化
    axes[0, 1].imshow(he_image, cmap='gray')
    axes[0, 1].set_title('普通直方图均衡化')
    axes[0, 1].axis('off')

    axes[1, 1].plot(hist_he, color='blue')
    axes[1, 1].set_title('均衡化后直方图')
    axes[1, 1].set_xlabel('像素值')
    axes[1, 1].set_ylabel('频数')
    axes[1, 1].grid(True, alpha=0.3)

    # CLAHE标准参数
    axes[0, 2].imshow(clahe_image_standard, cmap='gray')
    axes[0, 2].set_title('CLAHE (标准参数)')
    axes[0, 2].axis('off')

    axes[1, 2].plot(hist_clahe, color='red')
    axes[1, 2].set_title('CLAHE后直方图')
    axes[1, 2].set_xlabel('像素值')
    axes[1, 2].set_ylabel('频数')
    axes[1, 2].grid(True, alpha=0.3)

    # CLAHE不同参数对比
    axes[0, 3].imshow(clahe_image_aggressive, cmap='gray')
    axes[0, 3].set_title('CLAHE (激进参数)')
    axes[0, 3].axis('off')

    axes[1, 3].imshow(clahe_image_conservative, cmap='gray')
    axes[1, 3].set_title('CLAHE (保守参数)')
    axes[1, 3].axis('off')

    # 放大显示关键区域（肺部区域）
    height, width = gray_image.shape
    # 假设肺部在图像中心区域
    lung_region = gray_image[height // 4:3 * height // 4, width // 4:3 * width // 4]
    lung_he = he_image[height // 4:3 * height // 4, width // 4:3 * width // 4]
    lung_clahe = clahe_image_standard[height // 4:3 * height // 4, width // 4:3 * width // 4]

    # 显示放大区域
    axes[2, 0].imshow(lung_region, cmap='gray')
    axes[2, 0].set_title('原图肺部区域(放大)')
    axes[2, 0].axis('off')

    axes[2, 1].imshow(lung_he, cmap='gray')
    axes[2, 1].set_title('均衡化肺部区域(放大)')
    axes[2, 1].axis('off')

    axes[2, 2].imshow(lung_clahe, cmap='gray')
    axes[2, 2].set_title('CLAHE肺部区域(放大)')
    axes[2, 2].axis('off')

    # 添加技术指标对比
    axes[2, 3].axis('off')
    text_content = f"""
技术指标对比:

原始图像:
- 对比度: {gray_image.std():.1f}
- 亮度: {gray_image.mean():.1f}

普通直方图均衡化:
- 对比度: {he_image.std():.1f}
- 亮度: {he_image.mean():.1f}
- 对比度提升: {he_image.std() / gray_image.std():.1f}x

CLAHE (标准参数):
- 对比度: {clahe_image_standard.std():.1f}
- 亮度: {clahe_image_standard.mean():.1f}
- 对比度提升: {clahe_image_standard.std() / gray_image.std():.1f}x

CLAHE优势:
✓ 自适应局部处理
✓ 抑制噪声放大
✓ 保持组织细节
"""

    axes[2, 3].text(0.1, 0.9, text_content, transform=axes[2, 3].transAxes,
                    fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.savefig('histogram_comparison.png', dpi=300, bbox_inches='tight')  # 保存为高分辨率PNG，并裁剪空白
    plt.show()

    # 返回处理结果供进一步分析
    return {
        'original': gray_image,
        'he': he_image,
        'clahe_standard': clahe_image_standard,
        'clahe_aggressive': clahe_image_aggressive,
        'clahe_conservative': clahe_image_conservative
    }


def analyze_noise_suppression(original, he, clahe):
    """分析噪声抑制效果"""

    # 计算局部方差图来评估噪声
    def calculate_local_variance(image, kernel_size=5):
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
        mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
        mean_sq = cv2.filter2D((image.astype(np.float32) ** 2), -1, kernel)
        variance = mean_sq - mean ** 2
        return variance

    # 计算各图像的局部方差
    var_original = calculate_local_variance(original)
    var_he = calculate_local_variance(he)
    var_clahe = calculate_local_variance(clahe)

    # 创建噪声分析图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 显示原图和各处理结果
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(he, cmap='gray')
    axes[0, 1].set_title('普通直方图均衡化')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(clahe, cmap='gray')
    axes[0, 2].set_title('CLAHE')
    axes[0, 2].axis('off')

    # 显示局部方差图（噪声分布）
    vmin, vmax = 0, np.percentile(var_original, 95)  # 使用95分位数作为上限

    axes[1, 0].imshow(var_original, cmap='hot', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(f'原图局部方差\n均值: {var_original.mean():.1f}')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(var_he, cmap='hot', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title(f'均衡化局部方差\n均值: {var_he.mean():.1f}')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(var_clahe, cmap='hot', vmin=vmin, vmax=vmax)
    axes[1, 2].set_title(f'CLAHE局部方差\n均值: {var_clahe.mean():.1f}')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"噪声水平分析:")
    print(f"原始图像平均局部方差: {var_original.mean():.2f}")
    print(f"普通直方图均衡化平均局部方差: {var_he.mean():.2f} (增加 {var_he.mean() / var_original.mean():.1f}x)")
    print(f"CLAHE平均局部方差: {var_clahe.mean():.2f} (增加 {var_clahe.mean() / var_original.mean():.1f}x)")


# 使用示例
if __name__ == "__main__":
    # 请将下面的路径替换为您的胸部X光片路径
    image_path = "MontgomerySet/CXR_png/MCUCXR_0001_0.png"  # 替换为您的图像路径

    # 运行对比实验
    results = compare_histogram_equalizations(image_path)

    if results is not None:
        # 分析噪声抑制效果
        analyze_noise_suppression(results['original'], results['he'], results['clahe_standard'])

        # 保存处理结果
        cv2.imwrite("original.jpg", results['original'])
        cv2.imwrite("histogram_equalized.jpg", results['he'])
        cv2.imwrite("clahe_standard.jpg", results['clahe_standard'])
        cv2.imwrite("clahe_aggressive.jpg", results['clahe_aggressive'])
        cv2.imwrite("clahe_conservative.jpg", results['clahe_conservative'])

        print("所有处理结果已保存为图像文件")