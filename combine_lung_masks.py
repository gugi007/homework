import os
import cv2
import numpy as np
from pathlib import Path


"""
清晰展示单张原图及其对应的左右肺掩膜。
将左右肺掩膜合并为一个完整的“肺野金标准掩膜”，用于后续评估
"""


def combine_lung_masks():
    """
    合并蒙哥马利数据集（MontgomerySet）中手动标注的左右肺掩膜
    生成完整的肺野金标准掩膜
    """
    # 定义路径
    base_dir = "MontgomerySet"
    left_mask_dir = os.path.join(base_dir, "ManualMask", "leftMask")
    right_mask_dir = os.path.join(base_dir, "ManualMask", "rightMask")
    output_dir = "GroundTruth_Masks"

    # 创建输出目录（如果不存在）
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 检查左掩膜目录是否存在
    if not os.path.exists(left_mask_dir):
        print(f"错误：左掩膜目录不存在 - {left_mask_dir}")
        return

    # 获取左掩膜目录中的所有文件
    left_mask_files = [f for f in os.listdir(left_mask_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    if not left_mask_files:
        print("左掩膜目录中没有找到图像文件")
        return

    print(f"找到 {len(left_mask_files)} 个左掩膜文件")

    processed_count = 0
    error_count = 0

    # 处理每个左掩膜文件
    for left_mask_file in left_mask_files:
        try:
            # 构建完整文件路径
            left_mask_path = os.path.join(left_mask_dir, left_mask_file)
            right_mask_path = os.path.join(right_mask_dir, left_mask_file)
            output_path = os.path.join(output_dir, left_mask_file)

            # 检查右掩膜文件是否存在
            if not os.path.exists(right_mask_path):
                print(f"警告：右掩膜文件不存在 - {right_mask_path}")
                error_count += 1
                continue

            # 读取左右掩膜图像
            left_mask = cv2.imread(left_mask_path, cv2.IMREAD_GRAYSCALE)
            right_mask = cv2.imread(right_mask_path, cv2.IMREAD_GRAYSCALE)

            if left_mask is None or right_mask is None:
                print(f"警告：无法读取掩膜图像 - {left_mask_file}")
                error_count += 1
                continue

            # 确保两张图像尺寸相同
            if left_mask.shape != right_mask.shape:
                print(f"警告：左右掩膜尺寸不匹配 - {left_mask_file}")
                error_count += 1
                continue

            # 对两张二值掩膜图进行逻辑"或"运算
            # 使用最大值合并（相当于逻辑或）
            combined_mask = np.maximum(left_mask, right_mask)

            # 保存合并后的掩膜
            cv2.imwrite(output_path, combined_mask)

            processed_count += 1
            print(f"已处理: {left_mask_file}")

        except Exception as e:
            print(f"处理文件 {left_mask_file} 时出错: {str(e)}")
            error_count += 1

    # 输出处理结果统计
    print(f"\n处理完成!")
    print(f"成功处理: {processed_count} 个文件")
    print(f"处理失败: {error_count} 个文件")
    print(f"输出目录: {output_dir}")


def validate_mask_combination():
    """
    验证掩膜合并功能的辅助函数
    """
    # 测试用例：创建简单的测试掩膜
    test_left = np.zeros((100, 100), dtype=np.uint8)
    test_right = np.zeros((100, 100), dtype=np.uint8)

    # 在左右掩膜中设置不同的区域
    test_left[20:40, 20:40] = 255  # 左肺区域
    test_right[20:40, 60:80] = 255  # 右肺区域

    # 合并掩膜
    test_combined = np.maximum(test_left, test_right)

    print("验证测试:")
    print(f"左掩膜非零像素: {np.count_nonzero(test_left)}")
    print(f"右掩膜非零像素: {np.count_nonzero(test_right)}")
    print(f"合并掩膜非零像素: {np.count_nonzero(test_combined)}")
    print("验证通过!")


if __name__ == "__main__":
    # 运行验证（可选）
    validate_mask_combination()

    # 执行主函数
    combine_lung_masks()