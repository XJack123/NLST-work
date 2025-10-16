#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NLST肺结节检测独立运行脚本
只需运行这个文件即可完成检测
"""

import json
import logging
import sys
import time
import os
import numpy as np
import torch

import monai
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.data import DataLoader, Dataset
from monai.data.utils import no_collation
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    EnsureTyped,
    ToTensord,
    Invertd,
    DeleteItemsd,
    SelectItemsd,
    Lambda,
    Lambdad,
)

print("=" * 80)
print("NLST肺结节检测系统")
print("=" * 80)

# ==================== 配置参数 ====================
CONFIG = {
    "environment": {
        "data_base_dir": "./NLSTNIfTIdata/",
        "data_list_file_path": "./nlst_detection_datalist.json",
        "model_path": "./models/DLCSD-mD.pt",
        "result_list_file_path": "./nlst_detection_results.json",
    },
    "model": {
        "gt_box_mode": "cccwhd",
        "val_patch_size": [512, 512, 208],
        "score_thresh": 0.02,
        "nms_thresh": 0.22,
        "returned_layers": [1, 2],
        "base_anchor_shapes": [[6, 8, 4], [8, 6, 5], [10, 10, 6]],
    }
}

print("\n📋 配置信息:")
print(f"  数据目录: {CONFIG['environment']['data_base_dir']}")
print(f"  模型文件: {CONFIG['environment']['model_path']}")
print(f"  结果输出: {CONFIG['environment']['result_list_file_path']}")
print(f"  Patch大小: {CONFIG['model']['val_patch_size']}")

# ==================== 检查文件 ====================
def check_files():
    """检查必需文件是否存在"""
    print("\n🔍 检查必需文件...")
    
    required_files = [
        CONFIG['environment']['data_list_file_path'],
        CONFIG['environment']['model_path'],
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (缺失)")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ 错误: 以下文件缺失:")
        for f in missing_files:
            print(f"  - {f}")
        sys.exit(1)
    
    # 检查数据目录
    if os.path.exists(CONFIG['environment']['data_base_dir']):
        print(f"  ✅ {CONFIG['environment']['data_base_dir']}")
    else:
        print(f"  ❌ {CONFIG['environment']['data_base_dir']} (缺失)")
        sys.exit(1)
    
    print("✅ 所有必需文件检查通过!")

# ==================== 定义Transforms ====================
def generate_detection_inference_transform(
    image_key,
    pred_box_key,
    pred_label_key,
    pred_score_key,
    gt_box_mode,
    intensity_transform,
    affine_lps_to_ras=False,
    amp=True,
):
    """生成检测推理的transform"""
    
    # 推理transforms
    inference_transforms = Compose([
        LoadImaged(keys=[image_key], image_only=False),  # 保留元数据
        EnsureChannelFirstd(keys=[image_key]),
        # 强制选择第一个通道（处理多通道图像）
        Lambdad(keys=[image_key], func=lambda x: x[:1] if x.shape[0] > 1 else x),
        # Orientationd 在某些损坏的NIfTI文件上会失败，已移除
        # Orientationd(keys=[image_key], axcodes="RAS" if affine_lps_to_ras else "LPS"),
        intensity_transform,
        EnsureTyped(keys=[image_key], dtype=torch.float16 if amp else torch.float32),
    ])
    
    # 后处理transforms
    post_transforms = Compose([
        Invertd(
            keys=[pred_box_key],
            transform=inference_transforms,
            orig_keys=[image_key],
            nearest_interp=False,
            to_tensor=True,
        ),
        DeleteItemsd(keys=["image", "image_transforms"]),
    ])
    
    return inference_transforms, post_transforms

# ==================== 主程序 ====================
def main():
    """主程序"""
    
    # 检查文件
    check_files()
    
    # 设置
    amp = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n💻 使用设备: {device}")
    
    if device.type == "cpu":
        print("⚠️  警告: 未检测到GPU，将使用CPU运行（速度会很慢）")
    
    monai.config.print_config()
    
    # 加载配置
    env_dict = CONFIG['environment']
    config_dict = CONFIG['model']
    
    patch_size = config_dict['val_patch_size']
    
    # 1. 定义transform
    print("\n🔧 定义数据转换...")
    intensity_transform = ScaleIntensityRanged(
        keys=["image"],
        a_min=-1024,
        a_max=300.0,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    )
    
    inference_transforms, post_transforms = generate_detection_inference_transform(
        "image",
        "pred_box",
        "pred_label",
        "pred_score",
        config_dict['gt_box_mode'],
        intensity_transform,
        affine_lps_to_ras=False,
        amp=amp,
    )
    
    # 2. 创建数据加载器
    print("📂 加载数据列表...")
    
    # 手动加载JSON文件以正确处理UTF-8编码（包括BOM）
    with open(env_dict['data_list_file_path'], 'r', encoding='utf-8-sig') as f:
        datalist_json = json.load(f)
    
    # 处理数据列表
    base_dir = env_dict['data_base_dir']
    inference_data = []
    for item in datalist_json.get('validation', []):
        if isinstance(item, dict):
            # 添加完整路径，并清理可能的BOM字符
            item_copy = item.copy()
            if 'image' in item_copy:
                # 清理BOM和其他不可见字符
                image_path = item_copy['image'].lstrip('\ufeff').strip()
                item_copy['image'] = os.path.join(base_dir, image_path)
            inference_data.append(item_copy)
    
    print(f"  找到 {len(inference_data)} 个CT扫描文件")
    
    inference_ds = Dataset(
        data=inference_data,
        transform=inference_transforms,
    )
    
    inference_loader = DataLoader(
        inference_ds,
        batch_size=1,
        num_workers=0,  # Windows上避免pickle问题
        pin_memory=torch.cuda.is_available(),
        collate_fn=no_collation,
    )
    
    # 3. 构建模型
    print("\n🏗️  构建检测模型...")
    
    # 构建anchor生成器
    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2**l for l in range(len(config_dict['returned_layers']) + 1)],
        base_anchor_shapes=config_dict['base_anchor_shapes'],
    )
    
    # 加载网络
    print(f"📥 加载模型: {env_dict['model_path']}")
    try:
        net = torch.jit.load(env_dict['model_path']).to(device)
        print("  ✅ 模型加载成功!")
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        sys.exit(1)
    
    # 构建检测器
    detector = RetinaNetDetector(
        network=net, 
        anchor_generator=anchor_generator, 
        debug=False
    )
    
    # 设置推理参数
    detector.set_box_selector_parameters(
        score_thresh=config_dict['score_thresh'],
        topk_candidates_per_level=1000,
        nms_thresh=config_dict['nms_thresh'],
        detections_per_img=100,
    )
    
    detector.set_sliding_window_inferer(
        roi_size=patch_size,
        overlap=0.25,
        sw_batch_size=4,  # 增加batch size以充分利用GPU
        mode="constant",
        device="cuda" if torch.cuda.is_available() else "cpu",  # 使用GPU！
    )
    
    # 4. 运行推理
    print("\n" + "=" * 80)
    print("🚀 开始检测...")
    print("=" * 80)
    
    results_dict = {"validation": []}
    detector.eval()
    
    with torch.no_grad():
        start_time = time.time()
        
        for idx, inference_data in enumerate(inference_loader, 1):
            inference_img_filenames = [
                inference_data_i["image_meta_dict"]["filename_or_obj"]
                for inference_data_i in inference_data
            ]
            
            print(f"\n[{idx}/{len(inference_loader)}] 处理: {os.path.basename(inference_img_filenames[0])}")
            
            # 检查是否需要使用inferer
            use_inferer = not all([
                inference_data_i["image"][0, ...].numel() < np.prod(patch_size)
                for inference_data_i in inference_data
            ])
            
            if use_inferer:
                print("  使用sliding window推理（图像较大）")
            
            inference_inputs = [
                inference_data_i["image"].to(device)
                for inference_data_i in inference_data
            ]
            
            # 运行检测
            inference_start = time.time()
            if amp:
                with torch.cuda.amp.autocast():
                    inference_outputs = detector(
                        inference_inputs, use_inferer=use_inferer
                    )
            else:
                inference_outputs = detector(inference_inputs, use_inferer=use_inferer)
            
            inference_time = time.time() - inference_start
            print(f"  推理耗时: {inference_time:.2f}秒")
            
            del inference_inputs
            
            # 更新推理数据进行后处理
            for i in range(len(inference_outputs)):
                inference_data_i, inference_pred_i = (
                    inference_data[i],
                    inference_outputs[i],
                )
                inference_data_i["pred_box"] = inference_pred_i[
                    detector.target_box_key
                ].to(torch.float32)
                inference_data_i["pred_label"] = inference_pred_i[
                    detector.target_label_key
                ]
                inference_data_i["pred_score"] = inference_pred_i[
                    detector.pred_score_key
                ].to(torch.float32)
                inference_data[i] = post_transforms(inference_data_i)
            
            # 保存结果
            for inference_img_filename, inference_pred_i in zip(
                inference_img_filenames, inference_data
            ):
                num_detections = len(inference_pred_i["pred_label"])
                print(f"  检测到 {num_detections} 个候选结节")
                
                if num_detections > 0:
                    scores = inference_pred_i["pred_score"].cpu().detach().numpy()
                    high_conf = sum(scores > 0.5)
                    med_conf = sum((scores > 0.2) & (scores <= 0.5))
                    print(f"    - 高置信度 (>0.5): {high_conf} 个")
                    print(f"    - 中等置信度 (0.2-0.5): {med_conf} 个")
                
                result = {
                    "label": inference_pred_i["pred_label"]
                    .cpu()
                    .detach()
                    .numpy()
                    .tolist(),
                    "box": inference_pred_i["pred_box"].cpu().detach().numpy().tolist(),
                    "score": inference_pred_i["pred_score"]
                    .cpu()
                    .detach()
                    .numpy()
                    .tolist(),
                }
                result.update({"image": inference_img_filename})
                results_dict["validation"].append(result)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 80)
    print("✅ 检测完成!")
    print("=" * 80)
    print(f"总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"平均每个扫描: {total_time/len(inference_loader):.2f}秒")
    
    # 5. 保存结果
    print(f"\n💾 保存结果到: {env_dict['result_list_file_path']}")
    with open(env_dict['result_list_file_path'], "w", encoding='utf-8') as outfile:
        json.dump(results_dict, outfile, indent=2, ensure_ascii=False)
    
    print("✅ 结果已保存!")
    
    # 统计信息
    print("\n📊 检测统计:")
    total_detections = sum([len(r['label']) for r in results_dict['validation']])
    files_with_detections = sum([1 for r in results_dict['validation'] if len(r['label']) > 0])
    print(f"  总检测数: {total_detections}")
    print(f"  有检测结果的文件: {files_with_detections}/{len(results_dict['validation'])}")
    
    if total_detections > 0:
        all_scores = []
        for r in results_dict['validation']:
            all_scores.extend(r['score'])
        all_scores = np.array(all_scores)
        print(f"  平均置信度: {np.mean(all_scores):.3f}")
        print(f"  最高置信度: {np.max(all_scores):.3f}")
        print(f"  最低置信度: {np.min(all_scores):.3f}")
    
    print("\n🎉 完成! 请查看结果文件获取详细检测信息。")
    print(f"   结果文件: {env_dict['result_list_file_path']}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

