#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
肺结节特征提取与可解释性分析系统
从detection结果中提取结节，使用classification模型分析特征
"""

import json
import os
import numpy as np
import torch
import nibabel as nib
from scipy import ndimage
import pandas as pd

import monai
from monai.networks.nets import resnet50

print("=" * 80)
print("肺结节特征提取与可解释性分析系统")
print("=" * 80)

# ==================== 配置 ====================
CONFIG = {
    "detection_results": "./nlst_detection_results.json",
    "classification_model": "./models/DukeLungRADS_CancerCL_Fold1_resnet50WSPP.pt",
    "output_features": "./nodule_features.csv",
    "output_predictions": "./nodule_predictions.csv",
    "output_explanations": "./nodule_explanations/",
    
    # ROI提取参数
    "roi_size": [64, 64, 64],  # 结节ROI大小
    "margin_factor": 2.0,  # 边界框扩展倍数
    "min_confidence": 0.2,  # 最低置信度阈值（只分析置信度>0.2的结节）
    
    # 模型参数
    "num_classes": 2,  # 良性(0) vs 恶性(1)
    "spatial_dims": 3,
    "n_input_channels": 1,
}

print("\n📋 配置信息:")
print(f"  Detection结果: {CONFIG['detection_results']}")
print(f"  Classification模型: {CONFIG['classification_model']}")
print(f"  ROI大小: {CONFIG['roi_size']}")
print(f"  最低置信度阈值: {CONFIG['min_confidence']}")

# ==================== GradCAM实现（可解释性） ====================
class GradCAM:
    """梯度加权类激活映射 - 用于可视化模型关注的区域"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        """生成GradCAM热图"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # 计算权重
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        
        # 加权求和
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # 归一化到0-1
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.squeeze().cpu().numpy(), target_class

# ==================== 特征提取器 ====================
class NoduleFeatureExtractor:
    """肺结节特征提取器"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"\n💻 使用设备: {self.device}")
        
        # 加载模型
        print(f"📥 加载classification模型...")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 创建特征提取hook
        self.features = {}
        self._register_hooks()
        
        print("✅ 模型加载成功!")
    
    def _load_model(self, model_path):
        """加载模型"""
        # 创建ResNet50模型
        model = resnet50(
            pretrained=False,
            spatial_dims=CONFIG['spatial_dims'],
            n_input_channels=CONFIG['n_input_channels'],
            num_classes=CONFIG['num_classes']
        )
        
        # 加载权重
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        
        return model
    
    def _register_hooks(self):
        """注册特征提取hooks - 提取多层特征（浅层到深层）"""
        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output.detach()
            return hook
        
        # ResNet50的多层特征提取
        # layer1: 浅层特征 (256维特征图，分辨率较高，捕获边缘、纹理等低级特征)
        if hasattr(self.model, 'layer1'):
            self.model.layer1.register_forward_hook(get_features('layer1'))
        
        # layer2: 中浅层特征 (512维特征图，捕获简单形状和模式)
        if hasattr(self.model, 'layer2'):
            self.model.layer2.register_forward_hook(get_features('layer2'))
        
        # layer3: 中深层特征 (1024维特征图，捕获复杂形状和部件)
        if hasattr(self.model, 'layer3'):
            self.model.layer3.register_forward_hook(get_features('layer3'))
        
        # layer4: 深层特征 (2048维特征图，包含高级语义特征)
        if hasattr(self.model, 'layer4'):
            self.model.layer4.register_forward_hook(get_features('layer4'))
        
        # avgpool: 全局平均池化后的最终特征向量 (2048维)
        if hasattr(self.model, 'avgpool'):
            self.model.avgpool.register_forward_hook(get_features('avgpool'))
    
    def extract_roi(self, ct_image_path, bbox, margin_factor=2.0):
        """
        从CT图像中提取结节ROI
        
        参数:
            ct_image_path: CT图像路径
            bbox: 边界框 [x_min, y_min, z_min, x_max, y_max, z_max]
            margin_factor: 边界扩展倍数
        
        返回:
            roi: 提取的ROI数组
        """
        # 加载CT图像
        nii_img = nib.load(ct_image_path)
        ct_data = nii_img.get_fdata()
        
        # 计算边界框中心和大小
        x_min, y_min, z_min, x_max, y_max, z_max = bbox
        
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        cz = (z_min + z_max) / 2
        
        w = (x_max - x_min) * margin_factor
        h = (y_max - y_min) * margin_factor
        d = (z_max - z_min) * margin_factor
        
        # 确保最小ROI大小
        w = max(w, 20)
        h = max(h, 20)
        d = max(d, 10)
        
        # 计算新的边界
        x_start = int(max(0, cx - w/2))
        x_end = int(min(ct_data.shape[0], cx + w/2))
        y_start = int(max(0, cy - h/2))
        y_end = int(min(ct_data.shape[1], cy + h/2))
        z_start = int(max(0, cz - d/2))
        z_end = int(min(ct_data.shape[2], cz + d/2))
        
        # 提取ROI
        roi = ct_data[x_start:x_end, y_start:y_end, z_start:z_end]
        
        return roi, (x_start, x_end, y_start, y_end, z_start, z_end)
    
    def extract_features(self, roi_data):
        """
        提取多层深度特征（从浅层到深层）
        
        返回:
            multi_layer_features: 包含各层特征的字典
            logits: 分类logits
            prob: 分类概率
        """
        # 手动预处理
        # 1. HU值归一化
        roi_normalized = np.clip(roi_data, -1024, 300)
        roi_normalized = (roi_normalized + 1024) / 1324.0  # 归一化到[0, 1]
        
        # 2. 转为torch tensor并添加batch和channel维度
        roi_tensor = torch.from_numpy(roi_normalized).float()
        roi_tensor = roi_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        
        # 3. Resize到目标大小
        roi_tensor = torch.nn.functional.interpolate(
            roi_tensor,
            size=CONFIG['roi_size'],
            mode='trilinear',
            align_corners=False
        )
        
        # 移到设备
        roi_tensor = roi_tensor.to(self.device)
        
        # 前向传播
        with torch.no_grad():
            logits = self.model(roi_tensor)
            prob = torch.softmax(logits, dim=1)
        
        # 提取多层特征
        multi_layer_features = {}
        
        # avgpool特征：最终特征向量（2048维）
        if 'avgpool' in self.features:
            multi_layer_features['avgpool'] = self.features['avgpool'].squeeze().cpu().numpy()
        
        # 对各层特征图进行全局平均池化，得到特征向量
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if layer_name in self.features:
                # 特征图形状: [1, C, D, H, W]
                feature_map = self.features[layer_name]
                # 全局平均池化: [1, C, D, H, W] -> [1, C]
                pooled_features = torch.mean(feature_map, dim=[2, 3, 4])
                multi_layer_features[layer_name] = pooled_features.squeeze().cpu().numpy()
        
        return multi_layer_features, logits.cpu().numpy(), prob.cpu().numpy()
    
    def get_interpretable_features(self, roi_data, bbox):
        """
        提取可解释的特征
        
        返回包含以下内容的字典:
        - 形状特征：体积、球形度、紧凑度
        - 强度特征：平均HU、标准差、峰度、偏度
        - 纹理特征：熵、对比度
        - 位置特征：x,y,z坐标
        """
        features = {}
        
        # 1. 形状特征
        volume = np.prod(roi_data.shape)
        features['volume_voxels'] = volume
        
        # 计算球形度（简化版）
        bbox_volume = (bbox[3]-bbox[0]) * (bbox[4]-bbox[1]) * (bbox[5]-bbox[2])
        features['sphericity'] = volume / (bbox_volume + 1e-6)
        
        # 2. 强度特征（HU值）
        features['mean_hu'] = float(np.mean(roi_data))
        features['std_hu'] = float(np.std(roi_data))
        features['min_hu'] = float(np.min(roi_data))
        features['max_hu'] = float(np.max(roi_data))
        features['median_hu'] = float(np.median(roi_data))
        
        # 偏度和峰度
        from scipy import stats
        features['skewness'] = float(stats.skew(roi_data.flatten()))
        features['kurtosis'] = float(stats.kurtosis(roi_data.flatten()))
        
        # 3. 纹理特征
        # 计算熵
        hist, _ = np.histogram(roi_data, bins=50)
        hist = hist / (hist.sum() + 1e-6)
        entropy = -np.sum(hist * np.log2(hist + 1e-6))
        features['entropy'] = float(entropy)
        
        # 计算对比度（标准差/均值）
        features['contrast'] = float(features['std_hu'] / (abs(features['mean_hu']) + 1e-6))
        
        # 4. 位置特征
        features['center_x'] = float((bbox[0] + bbox[3]) / 2)
        features['center_y'] = float((bbox[1] + bbox[4]) / 2)
        features['center_z'] = float((bbox[2] + bbox[5]) / 2)
        
        features['width'] = float(bbox[3] - bbox[0])
        features['height'] = float(bbox[4] - bbox[1])
        features['depth'] = float(bbox[5] - bbox[2])
        
        return features

# ==================== 主程序 ====================
def main():
    """主程序"""
    
    # 检查文件
    if not os.path.exists(CONFIG['detection_results']):
        print(f"❌ 找不到detection结果文件: {CONFIG['detection_results']}")
        return
    
    if not os.path.exists(CONFIG['classification_model']):
        print(f"❌ 找不到classification模型: {CONFIG['classification_model']}")
        return
    
    # 创建输出目录
    os.makedirs(CONFIG['output_explanations'], exist_ok=True)
    
    # 加载detection结果
    print("\n📂 加载detection结果...")
    with open(CONFIG['detection_results'], 'r', encoding='utf-8') as f:
        detection_data = json.load(f)
    
    # 初始化特征提取器
    extractor = NoduleFeatureExtractor(CONFIG['classification_model'])
    
    # 存储所有结果
    all_features = []
    all_predictions = []
    
    print("\n" + "=" * 80)
    print("🔬 开始特征提取...")
    print("=" * 80)
    
    nodule_count = 0
    
    # 遍历所有检测结果
    for scan_idx, scan_result in enumerate(detection_data['validation'], 1):
        image_path = scan_result['image']
        boxes = scan_result['box']
        scores = scan_result['score']
        labels = scan_result['label']
        
        # 过滤低置信度结节
        high_conf_indices = [i for i, s in enumerate(scores) if s >= CONFIG['min_confidence']]
        
        if len(high_conf_indices) == 0:
            continue
        
        print(f"\n[{scan_idx}/59] 处理: {os.path.basename(image_path)}")
        print(f"  检测到 {len(boxes)} 个结节，其中 {len(high_conf_indices)} 个置信度 >= {CONFIG['min_confidence']}")
        
        # 处理每个高置信度结节
        for idx in high_conf_indices:
            nodule_count += 1
            bbox = boxes[idx]
            score = scores[idx]
            
            try:
                # 1. 提取ROI
                roi_data, roi_bounds = extractor.extract_roi(
                    image_path, 
                    bbox, 
                    margin_factor=CONFIG['margin_factor']
                )
                
                # 2. 提取多层深度学习特征
                multi_layer_features, logits, probs = extractor.extract_features(roi_data)
                
                # 3. 提取可解释的传统特征
                interpretable_features = extractor.get_interpretable_features(roi_data, bbox)
                
                # 4. 分类预测
                predicted_class = int(np.argmax(probs))
                malignancy_prob = float(probs[0, 1])  # 恶性概率
                benign_prob = float(probs[0, 0])      # 良性概率
                
                # 5. 整合所有信息
                result = {
                    'nodule_id': nodule_count,
                    'scan_file': os.path.basename(image_path),
                    'detection_score': float(score),
                    
                    # 分类结果
                    'predicted_class': predicted_class,
                    'class_name': '恶性' if predicted_class == 1 else '良性',
                    'malignancy_probability': malignancy_prob,
                    'benign_probability': benign_prob,
                    
                    # 位置信息
                    'bbox_x_min': float(bbox[0]),
                    'bbox_y_min': float(bbox[1]),
                    'bbox_z_min': float(bbox[2]),
                    'bbox_x_max': float(bbox[3]),
                    'bbox_y_max': float(bbox[4]),
                    'bbox_z_max': float(bbox[5]),
                }
                
                # 添加可解释特征
                result.update(interpretable_features)
                
                all_predictions.append(result)
                
                # 打印结果
                print(f"    结节#{nodule_count}: {result['class_name']} "
                      f"(恶性概率: {malignancy_prob:.2%}, 检测置信度: {score:.2%})")
                print(f"      - 位置: ({interpretable_features['center_x']:.1f}, "
                      f"{interpretable_features['center_y']:.1f}, "
                      f"{interpretable_features['center_z']:.1f})")
                print(f"      - 大小: {interpretable_features['width']:.1f} × "
                      f"{interpretable_features['height']:.1f} × "
                      f"{interpretable_features['depth']:.1f} 像素")
                print(f"      - 平均HU值: {interpretable_features['mean_hu']:.1f}")
                
                # 保存多层深度特征
                if multi_layer_features:
                    feature_dict = {
                        'nodule_id': nodule_count,
                        'scan_file': os.path.basename(image_path),
                    }
                    
                    # 保存各层特征的维度信息和实际特征
                    for layer_name, features in multi_layer_features.items():
                        feature_dict[f'{layer_name}_dim'] = len(features)
                        feature_dict[f'{layer_name}_features'] = features.tolist()
                    
                    all_features.append(feature_dict)
                
            except Exception as e:
                print(f"    ⚠️  结节处理失败: {e}")
                continue
    
    # 保存结果
    print("\n" + "=" * 80)
    print("💾 保存结果...")
    print("=" * 80)
    
    # 保存预测和可解释特征
    if all_predictions:
        df_predictions = pd.DataFrame(all_predictions)
        df_predictions.to_csv(CONFIG['output_predictions'], index=False, encoding='utf-8-sig')
        print(f"✅ 预测结果已保存: {CONFIG['output_predictions']}")
        print(f"   共 {len(all_predictions)} 个结节")
        
        # 统计
        print("\n📊 统计信息:")
        print(f"  总结节数: {len(all_predictions)}")
        malignant_count = sum([1 for p in all_predictions if p['predicted_class'] == 1])
        benign_count = sum([1 for p in all_predictions if p['predicted_class'] == 0])
        print(f"  预测为恶性: {malignant_count} 个")
        print(f"  预测为良性: {benign_count} 个")
        
        # 高危结节（恶性概率>0.5）
        high_risk = [p for p in all_predictions if p['malignancy_probability'] > 0.5]
        print(f"  高危结节(恶性概率>50%): {len(high_risk)} 个")
        
        if high_risk:
            print("\n⚠️  高危结节列表:")
            for h in sorted(high_risk, key=lambda x: x['malignancy_probability'], reverse=True)[:10]:
                print(f"    结节#{h['nodule_id']}: 恶性概率={h['malignancy_probability']:.2%}, "
                      f"检测置信度={h['detection_score']:.2%}")
                print(f"      文件: {h['scan_file']}")
    
    # 保存深度特征
    if all_features:
        with open(CONFIG['output_features'].replace('.csv', '.json'), 'w') as f:
            json.dump(all_features, f, indent=2)
        print(f"\n✅ 深度特征已保存: {CONFIG['output_features'].replace('.csv', '.json')}")
    
    print("\n🎉 特征提取完成!")
    print(f"\n📄 输出文件:")
    print(f"  1. {CONFIG['output_predictions']} - 可解释特征和预测结果")
    print(f"  2. {CONFIG['output_features'].replace('.csv', '.json')} - 多层深度学习特征")
    print(f"     • layer1: 256维 (浅层特征 - 边缘、纹理)")
    print(f"     • layer2: 512维 (中浅层特征 - 简单形状)")
    print(f"     • layer3: 1024维 (中深层特征 - 复杂形状)")
    print(f"     • layer4: 2048维 (深层特征 - 高级语义)")
    print(f"     • avgpool: 2048维 (最终特征向量)")

# ==================== 可解释特征说明 ====================
def print_feature_explanation():
    """打印特征说明文档"""
    explanation = """
    
📚 可解释特征说明
================================================================================

🧠 深度学习特征（多层ResNet50特征）
--------------------------------------------------------------------------------
现在提取了ResNet50的多层特征，从浅层到深层：

  • layer1 (256维): 浅层特征
    - 捕获低级视觉信息：边缘、角点、简单纹理
    - 分辨率高，空间细节丰富
    - 适合分析结节边缘的锐利度和纹理细节

  • layer2 (512维): 中浅层特征
    - 捕获简单的形状和局部模式
    - 适合分析结节的局部结构

  • layer3 (1024维): 中深层特征
    - 捕获复杂的形状和部件组合
    - 适合分析结节的整体形状特征

  • layer4 (2048维): 深层特征
    - 捕获高级语义特征和抽象表示
    - 适合区分结节类型（良性/恶性）

  • avgpool (2048维): 最终特征向量
    - layer4经过全局平均池化后的特征
    - 用于最终分类决策

💡 使用建议：
  - 浅层特征(layer1, layer2)更适合分析纹理和边缘特性
  - 深层特征(layer3, layer4, avgpool)更适合分类和语义理解
  - 可以根据需要选择不同层的特征进行分析

================================================================================

1. 形状特征 (Shape Features)
   - volume_voxels: 结节体积（体素数）
   - sphericity: 球形度（0-1，越接近1越圆）
   - width/height/depth: 结节三维尺寸（像素）

2. 强度特征 (Intensity Features) - 反映结节密度
   - mean_hu: 平均HU值（Hounsfield单位）
     * 实性结节: 通常 > -200 HU
     * 磨玻璃结节: -700 到 -200 HU
     * 囊性结节: < -700 HU
   - std_hu: HU值标准差（反映结节均匀性）
   - min_hu/max_hu: HU值范围
   - median_hu: HU值中位数

3. 分布特征 (Distribution Features)
   - skewness: 偏度（对称性）
     * > 0: 右偏（高密度像素较多）
     * < 0: 左偏（低密度像素较多）
   - kurtosis: 峰度（分布尖锐程度）
     * > 0: 尖峰分布（密度集中）
     * < 0: 平坦分布（密度分散）

4. 纹理特征 (Texture Features)
   - entropy: 熵（复杂度/不规则性）
     * 高熵: 纹理复杂、不均匀
     * 低熵: 纹理简单、均匀
   - contrast: 对比度（变化程度）

5. 位置特征 (Location Features)
   - center_x/y/z: 结节中心坐标
   - 可用于分析结节在肺部的位置分布

6. 分类预测 (Classification Prediction)
   - malignancy_probability: 恶性概率（0-1）
   - benign_probability: 良性概率（0-1）
   - predicted_class: 预测类别（0=良性，1=恶性）

📈 临床解释指南
================================================================================

高风险特征组合:
  ✓ 恶性概率 > 0.5
  ✓ 平均HU值 > -100（高密度/实性）
  ✓ 大小 > 8mm
  ✓ 形状不规则（低球形度 < 0.6）
  ✓ 高熵值（纹理复杂）

低风险特征组合:
  ✓ 恶性概率 < 0.3
  ✓ 平均HU值 < -400（低密度/磨玻璃）
  ✓ 大小 < 6mm
  ✓ 形状规则（高球形度 > 0.8）
  ✓ 低熵值（纹理均匀）

================================================================================
    """
    print(explanation)

if __name__ == "__main__":
    try:
        main()
        print_feature_explanation()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

