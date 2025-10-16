# AI在肺部健康中的应用：跨多个CT扫描数据集的检测和诊断模型基准测试 [![arXiv](https://img.shields.io/badge/arXiv-2405.04605-<color>.svg)](https://arxiv.org/abs/2405.04605)

# 摘要

肺癌仍然是全球癌症相关死亡的主要原因，而通过低剂量计算机断层扫描(LDCT)进行早期检测已显示出在降低死亡率方面的巨大前景。随着人工智能(AI)在医学影像中的日益融合，强大的AI模型的开发和评估需要访问大型、标注良好的数据集。在本研究中，我们介绍了杜克大学肺癌筛查(DLCS)数据集的实用性，这是最大的开放访问LDCT数据集，拥有超过2,000次扫描和3,000个专家验证的结节。我们在内部和外部数据集（包括LUNA16、LUNA25和NLST-3D+）上对用于3D结节检测和肺癌分类的深度学习模型进行了基准测试。对于检测任务，我们开发了两个基于MONAI的RetinaNet模型(DLCSD-mD和LUNA16-mD)，使用竞赛性能指标(CPM)进行评估。对于分类任务，我们比较了五个模型，包括最先进的预训练模型(Models Genesis、Med3D)、一个自监督基础模型(FMCB)、一个随机初始化的ResNet50，以及提出的一个新颖的战略热启动++(SWS++)模型。SWS++使用精选的候选补丁在同一检测流程内预训练分类主干，实现了任务相关的特征学习。我们的模型展示了强大的泛化能力，SWS++在多个数据集上实现了与现有基础模型相当或更优的性能(AUC: 0.71–0.90)。所有代码、模型和数据均已公开发布，以促进可重复性和协作。这项工作为肺癌AI研究建立了标准化的基准测试资源，支持模型开发、验证和临床转化方面的未来工作。


### 引用论文

[![arXiv](https://img.shields.io/badge/arXiv-2405.04605-<color>.svg)](https://arxiv.org/abs/2405.04605)


```ruby
@article{tushar2024ai,
  title={AI in Lung Health: Benchmarking Detection and Diagnostic Models Across Multiple CT Scan Datasets},
  author={Tushar, Fakrul Islam and Wang, Avivah and Dahal, Lavsen and Harowicz, Michael R and Lafata, Kyle J and Tailor, Tina D and Lo, Joseph Y},
  journal={arXiv preprint arXiv:2405.04605},
  year={2024}
}
```
```ruby
Tushar, Fakrul Islam, et al. "AI in Lung Health: Benchmarking Detection and Diagnostic Models Across Multiple CT Scan Datasets." arXiv preprint arXiv:2405.04605 (2024).
```
### 引用数据集 - Duke Lung

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13799069.svg)](https://doi.org/10.5281/zenodo.13799069) 
```ruby
A. Wang, F. I. TUSHAR, M. R. Harowicz, K. J. Lafata, T. D. Tailorand J. Y. Lo, "Duke Lung Cancer Screening Dataset 2024". Zenodo, Mar. 05, 2024. doi: 10.5281/zenodo.13799069.
```

## 🚀 更新日志

- **[1] 2025年3月5日** - 📢 公开发布**训练模型权重**。
- **[2] 2025年3月7日** - 🖼️ 添加了**DLCSD24**的**可视化脚本**。
- **[3]** - 📂 公开发布**预处理数据集** ![即将推出](https://img.shields.io/badge/Status-Coming%20Soon-orange)。
- **[4]** - 📊 在**LUNA25数据集**上进行基准测试 ![即将推出](https://img.shields.io/badge/Status-Coming%20Soon-orange)。
- **[5]** - 🔍 使用**Vista3D、nnUNetv2**和**[SYN-LUNGS](https://github.com/fitushar/SYN-LUNGS)**对DLCSD24结节进行**伪分割**。![即将推出](https://img.shields.io/badge/Status-Coming%20Soon-orange)
- **[6]** - ⚙️ **基于ML的分割和放射组学分类基准测试** ![即将推出](https://img.shields.io/badge/Status-Coming%20Soon-orange)。
- **[7]** - 🎯 模型预测的**事后可视化**及相关代码 ![即将推出](https://img.shields.io/badge/Status-Coming%20Soon-orange)。

# 相关研究：
### 精细化肺癌AI焦点：比较病灶中心和胸部区域模型及内外部验证的性能洞察。 [![arXiv](https://img.shields.io/badge/arXiv-2411.16823-<color>.svg)](https://arxiv.org/abs/2411.16823)
### 瘤周扩展放射组学用于改进肺癌分类。 [![arXiv](https://img.shields.io/badge/arXiv-2411.16008-<color>.svg)](https://arxiv.org/abs/2411.16008)

# 基准模型权重可从此处下载：
所有开发的模型权重均可在以下位置公开获取：
📥 Zenodo: https://zenodo.org/records/14967976

Model Genesis和MedicalNet3D的预训练权重可从此处下载：
* [Genesis_Chest_CT.pt](https://drive.google.com/file/d/16iIIRkl6zYAfQ14i9NOakwFd6w_xKBSY/view?usp=sharing)
* [resnet_50_23dataset.pth](https://drive.google.com/file/d/1dIyJd3jpz9mBx534UA7deqT7f8N0sbJL/view?usp=sharing)




# 数据集

## 杜克大学肺癌筛查数据集2024 (DLCS 2024) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13799069.svg)](https://doi.org/10.5281/zenodo.13799069) 

**背景：** 肺癌风险分类是一个日益重要的研究领域，因为低剂量胸部CT筛查项目已成为肺癌高危患者的标准护理。目前缺乏大型、标注完善的公共数据库用于肺结节分类算法的训练和测试。

**方法：** 本研究考虑了2015年1月1日至2021年6月30日期间在杜克大学健康系统进行的筛查性胸部CT扫描。通过使用在LUNA16数据集上训练的公开可用深度学习结节检测算法来识别初始候选者，然后根据放射学文本报告中的结节位置进行接受，或由医学生和经过培训的心胸放射科专家手动标注，从而实现了高效的结节标注。

**结果：** 该数据集包含1613个CT体积和2487个标注的结节，从总共2061名患者的数据集中选择，其余数据保留用于未来测试。放射科医生抽查确认半自动标注的准确率超过90%。

**结论：** 杜克大学肺癌筛查数据集2024是首个反映当前CT技术使用的大型CT筛查肺癌数据集。这代表了肺癌风险分类研究的有用资源，其创建中描述的高效标注方法可用于在未来生成类似的研究数据库。

## [DLCSD24标注可视化](https://github.com/fitushar/AI-in-Lung-Health-Benchmarking-Detection-and-Diagnostic-Models-Across-Multiple-CT-Scan-Datasets/blob/main/Visualize_DLCSD24.ipynb)

此笔记本提供了一个脚本，通过在CT扫描上叠加**标注框**来可视化**DLCSD24**标注。它还允许过滤特定的数据集拆分（训练、验证、测试）以进行针对性分析。设置数据集路径以访问原始CT扫描和相应的元数据，它还允许过滤特定的数据集拆分（训练、验证、测试）以进行针对性分析。

```python
raw_data_path = 'path/to/DLCS24/'
dataset_csv   = 'path/to/Zenodo_metadata/DLCSD24_Annotations.csv'
Final_dect    = df[(df['benchmark_split']=='test')]['ct_nifti_file'].unique()
```

**重要：** ⚠️
**DLCSD24**和**NLST**数据集在绘制可视化时使用略有不同的**图像坐标系统**。
要正确地在**CT图像上叠加标注**，请遵循提供的脚本以确保正确的坐标对齐。
使用不正确的坐标系统可能导致可视化错位和解释中的潜在混淆。



## [NLST](https://github.com/fitushar/AI-in-Lung-Health-Benchmarking-Detection-and-Diagnostic-Models-Across-Multiple-CT-Scan-Datasets/tree/main/NLST_Data_Annotations)

对于[国家肺部筛查试验(NLST)](https://www.nejm.org/doi/full/10.1056/NEJMoa1208962)，在检测评估中，我们使用了[Mikhael等人(2023)](https://doi.org/10.1200/JCO.22.01345)提供的开放访问标注。我们将来自900多名肺癌患者的9,000多个2D切片级边界框标注转换为3D表示，产生了1,100多个结节标注。


为了从2D标注中提取3D标注，我们首先验证了DICOM图像中的2D标注。然后，我们从DICOM头部提取了`seriesinstanceuid`、`slice_location`和`slice_number`。随后，图像坐标位置被转换为世界坐标。在相应的NIFTI图像中验证这些标注后，我们将同一病灶在多个切片上的重叠连续2D标注连接成单个3D标注。

生成3D标注的完整代码以及显示这些标注的可视化脚本将很快发布。可视化预览显示在此[Jupyter笔记本](https://github.com/fitushar/AI-in-Lung-Health-Benchmarking-Detection-and-Diagnostic-Models-Across-Multiple-CT-Scan-Datasets/blob/main/NLST_Data_Annotations/3D_Annotation_Visualizations_NLST.ipynb)中。

## LUNA16

[LUNA16](https://luna16.grand-challenge.org/)是LIDC-IDRI数据集的精炼版本，用于外部验证，应用标准的10折交叉验证程序进行肺结节检测。对于使用LUNA16进行癌症诊断分类，我们遵循了先前研究([Pai, S.等人(2024)](https://www.nature.com/articles/s42256-024-00807-9))的标注方案，该方案指定了至少有一位放射科医师指示为恶性的结节，得到677个标注结节。此方案称为"放射科医师视觉评估恶性指数"(RVAMI)。



# 基准测试 - 结节检测

肺癌（结节）检测任务定义为在3D CT扫描中识别肺结节并使用3D边界框对其进行定位。为实现这一目标，我们利用[MONAI](https://github.com/Project-MONAI/tutorials/tree/main/detection)检测工作流程来训练和验证基于RetinaNet的3D检测模型，从而能够直接实现我们的基准模型。

* **DLCSD-mD：** 使用DLCSD开发数据集开发的模型，经过300个epoch的训练，在20%的开发集上进行验证以确保选择最佳模型
* **LUNA16-mD：** 使用[MONAI教程文档](https://github.com/Project-MONAI/tutorials/tree/main/detection)中的官方LUNA16 10折交叉验证训练的模型。


#### 预处理
所有CT体积都重新采样到0.7 × 0.7 × 1.25毫米(x, y, z)的标准化分辨率。图像的强度值被截断在-1000到500 HU之间，每个体积都被归一化为均值0和标准差1。模型使用大小为192 × 192 × 80 (x, y, z)的3D补丁进行训练，并在预测阶段应用滑动窗口方法以覆盖整个体积。所有模型都使用相同的超参数训练300个epoch，并根据最低验证损失选择最佳模型。

#### 评估指标
使用自由响应接收者操作特征(FROC)分析评估模型性能，该分析测量各种假阳性率(FPR)下的灵敏度。主要性能指标是在预定义FPR下的平均灵敏度：每次扫描1/8、1/4、1/2、1、2、4和8个假阳性，如先前研究中所述。此外，使用接收者操作特征曲线下面积(AUC)以及96%置信区间(CI)评估病灶级别的性能。

## DLCSD-mD 运行和示例

### | DLCSD-mD 1.1 数据预处理


### | DLCSD-mD 1.2 训练配置和环境文件

我们提供了预处理的数据拆分json文件，可在以下位置找到：**/ct_detection/datasplit_folds/DukeLungRADs_trcv4_fold1.json**，模型需要用于训练/验证/评估。

首先请打开**"Config_DukeLungRADS_BaseModel_epoch300_patch192x192y80z.json"**，更改以下值：

* **"Model_save_path_and_utils"：** 将创建和存储bash、config、result、tfevent_train和trained_model文件夹的目录。
* **"raw_img_path"：** 存储重新采样图像的目录。
* **"dataset_info_path"：** 如果需要，存储元数据的目录。
* **"train_cinfig"：** 在此配置文件中定义的训练超参数。
* **"bash_path"：** 保存包含模型运行命令的bash文件的目录


#### Config_DukeLungRADS_BaseModel_epoch300_patch192x192y80z.json
```ruby
{
  "Model_save_path_and_utils": "path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/",
  "raw_img_path"             : "path/to/Data/LungRADS_resample/",
  "dataset_info_path"        : "path/to/ct_detection/dataset_files/",
  "dataset_split_path"       : "path/to/ct_detection/datasplit_folds/",
  "number_of_folds"          : 4,
  "seed"                     : 200,
  "run_prefix"               : "DukeLungRADS_BaseModel_epoch300_patch192x192y80z",
  "split_prefix"             : "DukeLungRADs_trcv4_fold",
  "train_cinfig"             : "path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/training_config.json",
  "bash_path"                : "path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/bash/"
}

```

#### training_config.json

```ruby
{
	"gt_box_mode": "cccwhd",
	"lr": 1e-2,
	"spacing": [0.703125, 0.703125, 1.25],
	"batch_size": 3,
	"patch_size": [192,192,80],
    "val_interval":  5,
    "val_batch_size": 1,
	"val_patch_size": [512,512,208],
	"fg_labels": [0],
	"n_input_channels": 1,
	"spatial_dims": 3,
	"score_thresh": 0.02,
	"nms_thresh": 0.22,
	"returned_layers": [1,2],
	"conv1_t_stride": [2,2,1],
	"max_epoch": 300,
	"base_anchor_shapes": [[6,8,4],[8,6,5],[10,10,6]],
	"balanced_sampler_pos_fraction": 0.3,
	"resume_training": false,
	"resume_checkpoint_path": "",
  "cached_dir":  "/path/to/data/cache/"
}

```

### | DLCSD-mD 1.3 生成训练/验证环境和Bash文件

```ruby
bash run.sh
```
#### run.sh
```ruby

python3 /path/to/ct_detection/env_main.py --config /path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/Config_DukeLungRADS_BaseModel_epoch300_patch192x192y80z.json
python3 /path/to/ct_detection/bash_main_cvit.py --config /path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/Config_DukeLungRADS_BaseModel_epoch300_patch192x192y80z.json

```

### | DLCSD-mD 1.4 训练/验证

该模型已使用singularity在集群上训练，运行创建的sub文件将启动训练

**创建日志文件夹：/path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/bash/slurm_logs/**
```ruby
sbatch run_DukeLungRADS_BaseModel_epoch300_patch192x192y80z_fold1.sub
```
#### run_DukeLungRADS_BaseModel_epoch300_patch192x192y80z_fold1.sub

```ruby
#!/bin/bash

#SBATCH --job-name=CVIT-VNLST_1
#SBATCH --mail-type=END,FAIL    
#SBATCH --mail-user=fakrulislam.tushar@duke.edu
#SBATCH --nodes=1
#SBATCH -w node001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --output=/path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/bash/slurm_logs/run_DukeLungRADS_BaseModel_epoch300_patch192x192y80z_fold1_.%j.out
#SBATCH --error=/path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/bash/slurm_logs/run_DukeLungRADS_BaseModel_epoch300_patch192x192y80z_fold1._%j.err

module load singularity/singularity.module
export NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

echo "VNLST Run "
echo "Job Running On "; hostname
echo "Nvidia Visible Devices: $NVIDIA_VISIBLE_DEVICES"

singularity run --nv --bind /path/to /home/ft42/For_Tushar/vnlst_ft42_v1.sif python3 /path/to/ct_detection/training.py -e /path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/config/environment_DukeLungRADS_BaseModel_epoch300_patch192x192y80z_fold1.json -c /path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/training_config.json
singularity run --nv --bind /path/to /home/ft42/For_Tushar/vnlst_ft42_v1.sif python3 /path/to/ct_detection/testing.py -e /path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/config/environment_DukeLungRADS_BaseModel_epoch300_patch192x192y80z_fold1.json -c /path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/training_config.json

```

* **您也可以选择使用Docker容器和简单的python调用，在这种情况下，请检查[fitushar/Luna16_Monai_Model_XAI_Project](https://github.com/fitushar/Luna16_Monai_Model_XAI_Project)中提到的docker容器要求**

```ruby
python3 /path/to/ct_detection/training.py -e /path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/config/environment_DukeLungRADS_BaseModel_epoch300_patch192x192y80z_fold1.json -c /path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/training_config.json

python3 /path/to/ct_detection/testing.py -e /path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/config/environment_DukeLungRADS_BaseModel_epoch300_patch192x192y80z_fold1.json -c /path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/training_config.json
```


### | DLCSD-mD 1.5 测试

🚀 即将推出

### | DLCSD-mD 1.6 评估和基准测试


🚀 即将推出


### 代码

🚀 即将推出


# 基准测试 - 肺癌分类任务


我们将肺癌分类任务定义为给定一个结节，将其分类为癌症或非癌症。为了对肺癌分类任务进行基准测试，我们采用了五种不同的基线模型，包括随机初始化、监督和自监督预训练模型，以及我们内部提出的战略热启动++(SWS++)模型。


![癌症分类](https://github.com/fitushar/AI-in-Lung-Health-Benchmarking-Detection-and-Diagnostic-Models-Across-Multiple-CT-Scan-Datasets/blob/main/readme_figures/cancer_classifications_1.PNG)

* **3D ResNet50**
* **FMCB：** 我们采用了最近发布的基于自监督ResNet50的基础模型，称为"FMCB"。我们使用它为每个数据点提取4,096个特征，并按照作者的建议使用scikit-learn框架训练逻辑回归模型。[Pai, S.等人(2024)](https://www.nature.com/articles/s42256-024-00807-9)
* **Genesis：** Models-Genesis的预训练ResNet50，在其上添加了分类层并进行端到端训练。[Zhou, Z.等人(2021)](https://www.sciencedirect.com/science/article/pii/S1361841520302048)
* **MedNet3D：** Med3D的预训练ResNet50，我们在其上添加了分类层并进行端到端训练。[Chen,S.等人(2019)](https://arxiv.org/abs/1904.00625)
* **ResNet50-SWS++：** 我们使用我们新颖的战略热启动++(SWS++)预训练方法开发了一个内部模型。该方法涉及训练ResNet50以减少肺结节检测中的假阳性，使用基于结节置信度分数的仔细分层数据集。然后将得到的模型"ResNet50-SWS++"微调用于端到端肺癌分类。[Tushar, F. I.等人(2024)](https://arxiv.org/abs/2405.04605)



# 训练肺癌分类模型

## ResNet50训练

```ruby
python3 path/to/ct_classification/training_AUC_StepLR.py -c /path/to/ct_classification/Model_resnet50/config_train_f1_resnet50.json
```

**|config_train_f1_resnet50.json**

```ruby
{
"Model_save_path_and_utils": "path/to/Model_resnet50/",
"run_prefix"               : "Model_resnet50",
"which_fold"               : 1,

"training_csv_path"   : "/path/to/fold_1_tr.csv",
"validation_csv_path" : "/path/to/fold_1_val.csv",
"data_column_name"    : "unique_Annotation_id_nifti",
"label_column_name"   : "Malignant_lbl",

"training_nifti_dir"    : "path/to/nifti/",
"validation_nifti_dir"  : "path/to/nifti/",

"image_key"           : "img",
"label_key"           : "label",
"img_patch_size"      : [64, 64, 64],
"cache_root_dir"      : "path/to/cache_root_dir/",


"train_batch_size" : 24,
"val_batch_size"   : 24,
"use_sampling"     : false,
"sampling_ratio"   : 1,
"num_worker"       : 8,
"val_interval"     : 5,
"max_epoch"        : 200,
"Model_name"       : "resnet50",
"spatial_dims"     : 3,
"n_input_channels" : 1,
"num_classes"      : 2,
"lr"               : 1e-2,
"resume_training": false,
"resume_checkpoint_path": ""

}
```


## Model Genesis训练

```ruby
python3 path/to/ct_classification/training_AUC_StepLR.py -c /path/to/ct_classification/Model_Genesis_FineTuning/config_train_f1_modelGenesis.json
```

**|config_train_f1_modelGenesis.json**

```ruby
{
"Model_save_path_and_utils": "path/to/Model_Genesis_FineTuning/",
"run_prefix"               : "DukeLungRADS_Genesis_FineTuning",
"which_fold"               : 1,

"training_csv_path"   : "/path/to/fold_1_tr.csv",
"validation_csv_path" : "/path/to/fold_1_val.csv",
"data_column_name"    : "unique_Annotation_id_nifti",
"label_column_name"   : "Malignant_lbl",

"training_nifti_dir"    : "path/to/nifti/",
"validation_nifti_dir"  : "path/to/nifti/",

"image_key"           : "img",
"label_key"           : "label",
"img_patch_size"      : [64, 64, 64],
"cache_root_dir"      : "path/to/cache_root_dir/",


"train_batch_size" : 24,
"val_batch_size"   : 24,
"use_sampling"     : false,
"sampling_ratio"   : 1,
"num_worker"       : 8,
"val_interval"     : 5,
"max_epoch"        : 200,
"Model_name"       : "Model_Genesis",
"spatial_dims"     : 3,
"n_input_channels" : 1,
"num_classes"      : 2,
"lr"               : 1e-2,
"resume_training": false,
"resume_checkpoint_path": ""

}
```

### Model MedNet3D训练

```ruby
python3 path/to/ct_classification/training_AUC_StepLR.py -c /path/to/ct_classification/Model_MedicalNet3D_FineTuning/config_train_f1_MedicalNet3D_resnet50.json
```

**|config_train_f1_MedicalNet3D_resnet50.json**
```ruby
{
"Model_save_path_and_utils": "/path/to/Model_MedicalNet3D_FineTuning/",
"run_prefix"               : "DukeLungRADS_MedicalNet3D_FineTuning",
"which_fold"               : 1,

"training_csv_path"   : "/path/to/fold_1_tr.csv",
"validation_csv_path" : "/path/to/fold_1_val.csv",
"data_column_name"    : "unique_Annotation_id_nifti",
"label_column_name"   : "Malignant_lbl",

"training_nifti_dir"    : "path/to/nifti/",
"validation_nifti_dir"  : "path/to/nifti/",

"image_key"           : "img",
"label_key"           : "label",
"img_patch_size"      : [64, 64, 64],
"cache_root_dir"      : "path/to/cache_root_dir/",


"train_batch_size" : 24,
"val_batch_size"   : 24,
"use_sampling"     : false,
"sampling_ratio"   : 1,
"num_worker"       : 8,
"val_interval"     : 5,
"max_epoch"        : 200,
"Model_name"       : "resnet50_MedicalNet3D",
"spatial_dims"     : 3,
"n_input_channels" : 1,
"num_classes"      : 2,
"lr"               : 1e-2,
"resume_training": false,
"resume_checkpoint_path": ""

}
```
# 引用文献

* Tushar, Fakrul Islam, et al. "AI in Lung Health: Benchmarking Detection and Diagnostic Models Across Multiple CT Scan Datasets." arXiv preprint arXiv:2405.04605 (2024).
* A. Wang, F. I. TUSHAR, M. R. Harowicz, K. J. Lafata, T. D. Tailorand J. Y. Lo, "Duke Lung Cancer Screening Dataset 2024". Zenodo, Mar. 05, 2024. doi: 10.5281/zenodo.13799069.
* Mikhael, Peter G., et al. "Sybil: a validated deep learning model to predict future lung cancer risk from a single low-dose chest computed tomography." Journal of Clinical Oncology 41.12 (2023): 2191-2200.
* Pai, S., Bontempi, D., Hadzic, I. et al. Foundation model for cancer imaging biomarkers. Nat Mach Intell 6, 354–367 (2024). https://doi.org/10.1038/s42256-024-00807-9
* Cardoso, M. Jorge, et al. "Monai: An open-source framework for deep learning in healthcare." arXiv preprint arXiv:2211.02701 (2022).
* Z. Zhou, V. Sodha, J. Pang, M. B. Gotway, and J. Liang, "Models genesis," Medical image analysis, vol. 67, p. 101840, 2021.
* S. Chen, K. Ma, and Y. Zheng, "Med3d: Transfer learning for 3d medical image analysis," arXiv preprint arXiv:1904.00625, 2019.
* National Lung Screening Trial Research Team. "Results of initial low-dose computed tomographic screening for lung cancer." New England Journal of Medicine 368.21 (2013): 1980-1991.
* Tushar, Fakrul Islam, et al. "Virtual NLST: towards replicating national lung screening trial." Medical Imaging 2024: Physics of Medical Imaging. Vol. 12925. SPIE, 2024.
* Tushar, Fakrul Islam, et al. "VLST: Virtual Lung Screening Trial for Lung Cancer Detection Using Virtual Imaging Trial." arXiv preprint arXiv:2404.11221 (2024).

