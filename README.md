# KAN-FIF: Spline-Parameterized Lightweight Physics-based Tropical Cyclone Estimation on Meteorological Satellite
## 📘 Overview

Tropical cyclones (TC) are among the most destructive natural disasters, causing catastrophic damage to coastal regions through extreme winds, heavy rainfall, and storm surges. Timely monitoring of tropical cyclones is critical to reduce the loss of life and property, yet it is hampered by computational inefficiency and high parameter counts of existing methods on resource-constrained edge devices. Current physics-guided models suffer from linear feature interactions that fail to capture high-order polynomial relationships between TC attributes, leading to inflated model sizes and hardware incompatibility.

To overcome these challenges, this study introduces the **K**olmogorov–**A**rnold **N**etwork-based **F**eature **I**nteraction **F**ramework (**KAN-FIF**), a lightweight multimodal architecture that systematically replaces MLP and CNN layers with spline-parameterized KAN layers. 

---

## 📁 Directory Structure

```
KAN-FIF-github/
├── README.md                 # This file
├── gpu_test/                 # GPU testing code
│   ├── TCtest.py            # GPU test script
│   ├── TCtrain.py           # Training script
│   ├── config.py             # Configuration file
│   ├── dataset.py            # Dataset class
│   └── net/                  # Network structure
│       ├── __init__.py
│       ├── KAN_FIF_net.py   # Main network structure
│       └── kan.py            # KAN layer implementation
├── ascend_test/              # Ascend NPU testing code
│   ├── Ascend_test.py       # Ascend test script
│   ├── Ascend_config.py     # Ascend configuration file
│   └── Ascend_dataset.py    # Ascend dataset class
├── models/                   # Model files
│   ├── checkpoint_20_scheduled_05191037.pth  # Model weights for GPU testing
│   └── deploy.om             # OM model for Ascend NPU inference
└── data/                     # Data directory
    └── test/                 # Test data (.npy files)
```

---

## ⚙️ Environment Setup

### GPU Testing Environment
- Python ≥ 3.7
- PyTorch ≥ 1.8.0
- CUDA ≥ 10.2 (if using GPU)
- Dependencies:
  ```bash
  torch
  numpy
  tqdm
  ```

### Ascend Testing Environment
- Python ≥ 3.7
- Huawei Ascend AI Processor (Ascend 310)
- ais_bench inference tool
- Dependencies:
  ```bash
  torch
  numpy
  tqdm
  ais_bench
  ```

---

## 🚀 Installation

### GPU Environment Installation

```bash
# Install PyTorch (select according to your CUDA version)
pip install torch torchvision torchaudio

# Install other dependencies
pip install numpy tqdm
```

### Ascend Environment Installation

On Huawei Ascend development boards, you need to install:
- CANN toolkit
- ais_bench inference tool

Please refer to the official Huawei Ascend documentation and our paper for detailed installation steps.

---

## 📊 Data Preparation

### Data Format

Test data should be in `.npy` format with the following naming convention:
```
{lat}_{lon}_{t}_{pre_category}_{pressure}_{wind}_{rmw}_{...}_{typhoon_name}_{obs_time}.npy
```

Where:
- `lat`, `lon`: Latitude and longitude
- `t`: Time feature
- `pre_category`: Precipitation category
- `pressure`: Pressure
- `wind`: Maximum wind speed (target value)
- `rmw`: Radius of maximum winds (target value)
- `typhoon_name`: Typhoon name
- `obs_time`: Observation time (format: YYYYMMDDHH)

Place test data files (.npy format) in the `data/test/` directory.

test data: 通过网盘分享的文件：test 链接: https://pan.baidu.com/s/1ltUp2b6TNSWajda-bQ1k1w?pwd=c9rg 提取码: c9rg --来自百度网盘超级会员v5的分享
models: 通过网盘分享的文件：models 链接: https://pan.baidu.com/s/1OMOXWLUD4mr18wjq4puJpQ?pwd=rwui 提取码: rwui --来自百度网盘超级会员v5的分享
---

## 🧪 Usage

### GPU Testing

1. **Modify Configuration File**

   Edit `gpu_test/config.py` to modify data paths and model paths:

   ```python
   class DataConfig:
       test_path = "data/test"  # Modify to your test data path
       batch_size = 128
       seq_length = 3
   ```

2. **Run Testing**

   ```bash
   cd gpu_test
   python TCtest.py
   ```


---

## 🏋️ Training

### Training Steps

1. **Prepare Training Data**

   Place training data in the specified directory and configure paths in `gpu_test/config.py`:

   ```python
   class DataConfig:
       train_path = "data/train"    # Training data path
       valid_path = "data/valid"    # Validation data path
   ```

2. **Configure Training Parameters**

   Modify training configuration in `gpu_test/config.py`:

   ```python
   class TrainConfig:
       epochs = 200
       lr = 0.001
       loss_weights = [0.3, 0.7]  # [wind_loss_weight, rmw_loss_weight]
       early_stop_threshold = 0.01
       save_interval = 10
   ```

3. **Start Training**

   ```bash
   cd gpu_test
   python TCtrain.py
   ```



### Model Files

- `checkpoint_20_scheduled_05191037.pth`: PyTorch format model weights file for GPU testing
- `deploy.om`: Huawei Ascend OM format model file for Ascend NPU inference

### Model Parameters

- **Input**:
  - Temporal features: `[batch_size, seq_length=3, features=5]`
  - Image features: `[batch_size, channels=8, height=156, width=156]`
  
- **Output**:
  - Wind prediction value (normalized)
  - RMW prediction value (normalized)

---

## 📈 Experimental Results

Model performance on the test set:

**NVIDIA GPU:**
- Size: 0.99M
- InferTime: 2.30ms
- Wind MAE: 3.21
- Wind RMSE: 4.31
- RMW MAE: 8.83
- RMW RMSE: 11.66

**Ascend 310 NPU:**
- Size: 0.92M
- InferTime: 14.41ms
- Wind MAE: 6.66
- Wind RMSE: 9.78
- RMW MAE: 9.37
- RMW RMSE: 12.22

---

## ✏️ Citation

If you use this code or dataset in your research, please cite the following paper:

> [Your citation information here]

---
