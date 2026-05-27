# AI Car Reinforcement Learning 🚗🧠
> 深度學習期末專案 (Deep Learning Final Project)

本專案是一個基於強化學習 (Reinforcement Learning) 與 NEAT 演算法的 2D 自駕車模擬環境。AI 能夠透過車載雷達感測器學習賽道邊界，並自動進化出能夠完美繞行賽道的駕駛策略。

本專案的核心概念與基礎程式碼引用自 [NeuralNine/ai-car-simulation](https://github.com/NeuralNine/ai-car-simulation)，並在此基礎上進行了深度的功能擴充與架構重構。

## ✨ 專案特色與擴充功能 (Features)

相較於原始的 NeuralNine 版本，本專案針對學術專案需求增加了以下實用功能：
- **獨立的訓練與評估架構**：將流程拆分為 `train.py` (模型訓練) 與 `eval.py` (模型驗證)，方便保存與載入最佳模型。
- **軌跡視覺化 (Trajectory Visualization)**：新增 `trajectory_viz.py`，可將 AI 車輛的學習路線與行駛軌跡視覺化，方便進行成效分析。
- **擴充賽道地圖**：內建多種不同難度的賽道 (`assets/maps/map3.png` ~ `map6.png`)。
- **開發輔助工具**：提供 `tools/pick_coords.py` 工具，開發者可直接在畫面上點擊獲取座標，大幅簡化自訂新地圖與設定起始點的流程。
- **參數設定檔分離**：整合至 `configs.py`，讓超參數 (Hyperparameters) 與環境設定更易於管理。

## 💻 環境與硬體需求 (Requirements)

建議使用 Anaconda 建立獨立的虛擬環境，並支援在原生 Windows 或 WSL2 環境下運行。

- **推薦硬體配置**：由於涉及物理環境渲染與多世代演算法運算，推薦使用如 Intel i5-14600KF 處理器、NVIDIA RTX 3080 顯示卡及 64GB RAM 或同等級之硬體配置，以獲得最佳的訓練速度與順暢度。
- **Python 版本**：Python 3.8 或以上。

## 🚀 安裝與執行 (Installation & Usage)

1. **複製專案到本地端：**
   ```bash
   git clone [https://github.com/MDFKBH/DL-Final-AI-car-RL.git](https://github.com/MDFKBH/DL-Final-AI-car-RL.git)
   cd DL-Final-AI-car-RL
安裝所需套件：
請確保已建立好虛擬環境，接著安裝依賴套件：

Bash
pip install -r requirements.txt
開始訓練模型：

Bash
python train.py
評估訓練成果：

Bash
python eval.py
執行軌跡視覺化：

Bash
python trajectory_viz.py
📂 專案結構 (Project Structure)
Plaintext
├── assets/             # 存放賽道地圖圖片 (map3.png ~ map6.png) 與車輛素材
├── configs.py          # NEAT 演算法與環境全域參數設定
├── car_env.py          # 車輛物理引擎、雷達感測與環境互動邏輯
├── train.py            # 執行 NEAT 演算法訓練模型
├── eval.py             # 載入已訓練之模型進行效能測試
├── trajectory_viz.py   # AI 行駛軌跡分析與視覺化腳本
├── tools/
│   └── pick_coords.py  # 輔助開發工具：賽道座標擷取器
└── requirements.txt    # Python 依賴套件清單
📜 鳴謝 (Credits)
基礎自駕車模擬邏輯與 Pygame 實作參考自：NeuralNine - AI Car Simulation

NEAT (NeuroEvolution of Augmenting Topologies) 演算法實作：neat-python