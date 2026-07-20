\# 🅿️ Parking Space Occupancy Detection \& License Plate Enhancement System

\# 停车位占用检测与车牌区域增强系统（MATLAB）



基于 MATLAB 实现的智能停车场管理辅助系统。本项目集成了\*\*自动车位网格划分、车位占用状态多特征判定、车位车牌定位（HSV+边缘融合）、频域图像增强\*\*以及\*\*包含防错映射的鲁棒车牌 OCR 识别\*\*功能。



\---



\## ✨ 核心功能特性



1\. \*\*自动车位划分与占用检测\*\*：

&#x20;  \* 支持通过自适应网格精准划分车位区域。

&#x20;  \* 综合运用图像灰度均值、方差及 Canny 边缘密度等多特征进行车位占用判定（`Occupied` / `Free`）。

2\. \*\*车牌区域定位\*\*：

&#x20;  \* 针对车辆区域结合 \*\*HSV 色彩空间（蓝色车牌特征）\*\* 与 \*\*Canny 边缘特征\*\* 进行双重级联定位，确保定位鲁棒性。

3\. \*\*频域滤波与图像增强\*\*：

&#x20;  \* 采用\*\*巴特沃斯低通滤波器（Butterworth Lowpass Filter）\*\*分离低频背景，实现高频细节增强与 `imadjust` 动态范围调整，使模糊车牌清晰可见。

4\. \*\*鲁棒的中文车牌 OCR 识别\*\*：

&#x20;  \* 集成多阈值二值化策略（自适应、Otsu、固定阈值）。

&#x20;  \* 包含\*\*中国大陆车牌格式强制校验与防误识别映射\*\*（如首位汉字自动纠错、数字/字母易混淆字符修正），大幅提高识别准确率。



\---



\## 🛠️ 环境依赖 (Prerequisites)



请确保你的 MATLAB 安装了以下工具箱 (Toolboxes)：

\* \*\*MATLAB\*\* (推荐 R2021a 及以上版本)

\* \*\*Computer Vision Toolbox\*\*

\* \*\*Image Processing Toolbox\*\*



\---



\## 🚀 快速开始 (Quick Start)



1\. 克隆或下载本仓库到本地。

2\. 将 `src/parking\_system.m` 在 MATLAB 中打开。

3\. 在脚本顶部的 \*\*用户设置\*\* 区域根据需求调整参数：

&#x20;  ```matlab

&#x20;  useSimulated = false;          % false: 选择真实图像; true: 模拟图像

&#x20;  enableChinese = true;          % 是否在 OCR 字符集中包含常见汉字

&#x20;  detectWholeImagePlate = false; % false: 执行车位检测与车牌增强流程; true: 直接识别整图车牌

