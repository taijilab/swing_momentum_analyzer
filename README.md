# 甩手动量分析器

基于 MediaPipe 姿态检测的视频分析工具，自动检测甩手动作并计算动量值。

## 功能特性

- 📹 支持多种视频格式 (MP4, AVI, MOV, MKV, WEBM)
- 🎯 自动检测手腕关键点位置
- 📊 实时计算手腕高度变化曲线
- ⚡ 计算甩手动作的动量值（从最高点到最低点）
- 📈 可视化展示分析结果
- 🎬 输出带骨骼标注的分析视频

## 安装

```bash
# 克隆项目
cd swing_momentum_analyzer

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 目录结构

```
swing_momentum_analyzer/
├── backend/
│   └── app.py              # Flask 后端服务
├── frontend/
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css   # 样式文件
│   │   └── js/
│   │       └── app.js      # 前端脚本
│   └── templates/
│       └── index.html      # 主页面
├── uploads/                # 上传文件目录
├── output/                 # 分析结果目录
└── requirements.txt        # 依赖列表
```

## 使用方法

```bash
# 启动服务
cd backend
python app.py
```

服务将在 http://localhost:5000 启动。

1. 打开浏览器访问 http://localhost:5000
2. 上传或拖拽视频文件到上传区域
3. 等待分析完成
4. 查看分析结果：
   - 左右手统计数据（最大/最小高度、活动范围、动量值）
   - 高度变化曲线图
   - 动量变化曲线图
   - 带骨骼标注的分析视频

## 动量计算说明

- **高度**: 视频画面的垂直位置，转换为 0-100% 范围
- **速度**: 手腕位置随时间的变化率 (像素/秒)
- **动量**: 假设手臂质量为 1kg，动量 = 质量 × 速度
  - 正值: 向下运动
  - 负值: 向上运动

## 技术栈

- **后端**: Flask + MediaPipe + OpenCV
- **前端**: HTML5 + CSS3 + JavaScript + Chart.js
- **姿态检测**: Google MediaPipe Pose

## 系统要求

- Python 3.8+
- 摄像头或视频文件
- 推荐使用 GPU 加速（可选）
