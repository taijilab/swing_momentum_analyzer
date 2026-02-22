#!/usr/bin/env python3
"""
甩手动量分析器 - 启动脚本
"""

import os
import sys

def main():
    """启动应用"""

    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(script_dir, 'backend')

    # 添加 backend 到路径
    sys.path.insert(0, backend_dir)

    # 检查依赖
    try:
        import flask
        import cv2
        import mediapipe
    except ImportError as e:
        print("缺少依赖包，请运行: pip install -r requirements.txt")
        sys.exit(1)

    # 导入应用
    from app import app

    print("\n" + "="*50)
    print("甩手动量分析器已启动")
    print("="*50)
    print("访问地址: http://localhost:8888")
    print("按 Ctrl+C 停止服务")
    print("="*50 + "\n")

    # 使用 use_reloader=False 避免路径问题
    app.run(debug=True, host='127.0.0.1', port=8888, use_reloader=False)

if __name__ == '__main__':
    main()
