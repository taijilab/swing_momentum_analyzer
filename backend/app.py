"""
视频甩手动量分析器 - 后端服务
使用 MediaPipe 进行姿态检测，计算甩手动作的动量值
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import json
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 使用绝对路径，确保从任何目录运行都能正常工作
BASE_DIR = Path(__file__).parent.parent  # 项目根目录
app.config['UPLOAD_FOLDER'] = str(BASE_DIR / 'uploads')
app.config['OUTPUT_FOLDER'] = str(BASE_DIR / 'output')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# 确保目录存在
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
    Path(folder).mkdir(exist_ok=True)


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


class SwingAnalyzer:
    """甩手动作分析器"""

    def __init__(self):
        # 初始化 MediaPipe 姿态检测
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # MediaPipe 关键点索引
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16

        # 存储分析数据
        self.data = {
            'frames': [],
            'left_wrist_y': [],
            'right_wrist_y': [],
            'left_shoulder_y': [],
            'right_shoulder_y': [],
            'left_velocity': [],
            'right_velocity': [],
            'left_momentum': [],
            'right_momentum': [],
            'swing_events': []
        }

    def calculate_velocity(self, positions, fps=30):
        """计算速度 (位置变化率)"""
        if len(positions) < 2:
            return [0] * len(positions)

        velocity = [0]  # 第一帧速度为0
        dt = 1 / fps  # 时间间隔

        for i in range(1, len(positions)):
            if positions[i] is not None and positions[i-1] is not None:
                # 计算垂直速度 (像素/秒，负值向上，正值向下)
                vy = (positions[i] - positions[i-1]) / dt
                velocity.append(vy)
            else:
                velocity.append(0)

        return velocity

    def detect_swing_cycles(self, y_positions, velocities):
        """检测甩手周期（从高点到低点）"""
        cycles = []

        if len(y_positions) < 10:
            return cycles

        # 寻找峰值和谷值
        for i in range(2, len(y_positions) - 2):
            if y_positions[i] is None:
                continue

            # 检测局部极值（高点或低点）
            is_peak = True
            is_trough = True

            for j in range(i-2, i+3):
                if j == i or y_positions[j] is None:
                    continue
                if y_positions[j] > y_positions[i]:
                    is_peak = False
                if y_positions[j] < y_positions[i]:
                    is_trough = False

            if is_peak:
                cycles.append({
                    'frame': i,
                    'type': 'peak',
                    'y': y_positions[i],
                    'velocity': velocities[i] if i < len(velocities) else 0
                })
            elif is_trough:
                cycles.append({
                    'frame': i,
                    'type': 'trough',
                    'y': y_positions[i],
                    'velocity': velocities[i] if i < len(velocities) else 0
                })

        return cycles

    def calculate_momentum(self, velocities, mass=1.0):
        """
        计算动量值
        momentum = mass * velocity
        这里我们关注垂直方向的动量变化
        """
        momentum = []
        for v in velocities:
            momentum.append(mass * v)
        return momentum

    def _smooth_positions(self, positions, window=9):
        """
        高斯平滑位置数据，减少噪声
        在计算速度前先平滑位置，可显著提高速度估算精度
        """
        if len(positions) < window:
            return list(positions)

        half = window // 2
        sigma = window / 5.0
        weights = [np.exp(-0.5 * ((k - half) / sigma) ** 2) for k in range(window)]
        w_total = sum(weights)
        weights = [w / w_total for w in weights]

        smoothed = []
        for i in range(len(positions)):
            if positions[i] is None:
                smoothed.append(None)
                continue
            s, w_sum = 0.0, 0.0
            for k in range(window):
                j = i - half + k
                if 0 <= j < len(positions) and positions[j] is not None:
                    s += weights[k] * positions[j]
                    w_sum += weights[k]
            smoothed.append(s / w_sum if w_sum > 0 else None)
        return smoothed

    def _compute_velocity(self, positions, fps):
        """
        用中心差分法计算速度（比前向差分更精确）

        坐标约定：positions 为高度百分比（0-100），值越大=手臂越高
          velocity > 0  →  手臂向上运动
          velocity < 0  →  手臂向下运动
        """
        n = len(positions)
        dt = 1.0 / fps
        velocity = []
        for i in range(n):
            if positions[i] is None:
                velocity.append(0.0)
                continue
            prev_ok = i > 0 and positions[i - 1] is not None
            next_ok = i < n - 1 and positions[i + 1] is not None
            if prev_ok and next_ok:
                v = (positions[i + 1] - positions[i - 1]) / (2.0 * dt)
            elif prev_ok:
                v = (positions[i] - positions[i - 1]) / dt
            elif next_ok:
                v = (positions[i + 1] - positions[i]) / dt
            else:
                v = 0.0
            velocity.append(v)
        return velocity

    def find_all_down_cycles(self, y_positions, momentums, fps, frames_buffer=None, shoulder_y=None):
        """向下落下动作检测（委托给 find_all_cycles，保持接口兼容）"""
        down_cycles, _ = self.find_all_cycles(y_positions, fps, shoulder_y)
        return down_cycles

    def find_all_cycles(self, y_positions, fps, shoulder_y=None):
        """
        统一的上抬/落下周期检测状态机。

        坐标约定（y_positions 为高度百分比 0-100）：
          值越大  = 手臂越高
          velocity > 0  = 手臂向上运动
          velocity < 0  = 手臂向下运动

        计数规则（修正版）：
          - 落下（down_cycle）：手臂落到最低点后速度反转向上时确认，
            记录本次落下的 peak→trough 数据
          - 上抬（up_cycle）：手臂升到最高点后速度反转向下时确认，
            记录本次上抬的 trough→peak 数据

        状态流转：
          unknown → at_peak/at_trough
          at_peak  → [确认上抬] → falling  → at_trough → [确认落下] → rising → at_peak

        返回: (down_cycles, up_cycles)
          down_cycles: 每次落下（peak→trough）的数据
          up_cycles:   每次上抬（trough→peak）的数据
        """
        if len(y_positions) < 20:
            return [], []

        # ── 1. 高斯平滑位置，消除 MediaPipe 抖动噪声 ──────────────────────
        smoothed_y = self._smooth_positions(y_positions, window=9)

        # ── 2. 中心差分计算速度（精度高于前向差分）──────────────────────────
        velocity = self._compute_velocity(smoothed_y, fps)

        # ── 3. 自适应阈值（基于实际运动范围）────────────────────────────────
        valid_y = [y for y in smoothed_y if y is not None]
        if len(valid_y) < 20:
            return [], []
        y_mean = float(np.mean(valid_y))
        y_std = float(np.std(valid_y))
        # 用均值±0.3σ作为高/低位判断，最小间距保证不小于10%
        half_band = max(0.3 * y_std, 5.0)
        HIGH_THRESHOLD = y_mean + half_band   # 高于此视为"高位"
        LOW_THRESHOLD  = y_mean - half_band   # 低于此视为"低位"

        # 速度阈值
        VEL_STRONG = 6.0  # 明显运动（触发 at_peak/at_trough → falling/rising）
        VEL_WEAK   = 2.0  # 弱反转（触发 falling/rising → at_trough/at_peak）
        MIN_HEIGHT_CHANGE = 8.0  # 有效周期最小幅度（%）

        down_cycles = []
        up_cycles   = []

        state = 'unknown'
        peak_frame,   peak_y   = None, None
        trough_frame, trough_y = None, None

        for i in range(len(smoothed_y)):
            if smoothed_y[i] is None:
                continue

            y = smoothed_y[i]
            v = velocity[i] if i < len(velocity) else 0.0
            shoulder = (shoulder_y[i]
                        if shoulder_y and i < len(shoulder_y) and shoulder_y[i] is not None
                        else None)

            # ── 初始状态：根据位置或速度确定起点 ────────────────────────────
            if state == 'unknown':
                if y >= HIGH_THRESHOLD:
                    state = 'at_peak'
                    peak_frame, peak_y = i, y
                elif y <= LOW_THRESHOLD:
                    state = 'at_trough'
                    trough_frame, trough_y = i, y
                elif v < -VEL_STRONG:          # 中间位置但明显在下落
                    state = 'falling'
                    peak_frame,   peak_y   = i, y
                    trough_frame, trough_y = i, y
                elif v > VEL_STRONG:           # 中间位置但明显在上升
                    state = 'rising'
                    trough_frame, trough_y = i, y
                    peak_frame,   peak_y   = i, y

            # ── 高位等待：持续更新最高点，等待开始下落 ───────────────────────
            elif state == 'at_peak':
                if y > peak_y:
                    peak_frame, peak_y = i, y
                # 速度明显向下 → 确认上抬周期（如有），然后开始下落
                if v < -VEL_STRONG:
                    # 如果存在完整上抬数据，记录本次上抬
                    if trough_frame is not None and peak_frame is not None:
                        height_change = peak_y - trough_y
                        if height_change >= MIN_HEIGHT_CHANGE:
                            phase_vels = [velocity[k]
                                          for k in range(trough_frame, peak_frame + 1)
                                          if k < len(velocity) and velocity[k] > 0]
                            avg_up = sum(phase_vels) / len(phase_vels) if phase_vels else 0.0
                            max_up = max(phase_vels) if phase_vels else 0.0
                            duration = (peak_frame - trough_frame) / fps
                            up_cycles.append({
                                'cycle_number': len(up_cycles) + 1,
                                'type': 'up',
                                'start_frame': trough_frame,
                                'trough_frame': trough_frame,
                                'peak_frame': peak_frame,
                                'trough_y': float(trough_y),
                                'peak_y': float(peak_y),
                                'height_change': float(height_change),
                                'avg_momentum': float(avg_up),
                                'max_momentum': float(max_up),
                                'duration': float(duration)
                            })
                            logger.info(
                                f"[上抬 #{len(up_cycles)}] "
                                f"低点帧{trough_frame}(Y={trough_y:.1f}%) → "
                                f"高点帧{peak_frame}(Y={peak_y:.1f}%), "
                                f"幅度={height_change:.1f}%, "
                                f"平均动量={avg_up:.2f}, 峰值动量={max_up:.2f}"
                            )
                    # 转入下落，重置低点追踪
                    state = 'falling'
                    trough_frame, trough_y = i, y

            # ── 下落中：持续追踪最低点，等待速度反转为正 ────────────────────
            # （不在此处记录落下，确保捕获到真正最低点后再统计）
            elif state == 'falling':
                if y < trough_y:
                    trough_frame, trough_y = i, y
                # 速度开始向上（弱阈值）→ 到达最低点，进入低位等待
                if v > VEL_WEAK:
                    state = 'at_trough'

            # ── 低位等待：持续更新最低点，等待开始上抬 ───────────────────────
            elif state == 'at_trough':
                if trough_frame is None or y < trough_y:
                    trough_frame, trough_y = i, y
                # 速度明显向上 → 确认落下周期（如有），然后开始上抬
                if v > VEL_STRONG:
                    # 如果存在完整落下数据，记录本次落下
                    if peak_frame is not None and trough_frame is not None:
                        height_change = peak_y - trough_y
                        above_shoulder = (shoulder is None or peak_y > shoulder)
                        if height_change >= MIN_HEIGHT_CHANGE:
                            phase_vels = [velocity[k]
                                          for k in range(peak_frame, trough_frame + 1)
                                          if k < len(velocity) and velocity[k] < 0]
                            avg_down = abs(sum(phase_vels) / len(phase_vels)) if phase_vels else 0.0
                            max_down = abs(min(phase_vels)) if phase_vels else 0.0
                            duration = (trough_frame - peak_frame) / fps
                            down_cycles.append({
                                'cycle_number': len(down_cycles) + 1,
                                'type': 'down',
                                'start_frame': peak_frame,
                                'peak_frame': peak_frame,
                                'trough_frame': trough_frame,
                                'peak_y': float(peak_y),
                                'trough_y': float(trough_y),
                                'height_change': float(height_change),
                                'avg_momentum': float(avg_down),
                                'max_momentum': float(max_down),
                                'duration': float(duration),
                                'above_shoulder': bool(above_shoulder)
                            })
                            logger.info(
                                f"[落下 #{len(down_cycles)}] "
                                f"高点帧{peak_frame}(Y={peak_y:.1f}%) → "
                                f"低点帧{trough_frame}(Y={trough_y:.1f}%), "
                                f"幅度={height_change:.1f}%, "
                                f"平均动量={avg_down:.2f}, 峰值动量={max_down:.2f}"
                            )
                    # 转入上抬，重置高点追踪
                    state = 'rising'
                    peak_frame, peak_y = i, y

            # ── 上抬中：持续追踪最高点，等待速度反转为负 ────────────────────
            # （不在此处记录上抬，确保捕获到真正最高点后再统计）
            elif state == 'rising':
                if peak_frame is None or y > peak_y:
                    peak_frame, peak_y = i, y
                # 速度开始向下（弱阈值）→ 到达最高点，进入高位等待
                if v < -VEL_WEAK:
                    state = 'at_peak'

        return down_cycles, up_cycles

    def find_max_down_momentum_cycle(self, y_positions, momentums, fps, frames_buffer=None):
        """
        找出左手最明显的落下动作
        返回: {peak_frame, trough_frame, momentum, peak_y, trough_y}
        """
        if len(y_positions) < 10:
            return None

        # 获取所有落下动作
        all_cycles = self.find_all_down_cycles(y_positions, momentums, fps, frames_buffer)

        if not all_cycles:
            return None

        # 选择动量最大的落下动作
        best_cycle = max(all_cycles, key=lambda x: x['max_momentum'])

        logger.info(f"[最大落下动作] 落下#{best_cycle['cycle_number']}, "
                  f"高点帧{best_cycle['peak_frame']}(高度{best_cycle['peak_y']:.1f}%) -> "
                  f"低点帧{best_cycle['trough_frame']}(高度{best_cycle['trough_y']:.1f}%)")
        logger.info(f"  动量={best_cycle['max_momentum']:.2f}, 高度差={best_cycle['height_change']:.1f}%")

        return {
            'peak_frame': best_cycle['peak_frame'],
            'trough_frame': best_cycle['trough_frame'],
            'momentum': best_cycle['max_momentum'],
            'peak_y': best_cycle['peak_y'],
            'trough_y': best_cycle['trough_y']
        }

    def create_down_motion_frames(self, frames_buffer, wrist_positions, cycle_info, fps):
        """
        创建向下运动的两张关键帧图片（高点和低点）
        只绘制左手骨架，标注统计信息
        """
        import base64

        peak_frame_idx = cycle_info['peak_frame']
        trough_frame_idx = cycle_info['trough_frame']

        # 获取两个关键帧
        frame_high = frames_buffer[peak_frame_idx]
        frame_low = frames_buffer[trough_frame_idx]

        # 统一尺寸
        target_h, target_w = frame_high.shape[:2]
        frame_low = cv2.resize(frame_low, (target_w, target_h))

        h, w = target_h, target_w

        # 计算运动时长
        duration = (trough_frame_idx - peak_frame_idx) / fps

        # 为高点和低点创建带标注的图片
        result = {
            'high_point': None,
            'low_point': None,
            'duration': duration,
            'momentum': cycle_info['momentum'],
            'height_change': cycle_info['peak_y'] - cycle_info['trough_y']
        }

        # 处理高点图片 - 只绘制左手
        frame_high_annotated = frame_high.copy()
        frame_rgb = cv2.cvtColor(frame_high, cv2.COLOR_BGR2RGB)
        pose_result = self.pose.process(frame_rgb)

        if pose_result.pose_landmarks:
            # 只绘制左手的关键点和连接
            landmarks = pose_result.pose_landmarks.landmark

            # MediaPipe 左手关键点索引
            LEFT_ARM_INDICES = [
                11,  # 左肩
                13,  # 左肘
                15,  # 左腕
            ]

            # 绘制左手关键点
            for idx in LEFT_ARM_INDICES:
                landmark = landmarks[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame_high_annotated, (x, y), 8, (34, 197, 94), -1)
                cv2.circle(frame_high_annotated, (x, y), 10, (255, 255, 255), 2)

            # 绘制左手连接线
            connections = [(11, 13), (13, 15)]  # 左肩->左肘->左腕
            for start_idx, end_idx in connections:
                start_pt = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                end_pt = (int(landmarks[end_idx].x * w), int(landmarks[end_idx]. y * h))
                cv2.line(frame_high_annotated, start_pt, end_pt, (34, 197, 94), 4)

        # 添加信息面板
        self._add_info_panel_high(frame_high_annotated, peak_frame_idx,
                                  cycle_info['peak_y'], duration)

        # 保存高点图片
        _, buffer = cv2.imencode('.jpg', frame_high_annotated)
        result['high_point'] = base64.b64encode(buffer).decode('utf-8')

        # 处理低点图片 - 只绘制左手
        frame_low_annotated = frame_low.copy()
        frame_rgb = cv2.cvtColor(frame_low, cv2.COLOR_BGR2RGB)
        pose_result = self.pose.process(frame_rgb)

        if pose_result.pose_landmarks:
            # 只绘制左手的关键点和连接
            landmarks = pose_result.pose_landmarks.landmark

            # MediaPipe 左手关键点索引
            LEFT_ARM_INDICES = [11, 13, 15]

            # 绘制左手关键点
            for idx in LEFT_ARM_INDICES:
                landmark = landmarks[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame_low_annotated, (x, y), 8, (239, 68, 68), -1)
                cv2.circle(frame_low_annotated, (x, y), 10, (255, 255, 255), 2)

            # 绘制左手连接线
            connections = [(11, 13), (13, 15)]
            for start_idx, end_idx in connections:
                start_pt = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                end_pt = (int(landmarks[end_idx].x * w), int(landmarks[end_idx]. y * h))
                cv2.line(frame_low_annotated, start_pt, end_pt, (239, 68, 68), 4)

        # 添加信息面板
        self._add_info_panel_low(frame_low_annotated, trough_frame_idx,
                                 cycle_info['trough_y'],
                                 cycle_info['momentum'],
                                 cycle_info['peak_y'] - cycle_info['trough_y'],
                                 duration)

        # 保存低点图片
        _, buffer = cv2.imencode('.jpg', frame_low_annotated)
        result['low_point'] = base64.b64encode(buffer).decode('utf-8')

        return result

    def _add_info_panel_high(self, frame, frame_idx, height, duration):
        """在高点图片上添加信息面板"""
        h, w = frame.shape[:2]

        # 半透明深色背景面板
        overlay = frame.copy()
        panel_height = 160
        cv2.rectangle(overlay, (15, 15), (420, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

        # 标题
        cv2.putText(frame, "HIGH POINT - 左手腕举起", (25, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.1, (34, 197, 94), 2)

        # 信息行
        lines = [
            (f"帧号: {frame_idx}", (255, 255, 255)),
            (f"手腕高度: {height:.1f}%", (255, 255, 255)),
            (f"动作时长: {duration:.2f} 秒", (255, 200, 0)),
        ]

        y_offset = 85
        for line, color in lines:
            cv2.putText(frame, line, (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            y_offset += 28

        return frame

    def _add_info_panel_low(self, frame, frame_idx, height, momentum, height_change, duration):
        """在低点图片上添加信息面板"""
        h, w = frame.shape[:2]

        # 半透明深色背景面板
        overlay = frame.copy()
        panel_height = 200
        cv2.rectangle(overlay, (15, 15), (480, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

        # 标题
        cv2.putText(frame, "LOW POINT - 左手腕下垂", (25, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.1, (239, 68, 68), 2)

        # 分隔线
        cv2.line(frame, (25, 65), (470, 65), (255, 255, 255), 1)

        # 信息行
        lines = [
            (f"帧号: {frame_idx}", (255, 255, 255)),
            (f"手腕高度: {height:.1f}%", (255, 255, 255)),
            (f"落下动量: {momentum:.2f} kg*m/s", (255, 200, 0)),
            (f"高度差: {height_change:.1f}%", (0, 200, 255)),
            (f"动作时长: {duration:.2f} 秒", (255, 255, 255)),
        ]

        y_offset = 90
        for line, color in lines:
            cv2.putText(frame, line, (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            y_offset += 25

        return frame

    def create_annotated_frame(self, frame, pose_data, title, subtitle, momentum_value, height_change, color):
        """
        创建带标注的关键帧
        """
        import base64

        annotated = frame.copy()

        # 绘制骨架
        if pose_data:
            self.mp_drawing.draw_landmarks(
                annotated, pose_data, self.mp_pose.POSE_CONNECTIONS)

        # 标题背景
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (550, 160), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0)

        # 标题
        cv2.putText(annotated, title, (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                   1.2, color, 2)

        # 副标题
        cv2.putText(annotated, subtitle, (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (255, 255, 255), 2)

        # 动量值
        momentum_text = f"Momentum: {momentum_value:.2f} kg*m/s"
        cv2.putText(annotated, momentum_text, (20, 115), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (255, 255, 255), 2)

        # 高度变化
        height_text = f"Height Change: {height_change:.1f}%"
        cv2.putText(annotated, height_text, (20, 145), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (255, 255, 255), 2)

        return annotated

    def extract_key_frames(self, frames_buffer, y_positions, momentums, output_dir, fps, all_cycles=None):
        """
        提取最大向下动量动作的关键帧
        包含：高点和低点两张图片，运动路径，时长和动量

        动量计算公式：
        - p = m × v (动量 = 质量 × 速度)
        - v = (y₂ - y₁) / Δt (速度 = 位置变化 / 时间间隔)
        - m = 1 kg (假设手臂质量)

        参数:
        - all_cycles: 所有检测到的落下动作列表，用于前端表格显示
        """
        import base64

        key_frames_data = {
            'max_down_motion': None,
            'all_down_cycles': all_cycles or [],  # 包含所有动作数据
            'formula_info': {
                'momentum_formula': 'p = m × v',
                'velocity_formula': 'v = (y₂ - y₁) / Δt',
                'mass': 'm = 1 kg (arm mass)',
                'note': '向下动量为正值(+)，表示手臂下落'
            }
        }

        if not frames_buffer or len(y_positions) < 10:
            return key_frames_data

        # 找出向下动量最大的动作周期
        cycle = self.find_max_down_momentum_cycle(y_positions, momentums, fps, frames_buffer)

        if not cycle:
            return key_frames_data

        # 创建带运动路径的两张关键帧图片
        motion_frames = self.create_down_motion_frames(
            frames_buffer, y_positions, cycle, fps
        )

        # 保存图片到文件
        high_filename = "high_point.jpg"
        low_filename = "low_point.jpg"

        cv2.imwrite(os.path.join(output_dir, high_filename),
                   cv2.imdecode(np.frombuffer(base64.b64decode(motion_frames['high_point']), np.uint8), cv2.IMREAD_COLOR))
        cv2.imwrite(os.path.join(output_dir, low_filename),
                   cv2.imdecode(np.frombuffer(base64.b64decode(motion_frames['low_point']), np.uint8), cv2.IMREAD_COLOR))

        key_frames_data['max_down_motion'] = {
            'high_point': {
                'image': motion_frames['high_point'],
                'url': f"/output/frames/{high_filename}",
                'frame': cycle['peak_frame'],
                'height': float(cycle['peak_y'])
            },
            'low_point': {
                'image': motion_frames['low_point'],
                'url': f"/output/frames/{low_filename}",
                'frame': cycle['trough_frame'],
                'height': float(cycle['trough_y'])
            },
            'duration': float(motion_frames['duration']),
            'momentum': float(motion_frames['momentum']),
            'height_change': float(motion_frames['height_change'])
        }

        logger.info(f"最大向下动量动作: 帧 {cycle['peak_frame']} -> {cycle['trough_frame']}")
        logger.info(f"时长: {motion_frames['duration']:.2f}秒, 动量: {motion_frames['momentum']:.2f} kg*m/s")

        return key_frames_data

    def analyze_video(self, video_path, output_video_path=None, output_frames_dir=None):
        """分析视频中的甩手动作"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"开始分析视频: {video_path}")
        logger.info(f"帧率: {fps}, 总帧数: {total_frames}")

        # 重置数据
        self.data = {
            'frames': [],
            'left_wrist_y': [],
            'right_wrist_y': [],
            'left_shoulder_y': [],
            'right_shoulder_y': [],
            'left_velocity': [],
            'right_velocity': [],
            'left_momentum': [],
            'right_momentum': [],
            'swing_events': [],
            'fps': fps,
            'total_frames': total_frames,
            'key_frames': [],  # 存储关键帧的 base64 编码
        }

        frame_idx = 0
        frames_buffer = []  # 保存所有帧用于后续提取关键帧

        # 创建关键帧输出目录
        if output_frames_dir:
            Path(output_frames_dir).mkdir(exist_ok=True, parents=True)

        # 用于输出视频
        writer = None
        if output_video_path:
            # 使用 H.264 编码器，兼容性更好
            # macOS: avc1, Linux: H264, Windows: H264
            try:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')  # macOS
            except:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Linux/Windows
                except:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 备用

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            writer = cv2.VideoWriter(
                output_video_path,
                fourcc,
                fps,
                (frame_width, frame_height)
            )

            # 检查 writer 是否成功创建
            if not writer.isOpened():
                logger.warning("无法创建视频写入器，将不输出标注视频")
                writer = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 保存原始帧用于后续提取关键帧
            frames_buffer.append(frame.copy())

            # 转换为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 进行姿态检测
            results = self.pose.process(frame_rgb)

            left_y = None
            right_y = None
            left_shoulder_y = None
            right_shoulder_y = None

            if results.pose_landmarks:
                # 获取关键点位置
                landmarks = results.pose_landmarks.landmark

                # MediaPipe 的 y 坐标：0 在顶部，1 在底部
                # 转换为正值表示高度（0-100范围）
                left_y = (1 - landmarks[self.LEFT_WRIST].y) * 100
                right_y = (1 - landmarks[self.RIGHT_WRIST].y) * 100
                left_shoulder_y = (1 - landmarks[self.LEFT_SHOULDER].y) * 100
                right_shoulder_y = (1 - landmarks[self.RIGHT_SHOULDER].y) * 100

                # 在帧上绘制关键点
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            self.data['frames'].append(frame_idx)
            self.data['left_wrist_y'].append(left_y)
            self.data['right_wrist_y'].append(right_y)
            self.data['left_shoulder_y'].append(left_shoulder_y)
            self.data['right_shoulder_y'].append(right_shoulder_y)

            # 写入输出视频
            if writer:
                writer.write(frame)

            frame_idx += 1

            # 进度日志
            if frame_idx % 30 == 0:
                logger.info(f"已处理 {frame_idx}/{total_frames} 帧")

        cap.release()
        if writer:
            writer.release()

        # 计算速度和动量
        self.data['left_velocity'] = self.calculate_velocity(
            self.data['left_wrist_y'], fps)
        self.data['right_velocity'] = self.calculate_velocity(
            self.data['right_wrist_y'], fps)

        # 计算动量 (假设手臂质量为 1kg)
        self.data['left_momentum'] = self.calculate_momentum(
            self.data['left_velocity'], mass=1.0)
        self.data['right_momentum'] = self.calculate_momentum(
            self.data['right_velocity'], mass=1.0)

        # 检测甩手周期（保留旧字段兼容性）
        left_cycles = self.detect_swing_cycles(
            self.data['left_wrist_y'], self.data['left_velocity'])
        right_cycles = self.detect_swing_cycles(
            self.data['right_wrist_y'], self.data['right_velocity'])

        self.data['swing_events'] = {
            'left': left_cycles,
            'right': right_cycles
        }

        # ── 新：用统一状态机分别检测左右手的上抬/落下周期 ──────────────────
        left_down, left_up = self.find_all_cycles(
            self.data['left_wrist_y'], fps, self.data['left_shoulder_y'])
        right_down, right_up = self.find_all_cycles(
            self.data['right_wrist_y'], fps, self.data['right_shoulder_y'])

        # 给每个周期打上左/右手标签
        for c in left_down:  c['hand'] = 'left'
        for c in left_up:    c['hand'] = 'left'
        for c in right_down: c['hand'] = 'right'
        for c in right_up:   c['hand'] = 'right'

        self.data['left_down_cycles'] = left_down
        self.data['left_up_cycles'] = left_up
        self.data['right_down_cycles'] = right_down
        self.data['right_up_cycles'] = right_up

        # 计算统计信息（现在包含周期级数据）
        self.data['stats'] = self._calculate_stats()

        # 提取完整甩手周期的关键帧
        if output_frames_dir and frames_buffer:
            logger.info("正在提取关键帧...")

            # 优先使用左手数据，左手检测不足时改用右手
            valid_left = sum(1 for y in self.data['left_wrist_y'] if y is not None) > len(self.data['left_wrist_y']) * 0.5
            if valid_left:
                y_data = self.data['left_wrist_y']
                momentum_data = self.data['left_momentum']
                all_down_cycles = self.data['left_down_cycles']
                all_up_cycles = self.data['left_up_cycles']
            else:
                y_data = self.data['right_wrist_y']
                momentum_data = self.data['right_momentum']
                all_down_cycles = self.data['right_down_cycles']
                all_up_cycles = self.data['right_up_cycles']

            # 生成动作日志文件 (CSV格式)，包含上抬和落下
            log_file_path = os.path.join(output_frames_dir, "down_cycles_log.csv")
            self._save_cycles_log(all_down_cycles, all_up_cycles, log_file_path, fps)
            logger.info(f"已记录 {len(all_down_cycles)} 次落下 / {len(all_up_cycles)} 次上抬 到 CSV 日志文件")

            # 提取关键帧（最大落下动量动作）
            key_frames_data = self.extract_key_frames(
                frames_buffer, y_data, momentum_data, output_frames_dir, fps, all_down_cycles
            )

            # 合并左右手周期，按起始帧排序，方便前端按时间顺序展示
            combined_down = sorted(
                self.data['left_down_cycles'] + self.data['right_down_cycles'],
                key=lambda c: c.get('start_frame', 0)
            )
            combined_up = sorted(
                self.data['left_up_cycles'] + self.data['right_up_cycles'],
                key=lambda c: c.get('start_frame', 0)
            )
            # key_frames 里的 all_down_cycles 也同步为合并后数据
            key_frames_data['all_down_cycles'] = combined_down
            self.data['key_frames'] = key_frames_data
            self.data['all_down_cycles'] = combined_down
            self.data['all_up_cycles'] = combined_up
            logger.info("关键帧提取完成")

        logger.info(f"分析完成！共处理 {frame_idx} 帧")
        return self.data

    def _calculate_stats(self):
        """计算统计信息，包含逐周期的上抬/落下动量数据"""
        stats = {
            'left': {},
            'right': {}
        }

        for side, prefix in [('left', 'left'), ('right', 'right')]:
            y_data = [y for y in self.data[f'{prefix}_wrist_y'] if y is not None]
            momentum_data = self.data[f'{prefix}_momentum']

            if y_data:
                stats[side]['max_height'] = float(max(y_data))
                stats[side]['min_height'] = float(min(y_data))
                stats[side]['avg_height'] = float(sum(y_data) / len(y_data))
                stats[side]['range'] = float(max(y_data) - min(y_data))

            if momentum_data:
                stats[side]['max_momentum_down'] = float(max(momentum_data))
                stats[side]['max_momentum_up'] = float(min(momentum_data))
                stats[side]['avg_momentum'] = float(sum(momentum_data) / len(momentum_data))

            # 逐周期统计：落下
            down_cycles = self.data.get(f'{side}_down_cycles', [])
            stats[side]['down_cycle_count'] = len(down_cycles)
            if down_cycles:
                stats[side]['avg_down_momentum'] = float(
                    sum(c['avg_momentum'] for c in down_cycles) / len(down_cycles))
                stats[side]['max_down_cycle_momentum'] = float(
                    max(c['max_momentum'] for c in down_cycles))
            else:
                stats[side]['avg_down_momentum'] = 0.0
                stats[side]['max_down_cycle_momentum'] = 0.0

            # 逐周期统计：上抬
            up_cycles = self.data.get(f'{side}_up_cycles', [])
            stats[side]['up_cycle_count'] = len(up_cycles)
            if up_cycles:
                stats[side]['avg_up_momentum'] = float(
                    sum(c['avg_momentum'] for c in up_cycles) / len(up_cycles))
                stats[side]['max_up_cycle_momentum'] = float(
                    max(c['max_momentum'] for c in up_cycles))
            else:
                stats[side]['avg_up_momentum'] = 0.0
                stats[side]['max_up_cycle_momentum'] = 0.0

        return stats

    def _save_down_cycles_log(self, cycles, log_file_path, fps):
        """兼容旧调用：仅传入落下周期时的封装"""
        self._save_cycles_log(cycles, [], log_file_path, fps)

    def _save_cycles_log(self, down_cycles, up_cycles, log_file_path, fps):
        """
        保存上抬和落下动作记录到 CSV 日志文件。
        两类动作按时间顺序合并，并输出分类统计摘要。
        """
        import csv

        # 合并并按起始帧排序
        all_records = []
        for c in down_cycles:
            all_records.append({**c, '_label': '落下↓'})
        for c in up_cycles:
            all_records.append({**c, '_label': '上抬↑'})
        all_records.sort(key=lambda x: x.get('start_frame', 0))

        with open(log_file_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)

            # 表头
            writer.writerow([
                '序号', '动作类型',
                '起始帧', '高点帧', '低点帧',
                '幅度(%)', '平均动量(kg·m/s)', '峰值动量(kg·m/s)', '时长(秒)'
            ])

            for idx, c in enumerate(all_records, 1):
                writer.writerow([
                    idx,
                    c['_label'],
                    c.get('start_frame', ''),
                    c.get('peak_frame', ''),
                    c.get('trough_frame', ''),
                    f"{c['height_change']:.2f}",
                    f"{c['avg_momentum']:.2f}",
                    f"{c['max_momentum']:.2f}",
                    f"{c['duration']:.2f}"
                ])

            # 统计摘要
            writer.writerow([])
            writer.writerow(['=== 统计摘要 ==='])
            writer.writerow(['落下动作次数', len(down_cycles)])
            writer.writerow(['上抬动作次数', len(up_cycles)])

            if down_cycles:
                writer.writerow(['落下平均幅度(%)',
                                  f"{sum(c['height_change'] for c in down_cycles) / len(down_cycles):.2f}"])
                writer.writerow(['落下平均动量(kg·m/s)',
                                  f"{sum(c['avg_momentum'] for c in down_cycles) / len(down_cycles):.2f}"])
                writer.writerow(['落下峰值动量(kg·m/s)',
                                  f"{max(c['max_momentum'] for c in down_cycles):.2f}"])

            if up_cycles:
                writer.writerow(['上抬平均幅度(%)',
                                  f"{sum(c['height_change'] for c in up_cycles) / len(up_cycles):.2f}"])
                writer.writerow(['上抬平均动量(kg·m/s)',
                                  f"{sum(c['avg_momentum'] for c in up_cycles) / len(up_cycles):.2f}"])
                writer.writerow(['上抬峰值动量(kg·m/s)',
                                  f"{max(c['max_momentum'] for c in up_cycles):.2f}"])

            writer.writerow([])
            writer.writerow(['=== 说明 ==='])
            writer.writerow(['动量计算: p = m × v，速度由高斯平滑位置的中心差分得出'])
            writer.writerow(['质量假设: m = 1 kg'])
            writer.writerow(['落下动量: 取落下阶段负速度的绝对值均值/峰值'])
            writer.writerow(['上抬动量: 取上抬阶段正速度的均值/峰值'])


# 创建全局分析器实例
analyzer = SwingAnalyzer()


@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    """上传并分析视频"""
    if 'video' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    if file and allowed_file(file.filename):
        # 统一文件名为小写，避免大小写问题
        filename = secure_filename(file.filename).lower()
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        logger.info(f"文件已保存: {video_path}")

        # 创建关键帧输出目录
        frames_id = filename.rsplit('.', 1)[0]
        frames_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'frames', frames_id)
        Path(frames_dir).mkdir(exist_ok=True, parents=True)

        try:
            # 分析视频并提取关键帧
            result = analyzer.analyze_video(video_path, None, frames_dir)

            # 保存分析结果为 JSON
            json_filename = f"analysis_{frames_id}.json"
            json_path = os.path.join(app.config['OUTPUT_FOLDER'], json_filename)

            # 清理数据以减少传输大小
            clean_result = {
                'stats': result.get('stats', {}),
                'fps': result.get('fps', 30),
                'total_frames': result.get('total_frames', 0),
                'left_wrist_y': result.get('left_wrist_y', [])[:1000],
                'right_wrist_y': result.get('right_wrist_y', [])[:1000],
                'left_momentum': result.get('left_momentum', [])[:1000],
                'right_momentum': result.get('right_momentum', [])[:1000],
                'frames': result.get('frames', [])[:1000],
                'key_frames': result.get('key_frames', {}),
                'all_down_cycles': result.get('all_down_cycles', []),
                'all_up_cycles': result.get('all_up_cycles', [])
            }

            # 读取日志文件内容
            log_content = ""
            log_file_path = os.path.join(frames_dir, "down_cycles_log.csv")
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r', encoding='utf-8-sig') as f:
                    log_content = f.read()

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(clean_result, f, ensure_ascii=False, indent=2)

            n_down = len(result.get('all_down_cycles', []))
            n_up = len(result.get('all_up_cycles', []))
            return jsonify({
                'success': True,
                'message': f'分析完成，检测到 {n_down} 次落下、{n_up} 次上抬动作',
                'data': clean_result,
                'log_content': log_content,  # 添加日志内容
                'original_video': f"/uploads/{filename}",
                'analysis_json': f"/output/{json_filename}",
                'frames_base_url': f"/output/frames/{frames_id}"
            })

        except Exception as e:
            logger.error(f"分析失败: {str(e)}")
            return jsonify({'error': f'分析失败: {str(e)}'}), 500

    return jsonify({'error': '不支持的文件格式'}), 400


@app.route('/output/<path:filepath>')
def get_output(filepath):
    """获取输出文件（支持子路径）"""
    # 处理 frames/ 子目录
    if '/' in filepath:
        return send_from_directory(app.config['OUTPUT_FOLDER'], filepath)
    return send_from_directory(app.config['OUTPUT_FOLDER'], filepath)


@app.route('/uploads/<filename>')
def get_upload(filename):
    """获取上传的视频文件"""
    response = send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    # 设置正确的 MIME 类型，确保浏览器能播放视频
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    mime_types = {
        'mp4': 'video/mp4',
        'avi': 'video/x-msvideo',
        'mov': 'video/quicktime',
        'mkv': 'video/x-matroska',
        'webm': 'video/webm'
    }
    if ext in mime_types:
        response.headers['Content-Type'] = mime_types[ext]

    # 添加支持跨域和范围请求（大文件播放必需）
    response.headers['Accept-Ranges'] = 'bytes'

    return response


@app.route('/health')
def health():
    """健康检查"""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8888)
