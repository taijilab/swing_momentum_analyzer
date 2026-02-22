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

    def find_all_down_cycles(self, y_positions, momentums, fps, frames_buffer=None, shoulder_y=None):
        """
        找出所有左手手腕的落下动作
        条件：手腕必须超过肩膀高度才算一次成功的甩手

        参数:
        - y_positions: 手腕高度数组
        - momentums: 动量数组
        - fps: 帧率
        - frames_buffer: 帧缓冲
        - shoulder_y: 肩膀高度数组

        返回每次落下动作的记录列表
        """
        if len(y_positions) < 20:
            return []

        cycles = []

        # 使用速度来判断动作方向
        velocity = self.calculate_velocity(y_positions, fps)

        # 平滑速度数据
        window = 5
        smoothed_velocity = []
        for i in range(len(velocity)):
            start = max(0, i - window // 2)
            end = min(len(velocity), i + window // 2 + 1)
            smoothed_velocity.append(sum(velocity[start:end]) / (end - start))

        # 状态机检测落下动作
        state = 'unknown'

        # 记录当前动作的关键点
        action_start_frame = None
        peak_frame = None
        peak_y = None
        peak_above_shoulder = False  # 高点是否超过肩膀
        trough_frame = None
        trough_y = None

        # 设定阈值
        HIGH_THRESHOLD = 55
        LOW_THRESHOLD = 48

        for i in range(len(y_positions)):
            if y_positions[i] is None:
                continue

            current_y = y_positions[i]
            current_vel = smoothed_velocity[i] if i < len(smoothed_velocity) else 0

            # 获取当前帧的肩膀高度
            current_shoulder_y = shoulder_y[i] if shoulder_y and i < len(shoulder_y) else None

            if state == 'unknown':
                if current_y > HIGH_THRESHOLD:
                    state = 'at_peak'
                    peak_frame = i
                    peak_y = current_y
                    action_start_frame = i
                    if current_shoulder_y:
                        peak_above_shoulder = current_y > current_shoulder_y

            elif state == 'at_peak':
                if current_y > peak_y:
                    peak_frame = i
                    peak_y = current_y
                    action_start_frame = i
                    if current_shoulder_y:
                        peak_above_shoulder = current_y > current_shoulder_y

                if current_vel > 12:
                    state = 'falling'

            elif state == 'falling':
                if trough_frame is None or current_y < trough_y:
                    trough_frame = i
                    trough_y = current_y

                if current_vel < -12 or current_y < LOW_THRESHOLD:
                    if peak_frame is not None and trough_frame is not None:
                        height_change = peak_y - trough_y if peak_y and trough_y else 0

                        # 只有手腕超过肩膀的高度变化才计入
                        if peak_above_shoulder and height_change >= 8:
                            fall_momentums = [m for m in momentums[peak_frame:trough_frame+1] if m > 0]
                            max_momentum = max(fall_momentums) if fall_momentums else 0
                            duration = (trough_frame - peak_frame) / fps if trough_frame > peak_frame else 0

                            cycle_info = {
                                'cycle_number': len(cycles) + 1,
                                'start_frame': action_start_frame,
                                'peak_frame': peak_frame,
                                'trough_frame': trough_frame,
                                'peak_y': peak_y,
                                'trough_y': trough_y,
                                'height_change': height_change,
                                'max_momentum': max_momentum,
                                'duration': duration,
                                'above_shoulder': peak_above_shoulder
                            }
                            cycles.append(cycle_info)

                            logger.info(f"[左手落下 #{len(cycles)}] "
                                      f"开始帧{action_start_frame}, 高点帧{peak_frame}(高度{peak_y:.1f}%, 超肩:{'是' if peak_above_shoulder else '否'}) -> "
                                      f"低点帧{trough_frame}(高度{trough_y:.1f}%), "
                                      f"高度差={height_change:.1f}%, 动量={max_momentum:.2f}, 时长={duration:.2f}s")

                    state = 'at_trough'

            elif state == 'at_trough':
                if current_vel < -12:
                    state = 'raising'
                if current_y > HIGH_THRESHOLD:
                    state = 'at_peak'
                    peak_frame = i
                    peak_y = current_y
                    action_start_frame = i
                    peak_above_shoulder = False
                    if current_shoulder_y:
                        peak_above_shoulder = current_y > current_shoulder_y
                    trough_frame = None
                    trough_y = None

            elif state == 'raising':
                if current_y > HIGH_THRESHOLD and abs(current_vel) < 8:
                    state = 'at_peak'
                    peak_frame = i
                    peak_y = current_y
                    action_start_frame = i
                    peak_above_shoulder = False
                    if current_shoulder_y:
                        peak_above_shoulder = current_y > current_shoulder_y
                    trough_frame = None
                    trough_y = None

        return cycles

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

        # 检测甩手周期
        left_cycles = self.detect_swing_cycles(
            self.data['left_wrist_y'], self.data['left_velocity'])
        right_cycles = self.detect_swing_cycles(
            self.data['right_wrist_y'], self.data['right_velocity'])

        self.data['swing_events'] = {
            'left': left_cycles,
            'right': right_cycles
        }

        # 计算统计信息
        self.data['stats'] = self._calculate_stats()

        # 提取完整甩手周期的关键帧
        if output_frames_dir and frames_buffer:
            logger.info("正在提取关键帧...")

            # 使用左手数据提取关键帧（如果左手数据不可用则用右手）
            y_data = self.data['left_wrist_y']
            momentum_data = self.data['left_momentum']

            # 获取肩膀数据
            shoulder_data = self.data['left_shoulder_y']

            # 检查数据有效性，左手数据不足则用右手
            valid_left = sum(1 for y in y_data if y is not None) > len(y_data) * 0.5
            if not valid_left:
                y_data = self.data['right_wrist_y']
                momentum_data = self.data['right_momentum']
                shoulder_data = self.data['right_shoulder_y']

            # 记录所有落下动作（只关注左手，需要超过肩膀）
            all_cycles = self.find_all_down_cycles(y_data, momentum_data, fps, frames_buffer, shoulder_data)

            # 生成落下动作日志文件 (CSV格式)
            if all_cycles:
                log_file_path = os.path.join(output_frames_dir, "down_cycles_log.csv")
                self._save_down_cycles_log(all_cycles, log_file_path, fps)
                logger.info(f"已记录 {len(all_cycles)} 次左手落下动作到 CSV 日志文件")

            # 提取关键帧（最大动量动作），传入所有动作数据
            key_frames_data = self.extract_key_frames(
                frames_buffer, y_data, momentum_data, output_frames_dir, fps, all_cycles
            )

            self.data['key_frames'] = key_frames_data
            self.data['all_down_cycles'] = all_cycles  # 保留用于直接访问
            logger.info("关键帧提取完成")

        logger.info(f"分析完成！共处理 {frame_idx} 帧")
        return self.data

    def _calculate_stats(self):
        """计算统计信息"""
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

        return stats

    def _save_down_cycles_log(self, cycles, log_file_path, fps):
        """保存落下动作记录到 CSV 格式日志文件"""
        import csv

        with open(log_file_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)

            # 写入表头
            writer.writerow([
                '序号',
                '开始帧',
                '高点帧',
                '低点帧',
                '高度差(%)',
                '落下动量(kg·m/s)',
                '动作时长(秒)'
            ])

            # 写入每次动作记录
            for cycle in cycles:
                writer.writerow([
                    cycle['cycle_number'],
                    cycle['start_frame'],
                    cycle['peak_frame'],
                    cycle['trough_frame'],
                    f"{cycle['height_change']:.2f}",
                    f"{cycle['max_momentum']:.2f}",
                    f"{cycle['duration']:.2f}"
                ])

            # 添加统计摘要
            if cycles:
                writer.writerow([])
                writer.writerow(['=== 统计摘要 ==='])
                writer.writerow(['总动作次数', len(cycles)])
                writer.writerow(['平均高度差(%)', f"{sum(c['height_change'] for c in cycles) / len(cycles):.2f}"])
                writer.writerow(['平均动量(kg·m/s)', f"{sum(c['max_momentum'] for c in cycles) / len(cycles):.2f}"])
                writer.writerow(['平均时长(秒)', f"{sum(c['duration'] for c in cycles) / len(cycles):.2f}"])
                writer.writerow(['最大高度差(%)', f"{max(c['height_change'] for c in cycles):.2f}"])
                writer.writerow(['最大动量(kg·m/s)', f"{max(c['max_momentum'] for c in cycles):.2f}"])
                writer.writerow([])
                writer.writerow(['=== 说明 ==='])
                writer.writerow(['动量计算: p = m × v (动量 = 质量 × 速度)'])
                writer.writerow(['质量假设: m = 1 kg'])
                writer.writerow(['检测对象: 左手手腕的落下动作'])


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
                'all_down_cycles': result.get('all_down_cycles', [])  # 所有落下动作记录
            }

            # 读取日志文件内容
            log_content = ""
            log_file_path = os.path.join(frames_dir, "down_cycles_log.csv")
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r', encoding='utf-8-sig') as f:
                    log_content = f.read()

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(clean_result, f, ensure_ascii=False, indent=2)

            return jsonify({
                'success': True,
                'message': f'分析完成，检测到 {len(result.get("all_down_cycles", []))} 次落下动作',
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
