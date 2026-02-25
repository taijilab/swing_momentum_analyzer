// 甩手动量分析器 - 前端脚本

let currentAnalysisData = null;
let heightChart = null;
let momentumChart = null;
let currentVideoFps = 30;  // 保存当前视频的帧率
let videoPlaybackTimer = null;  // 用于控制播放片段的定时器

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    initUploadArea();
    initVideoInput();
});

// 初始化上传区域
function initUploadArea() {
    const uploadArea = document.getElementById('uploadArea');

    // 拖拽事件
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleVideoUpload(files[0]);
        }
    });

    // 点击上传
    uploadArea.addEventListener('click', (e) => {
        if (e.target.tagName !== 'BUTTON') {
            document.getElementById('videoInput').click();
        }
    });
}

// 初始化视频输入
function initVideoInput() {
    const videoInput = document.getElementById('videoInput');
    videoInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleVideoUpload(e.target.files[0]);
        }
    });
}

// 处理视频上传
function handleVideoUpload(file) {
    // 验证文件类型
    const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm'];
    const fileExt = file.name.split('.').pop().toLowerCase();

    if (!validTypes.includes(file.type) && !['mp4', 'avi', 'mov', 'mkv', 'webm'].includes(fileExt)) {
        showError('请上传有效的视频文件 (MP4, AVI, MOV, MKV, WEBM)');
        return;
    }

    // 验证文件大小 (100MB)
    if (file.size > 100 * 1024 * 1024) {
        showError('文件大小不能超过 100MB');
        return;
    }

    // 显示进度
    showProgress();

    // 上传文件
    uploadAndAnalyze(file);
}

// 显示进度
function showProgress() {
    document.getElementById('uploadSection').style.display = 'none';
    document.getElementById('progressSection').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';

    // 模拟进度动画
    let progress = 0;
    const progressFill = document.getElementById('progressFill');
    const progressInterval = setInterval(() => {
        progress += Math.random() * 5;
        if (progress > 90) progress = 90;
        progressFill.style.width = progress + '%';
    }, 200);

    // 保存 interval ID 以便后续清除
    window.progressInterval = progressInterval;
}

// 上传并分析
async function uploadAndAnalyze(file) {
    const formData = new FormData();
    formData.append('video', file);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            // 完成进度
            clearInterval(window.progressInterval);
            document.getElementById('progressFill').style.width = '100%';

            setTimeout(() => {
                showResults(result);
            }, 500);
        } else {
            throw new Error(result.error || '分析失败');
        }
    } catch (error) {
        clearInterval(window.progressInterval);
        showError('上传或分析失败: ' + error.message);
    }
}

// 显示结果
function showResults(result) {
    currentAnalysisData = result.data;
    currentVideoFps = result.data.fps || 30;

    // 隐藏进度，显示结果
    document.getElementById('uploadSection').style.display = 'none';
    document.getElementById('progressSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'block';

    // 显示日志内容
    displayLogContent(result.log_content || '');

    // 更新统计数据
    updateStats(result.data.stats);

    // 绘制图表
    drawCharts(result.data);

    // 显示关键帧图片
    displayKeyFrames(result.data.key_frames || {});

    // 设置输出视频（使用原始视频）
    const video = document.getElementById('outputVideo');
    video.src = result.original_video;
}

// 显示关键帧图片
function displayKeyFrames(keyFrames) {
    const keyframesSection = document.getElementById('keyframesSection');
    const motionContainer = document.getElementById('motionContainer');
    const cyclesTableCard = document.getElementById('cyclesTableCard');

    const hasMotion = keyFrames.max_down_motion &&
                      keyFrames.max_down_motion.high_point &&
                      keyFrames.max_down_motion.low_point;

    if (!hasMotion) {
        keyframesSection.style.display = 'none';
        return;
    }

    keyframesSection.style.display = 'block';

    const motion = keyFrames.max_down_motion;

    // ── 合并上抬和落下周期，按起始帧排序 ────────────────────────────────
    const downCycles = (keyFrames.all_down_cycles || []).map(c => ({ ...c, _type: 'down' }));
    const upCycles   = (currentAnalysisData.all_up_cycles || []).map(c => ({ ...c, _type: 'up' }));
    const allCycles  = [...downCycles, ...upCycles].sort(
        (a, b) => (a.start_frame || 0) - (b.start_frame || 0)
    );

    if (allCycles.length > 0) {
        cyclesTableCard.style.display = 'block';
        document.getElementById('totalCycles').textContent = allCycles.length;

        const tbody = document.getElementById('cyclesTableBody');
        tbody.innerHTML = '';

        allCycles.forEach((cycle, idx) => {
            const row = document.createElement('tr');

            const isDown = cycle._type === 'down';
            // 标记最大落下动量动作
            if (isDown &&
                cycle.peak_frame === motion.high_point.frame &&
                cycle.trough_frame === motion.low_point.frame) {
                row.classList.add('highlight-cycle');
            }

            // 颜色样式
            row.classList.add(isDown ? 'cycle-down' : 'cycle-up');
            row.style.cursor = 'pointer';

            const endFrame = isDown ? cycle.trough_frame : cycle.peak_frame;
            row.onclick = () => seekVideoToFrame(cycle.start_frame, endFrame);
            row.title = `点击播放：帧 ${cycle.start_frame} → ${endFrame}`;

            const typeLabel = isDown ? '↓ 落下' : '↑ 上抬';
            const handLabel = cycle.hand === 'left' ? '👈 左手' : cycle.hand === 'right' ? '👉 右手' : '-';
            const handClass = cycle.hand === 'left' ? 'badge-left' : cycle.hand === 'right' ? 'badge-right' : '';

            row.innerHTML = `
                <td>${idx + 1}</td>
                <td><span class="type-badge ${handClass}">${handLabel}</span></td>
                <td><span class="type-badge ${isDown ? 'badge-down' : 'badge-up'}">${typeLabel}</span></td>
                <td>${cycle.start_frame}</td>
                <td>${cycle.peak_frame !== undefined ? cycle.peak_frame : '-'}</td>
                <td>${cycle.trough_frame !== undefined ? cycle.trough_frame : '-'}</td>
                <td>${cycle.height_change.toFixed(1)}%</td>
                <td>${cycle.avg_momentum.toFixed(2)} kg·m/s</td>
                <td>${cycle.max_momentum.toFixed(2)} kg·m/s</td>
                <td>${cycle.duration.toFixed(2)} s</td>
            `;
            tbody.appendChild(row);
        });
    } else {
        cyclesTableCard.style.display = 'none';
    }

    // ── 显示最大落下动量关键帧图片 ───────────────────────────────────────
    motionContainer.style.display = 'block';

    const highImg = document.getElementById('highPointImage');
    highImg.src = `data:image/jpeg;base64,${motion.high_point.image}`;
    highImg.onclick = () => showImageModal(motion.high_point.image, '高点 - 手臂举起');
    document.getElementById('highPointHeight').textContent = motion.high_point.height.toFixed(1) + '%';
    document.getElementById('highPointFrame').textContent = motion.high_point.frame;

    const lowImg = document.getElementById('lowPointImage');
    lowImg.src = `data:image/jpeg;base64,${motion.low_point.image}`;
    lowImg.onclick = () => showImageModal(motion.low_point.image, '低点 - 手臂下垂');
    document.getElementById('lowPointHeight').textContent = motion.low_point.height.toFixed(1) + '%';
    document.getElementById('lowPointFrame').textContent = motion.low_point.frame;

    document.getElementById('connectorDuration').textContent = motion.duration.toFixed(1) + 's';
    document.getElementById('motionDuration').textContent = motion.duration.toFixed(2) + ' 秒';
    document.getElementById('motionMomentum').textContent = motion.momentum.toFixed(2) + ' kg·m/s';
    document.getElementById('motionHeightChange').textContent = motion.height_change.toFixed(1) + ' %';
}

// 显示图片模态框
function showImageModal(base64Image, title) {
    // 移除旧的模态框
    const existingModal = document.getElementById('imageModal');
    if (existingModal) {
        existingModal.remove();
    }

    const modal = document.createElement('div');
    modal.id = 'imageModal';
    modal.className = 'image-modal';
    modal.innerHTML = `
        <div class="modal-content">
            <span class="modal-close">&times;</span>
            <img src="data:image/jpeg;base64,${base64Image}" alt="${title}">
            <div class="modal-caption">${title}</div>
        </div>
    `;

    document.body.appendChild(modal);

    // 关闭事件
    modal.querySelector('.modal-close').addEventListener('click', () => modal.remove());
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.remove();
    });
}

// 更新统计数据
function updateStats(stats) {
    const fmt = (num) => (num !== undefined && num !== null) ? num.toFixed(2) : '-';

    // 左手：高度
    document.getElementById('leftMaxHeight').textContent = fmt(stats.left.max_height) + ' %';
    document.getElementById('leftMinHeight').textContent = fmt(stats.left.min_height) + ' %';
    document.getElementById('leftRange').textContent = fmt(stats.left.range) + ' %';

    // 左手：落下周期统计
    const ldCount = stats.left.down_cycle_count || 0;
    const ldMax = fmt(stats.left.max_down_cycle_momentum);
    document.getElementById('leftDownStat').textContent = `${ldCount} 次 / ${ldMax} kg·m/s`;
    document.getElementById('leftAvgDownMomentum').textContent = fmt(stats.left.avg_down_momentum) + ' kg·m/s';

    // 左手：上抬周期统计
    const luCount = stats.left.up_cycle_count || 0;
    const luMax = fmt(stats.left.max_up_cycle_momentum);
    document.getElementById('leftUpStat').textContent = `${luCount} 次 / ${luMax} kg·m/s`;
    document.getElementById('leftAvgUpMomentum').textContent = fmt(stats.left.avg_up_momentum) + ' kg·m/s';

    // 右手：高度
    document.getElementById('rightMaxHeight').textContent = fmt(stats.right.max_height) + ' %';
    document.getElementById('rightMinHeight').textContent = fmt(stats.right.min_height) + ' %';
    document.getElementById('rightRange').textContent = fmt(stats.right.range) + ' %';

    // 右手：落下周期统计
    const rdCount = stats.right.down_cycle_count || 0;
    const rdMax = fmt(stats.right.max_down_cycle_momentum);
    document.getElementById('rightDownStat').textContent = `${rdCount} 次 / ${rdMax} kg·m/s`;
    document.getElementById('rightAvgDownMomentum').textContent = fmt(stats.right.avg_down_momentum) + ' kg·m/s';

    // 右手：上抬周期统计
    const ruCount = stats.right.up_cycle_count || 0;
    const ruMax = fmt(stats.right.max_up_cycle_momentum);
    document.getElementById('rightUpStat').textContent = `${ruCount} 次 / ${ruMax} kg·m/s`;
    document.getElementById('rightAvgUpMomentum').textContent = fmt(stats.right.avg_up_momentum) + ' kg·m/s';
}

// 绘制图表
function drawCharts(data) {
    // 销毁现有图表
    if (heightChart) heightChart.destroy();
    if (momentumChart) momentumChart.destroy();

    // 采样数据（避免点太多）
    const sampleRate = Math.max(1, Math.floor(data.frames.length / 200));

    const frames = data.frames.filter((_, i) => i % sampleRate === 0);
    const leftHeight = data.left_wrist_y.filter((_, i) => i % sampleRate === 0);
    const rightHeight = data.right_wrist_y.filter((_, i) => i % sampleRate === 0);
    const leftMomentum = data.left_momentum.filter((_, i) => i % sampleRate === 0);
    const rightMomentum = data.right_momentum.filter((_, i) => i % sampleRate === 0);

    // 时间轴（秒）
    const timeLabels = frames.map(f => (f / (data.fps || 30)).toFixed(1));

    // 高度图表
    const heightCtx = document.getElementById('heightChart').getContext('2d');
    heightChart = new Chart(heightCtx, {
        type: 'line',
        data: {
            labels: timeLabels,
            datasets: [
                {
                    label: '左手高度',
                    data: leftHeight,
                    borderColor: '#4f46e5',
                    backgroundColor: 'rgba(79, 70, 229, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0
                },
                {
                    label: '右手高度',
                    data: rightHeight,
                    borderColor: '#22c55e',
                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: '时间 (秒)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: '高度 (%)'
                    },
                    min: 0,
                    max: 100
                }
            },
            plugins: {
                legend: {
                    position: 'top'
                }
            }
        }
    });

    // 动量图表
    const momentumCtx = document.getElementById('momentumChart').getContext('2d');
    momentumChart = new Chart(momentumCtx, {
        type: 'line',
        data: {
            labels: timeLabels,
            datasets: [
                {
                    label: '左手动量',
                    data: leftMomentum,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0
                },
                {
                    label: '右手动量',
                    data: rightMomentum,
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: '时间 (秒)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: '动量 (kg·m/s)'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top'
                },
                annotation: {
                    annotations: {
                        line1: {
                            type: 'line',
                            yMin: 0,
                            yMax: 0,
                            borderColor: '#94a3b8',
                            borderWidth: 1
                        }
                    }
                }
            }
        }
    });
}

// 重置上传
function resetUpload() {
    document.getElementById('uploadSection').style.display = 'block';
    document.getElementById('progressSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('videoInput').value = '';
    currentAnalysisData = null;
}

// 下载分析数据
function downloadAnalysis() {
    if (!currentAnalysisData) return;

    const dataStr = JSON.stringify(currentAnalysisData, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'swing_analysis.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// 显示错误
function showError(message) {
    alert(message);
    resetUpload();
}

// 跳转视频到指定帧并播放片段
function seekVideoToFrame(startFrame, endFrame) {
    const video = document.getElementById('outputVideo');
    if (!video || currentVideoFps === 0) return;

    // 清除之前的定时器和事件监听
    if (videoPlaybackTimer) {
        clearTimeout(videoPlaybackTimer);
        videoPlaybackTimer = null;
    }

    // 移除之前的片段播放监听器
    video._clipEndHandler = null;

    // 计算时间（秒）
    const startTimeInSeconds = startFrame / currentVideoFps;
    const endTimeInSeconds = endFrame ? endFrame / currentVideoFps : null;

    // 设置视频时间
    video.currentTime = startTimeInSeconds;

    // 如果有结束帧，设置自动停止
    if (endTimeInSeconds && endTimeInSeconds > startTimeInSeconds) {
        // 创建新的结束监听器
        video._clipEndHandler = () => {
            if (video.currentTime >= endTimeInSeconds) {
                video.pause();
                video.removeEventListener('timeupdate', video._clipEndHandler);
                video._clipEndHandler = null;
                console.log(`播放完成: 帧 ${startFrame} -> ${endFrame}`);
            }
        };
        video.addEventListener('timeupdate', video._clipEndHandler);
    }

    // 开始播放
    video.play();

    // 添加视觉反馈
    const endInfo = endFrame ? ` -> 帧 ${endFrame}` : '';
    console.log(`视频播放片段: 帧 ${startFrame}${endInfo} (${startTimeInSeconds.toFixed(2)}秒)`);
}

// 显示日志内容
function displayLogContent(logContent) {
    const logContentEl = document.getElementById('logContent');
    if (!logContentEl) return;

    if (logContent) {
        // 将 CSV 内容格式化为更好的显示效果
        const lines = logContent.split('\n');
        let formattedContent = '';

        lines.forEach(line => {
            if (line.includes('===')) {
                // 统计摘要标题
                formattedContent += `\n\x1b[33m${line}\x1b[0m\n`;
            } else if (line.startsWith('序号') || line.includes('---')) {
                // 表头或分隔线
                formattedContent += `\x1b[36m${line}\x1b[0m\n`;
            } else if (line.trim()) {
                // 数据行
                formattedContent += `${line}\n`;
            }
        });

        logContentEl.textContent = logContent;  // 直接显示原始内容
    } else {
        logContentEl.textContent = '暂无日志数据';
    }
}
