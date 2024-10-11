import cv2
import mediapipe as mp
import math
import numpy as np

# 初始化 MediaPipe Pose 模型
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 计算两个向量之间的夹角
def calculate_angle(a, b, c):
    ba = [a[0] - b[0], a[1] - b[1]]  # 向量 BA
    bc = [c[0] - b[0], c[1] - b[1]]  # 向量 BC
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    magnitude_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    cos_angle = dot_product / (magnitude_ba * magnitude_bc)
    angle = math.degrees(math.acos(max(-1, min(1, cos_angle))))
    return angle

# 检查右手肘是否高于肩膀
def is_right_elbow_above_shoulder(landmarks):
    return landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y < landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y

# 检查右手肘角度
def check_right_elbow_angle(angle):
    return 160 <= angle <= 170

# 检查左膝盖角度
def check_left_knee_angle(angle):
    return angle >= 170  # 假设大于170度视为"接近伸直"

# 检查脚尖朝向
def check_foot_direction(landmarks):
    left_foot = np.array([landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y])
    right_foot = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y])
    foot_direction = left_foot - right_foot
    # 假设y轴正方向为捕手方向，检查脚尖是否大致朝向y轴正方向
    return foot_direction[1] > 0

# 打开视频文件
cap = cv2.VideoCapture("IMG_7239.MOV")  # 请替换为您的视频文件路径

correct_elbow_angle_detected = False
correct_knee_angle_detected = False
correct_foot_direction_detected = False
detection_phase = "Not started"  # 用于追踪检测阶段

cv2.namedWindow('Pitcher Pose Analysis', cv2.WINDOW_NORMAL)
# 使用 MediaPipe Pose 进行骨架检测
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 将图像从 BGR 转换为 RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # 计算右手肘角度
            right_elbow_angle = calculate_angle(
                [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            )

            # 计算左膝盖角度
            left_knee_angle = calculate_angle(
                [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            )

            # 检测逻辑
            if detection_phase == "Not started" and is_right_elbow_above_shoulder(landmarks):
                detection_phase = "Started"
                print("检测开始：右手肘高于肩膀")
            
            elif detection_phase == "Started":
                if check_right_elbow_angle(right_elbow_angle):
                    correct_elbow_angle_detected = True
                
                if check_left_knee_angle(left_knee_angle):
                    correct_knee_angle_detected = True
                
                if check_foot_direction(landmarks):
                    correct_foot_direction_detected = True
                
                if is_right_elbow_above_shoulder(landmarks) and right_elbow_angle > 170:
                    detection_phase = "Ended"
                    print("检测结束：右手肘高于肩膀且角度大于170度")

            # 绘制骨架
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

        # 显示处理后的图像
        cv2.imshow('Pitcher Pose Analysis', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# 输出结果
print("\n投球姿势分析结果：")
if correct_elbow_angle_detected:
    print("- 在检测过程中，右手肘角度有达到正确范围（160-170度）。")
else:
    print("- 在整个检测过程中，右手肘角度未达到正确范围（160-170度）。")

if correct_knee_angle_detected:
    print("- 投球动作结束时，左膝盖接近伸直。")
else:
    print("- 投球动作结束时，左膝盖未充分伸直。")

if correct_foot_direction_detected:
    print("- 脚尖正确朝向捕手方向。")
else:
    print("- 脚尖未正确朝向捕手方向。")