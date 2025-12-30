import cv2
import numpy as np
from PIL import Image
from math import ceil

# def correct_pitch_distortion(observed_angle_deg, camera_pitch_deg=30):
#     """
#     카메라가 위에서 내려다볼 때(Pitch) 발생하는 각도 왜곡을 보정합니다.
    
#     Args:
#         observed_angle_deg: OpenCV로 측정한 각도 (도 단위)
#         camera_pitch_deg: 카메라가 내려다보는 각도 (기본 30도)
        
#     Returns:
#         보정된 실제 각도 (도 단위)
#     """
#     if observed_angle_deg == 0:
#         return 0.0
        
#     # 1. 라디안 변환
#     obs_rad = np.radians(observed_angle_deg)
#     pitch_rad = np.radians(camera_pitch_deg)
    
#     # 2. 보정 공식 적용: tan(real) = tan(obs) * cos(pitch)
#     # 코사인 값만큼 수직 길이가 압축되었으므로, 탄젠트 값에 코사인을 곱해 기울기를 완만하게 만듦
#     real_tan = np.tan(obs_rad) * np.cos(pitch_rad)
    
#     # 3. 아크탄젠트로 다시 각도 변환
#     real_rad = np.arctan(real_tan)
#     real_deg = np.degrees(real_rad)
    
#     return real_deg

def detect_pallet_tilt(image_input, mean_threshold=3.0, std_threshold=2.0):
    """
    실시간용 빠른 기울기 계산 함수 (그래프 없음).
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(data_list, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvspan(mean_val - std_val, mean_val + std_val, color='green', alpha=0.1, label=f'Std: {std_val:.2f}')
    ax.set_title('Tilt Angle Distribution')
    ax.set_xlabel('Angle (deg)')
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plot_img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
    plt.close(fig)
    h_plot, w_plot = plot_img.shape[:2]
    scale = height / h_plot
    plot_img_resized = cv2.resize(plot_img, (int(w_plot * scale), height))
    return plot_img_resized

# 기울기 분석 함수
def analyze_tilt_fast(roi_img, tilt_threshold=10):
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return "NORMAL", (0, 255, 0), 0.0

    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    angle = abs(rect[-1])

    # angle normalization
    if angle > 45:
        angle = 90 - angle

    if angle > tilt_threshold:
        return "TILTED", (0, 0, 255), angle

    return "NORMAL", (0, 255, 0), angle

def analyze_tilt_hough(roi_img, shift_min=2.0, shift_max=6.0, std_threshold=1.5):
    """
    화물의 미세한 쏠림(Shifting) 현상을 정밀 감지하는 함수.
    
    Args:
        roi_img: 입력 이미지
        shift_min: 쏠림으로 판단할 최소 각도 (이 이하는 정상)
        shift_max: 쏠림 감지 최대 각도 (이 이상은 이미 쓰러진 것으로 간주하여 별도 처리)
        std_threshold: 신뢰도 기준 (표준편차가 이보다 작아야 '진짜 쏠림'으로 인정)
        
    Returns:
        (status_str, color_bgr, angle_float)
    """
    
    # 1. 입력 예외 처리
    image = None
    if isinstance(roi_img, str):
        image = cv2.imread(roi_img)
    elif isinstance(roi_img, Image.Image):
        temp = np.array(roi_img)
        image = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
    elif isinstance(roi_img, np.ndarray):
        image = roi_img
    
    if image is None: return "Error: Image None", (0, 0, 0), 0.0

    # 2. 전처리 (Resize -> Canny)
    target_height = 800
    h, w = image.shape[:2]
    if h == 0 or w == 0: return "Error: Empty Frame", (0, 0, 0), 0.0

    scale = target_height / h
    image_resized = cv2.resize(image, (int(w * scale), target_height))
    
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # 3. 선 검출 (HoughLinesP)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=target_height / 10,
        maxLineGap=20
    )

    angles = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            
            if dy == 0 or abs(dx) > abs(dy): continue
                
            angle_rad = np.arctan(dx / dy)
            angle_deg = np.degrees(angle_rad)
            
            # [중요] 정밀 감지를 위해 ceil(올림) 제거하고 float 유지
            abs_angle = abs(angle_deg)
            
            if abs_angle > 45: continue # 노이즈 제거
                
            angles.append(abs_angle)

    # 4. 결과 분석 (Logic Refactoring)
    if not angles:
        return "NORMAL (No lines)", (0, 255, 0), 0.0

    avg_angle = np.mean(angles)    # 기울기 (얼마나?)
    std_dev = np.std(angles)       # 신뢰도 (확실해?)
    # --- 판단 로직 (User Logic) ---
    
    # CASE 1: 너무 극단적으로 기울어짐 -> 쏠림 감지 대상 아님 (이미 사고)
    if avg_angle > shift_max:
        return f"DANGER: EXTREME ({avg_angle:.1f}°)", (0, 0, 255), avg_angle

    # CASE 2: 정상 범위 (너무 미미함)
    if avg_angle < shift_min:
        return "NORMAL", (0, 255, 0), avg_angle

    # CASE 3: 쏠림 의심 구간 (shift_min ~ shift_max 사이)
    # 여기서 표준편차가 '거름망' 역할을 함
    
    if std_dev < std_threshold:
        # 평균은 떴는데, 편차가 작다? -> "모든 선이 쏠림을 가리킴" (진짜)
        return f"WARNING: SHIFTING ({avg_angle:.1f}°)", (0, 165, 255), avg_angle
    else:
        # 평균은 떴는데, 편차가 크다? -> "덜컹거려서 평균이 튄 것" (가짜/진동)
        return f"NORMAL (Vibration) ({avg_angle:.1f}°)", (0, 255, 0), avg_angle