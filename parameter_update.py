import cv2
import numpy as np
import time
from picamera2 import Picamera2
from ultralytics import YOLO
from scipy.optimize import curve_fit

# 모델 로드
model = YOLO("figure_pose.pt")

# ==========================================
# [함수] 카메라 초기화 (Wide View)
# ==========================================
def init_camera():
    print("[파라미터 통합 산출] 카메라 초기화 중...")
    picam2 = Picamera2()
    # [수정] 1280x960 (4:3) 설정으로 센서 전체 시야 확보 (크롭 방지)
    config = picam2.create_preview_configuration(main={"size": (1640, 1232), "format": "BGR888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    return picam2

# ==========================================
# [함수] 데이터 처리
# ==========================================
def get_torso_length(keypoints):
    """ 어깨 중점 ~ 골반 중점 길이 (상반신) """
    kps = keypoints.data[0].cpu().numpy()
    
    # 8개 관절 모델 기준 인덱스
    # 0: Left Shoulder, 1: Right Shoulder
    # 2: Left Hip,      3: Right Hip
    l_sh, r_sh = kps[0], kps[1]
    l_hip, r_hip = kps[2], kps[3]
    
    # 신뢰도 체크 (0.5 미만이면 무시)
    if l_sh[2] < 0.5 or r_sh[2] < 0.5 or l_hip[2] < 0.5 or r_hip[2] < 0.5:
        return None
    
    shoulder_y = (l_sh[1] + r_sh[1]) / 2
    hip_y = (l_hip[1] + r_hip[1]) / 2
    
    # 절대값 반환 (y축이 아래로 갈수록 커지므로)
    return abs(hip_y - shoulder_y)

# 1차 공식 함수 형태 (반비례)
def inverse_func(h, a, b):
    return a * (1/h) + b

# ==========================================
# [메인] 실행 로직
# ==========================================
def main():
    picam2 = init_camera()
    
    torso_data = []      # 상반신 길이 (픽셀)
    real_dist_data = []  # 실제 거리 (미터)
    
    # 측정할 거리 목록 (정확도를 위해 1m ~ 4m 사이를 촘촘히 하는 것 권장)
    target_distances = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    current_idx = 0
    
    print("\n" + "="*40)
    print("파라미터 자동 산출 모드 (Wide View)")
    print("화면의 지시에 따라 거리를 맞추고 'Space' 키를 누르세요.")
    print("="*40)
    
    try:
        while current_idx < len(target_distances):
            # 1. 고해상도 원본 캡처
            raw_frame = picam2.capture_array()
            
            # 2. 640x480으로 리사이징
            # 메인 코드에서 640x480 기준으로 거리를 계산하므로,
            # 여기서 데이터를 수집할 때도 똑같은 크기여야 합니다.
            frame = cv2.resize(raw_frame, (640, 480))
            
            target_dist = target_distances[current_idx]
            
            # 가이드 텍스트
            msg = f"Target: {target_dist}m -> Press SPACE"
            cv2.putText(frame, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 현재 수집된 데이터 개수 표시
            cv2.putText(frame, f"Collected: {len(torso_data)}/{len(target_distances)}", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Parameter Update", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32: # Spacebar
                # 3. 리사이즈된 프레임(frame)을 모델에 입력
                results = model(frame, verbose=False)
                
                # 가장 크고 선명한 사람 찾기
                max_torso = 0
                found = False
                
                for result in results:
                    if result.keypoints is not None:
                        for i in range(len(result.keypoints)):
                            temp_wrapper = type('', (), {})()
                            temp_wrapper.data = result.keypoints.data[i].unsqueeze(0)
                            
                            torso = get_torso_length(temp_wrapper)
                            if torso and torso > max_torso:
                                max_torso = torso
                                found = True
                
                if found:
                    torso_data.append(max_torso)
                    real_dist_data.append(target_dist)
                    print(f"[OK] 거리 {target_dist}m : 상반신 {max_torso:.1f}px")
                    current_idx += 1
                    time.sleep(0.5)
                else:
                    print("사람을 찾지 못했습니다. 전신이 잘 보이게 서주세요.")
            
            elif key == ord('q'):
                print("중단됨.")
                break
        
        picam2.stop()
        cv2.destroyAllWindows()

        # --------------------------------------------
        # 데이터 분석 및 계수 산출
        # --------------------------------------------
        if len(torso_data) >= 5:
            print("\n데이터 분석 중...")
            
            X = np.array(torso_data)
            y = np.array(real_dist_data)
            
            # 1. 1차 공식 계수 (Alpha, Beta)
            popt, _ = curve_fit(inverse_func, X, y)
            alpha, beta = popt
            
            # 2. 1차 예측값 생성
            y_pred_1st = (alpha / X) + beta
            
            # 3. 2차 보정 계수 (a, b, c)
            coeffs = np.polyfit(y_pred_1st, y, 2)
            a, b, c = coeffs
            
            print("\n" + "="*50)
            print("[최종 결과] 아래 내용을 main.py의 설정 부분에 붙여넣으세요!")
            print("="*50)
            print(f"ALPHA = {alpha:.2f}")
            print(f"BETA  = {beta:.2f}")
            print(f"\nCORRECT_A = {a:.6f}")
            print(f"CORRECT_B = {b:.6f}")
            print(f"CORRECT_C = {c:.6f}")
            print("="*50)
            
        else:
            print("데이터가 부족합니다.")

    except Exception as e:
        print(f"에러 발생: {e}")
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()