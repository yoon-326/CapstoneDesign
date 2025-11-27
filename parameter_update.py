import cv2
import numpy as np
import time
from picamera2 import Picamera2
from ultralytics import YOLO
from scipy.optimize import curve_fit

# 모델 로드
model = YOLO("yolov8n-pose.pt")

# ==========================================
# [함수] 카메라 초기화
# ==========================================
def init_camera():
    print("[파라미터 통합 산출] 카메라 초기화 중...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
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
    l_sh, r_sh = kps[5], kps[6]
    l_hip, r_hip = kps[11], kps[12]
    
    # 신뢰도 체크
    if l_sh[2] < 0.5 or r_sh[2] < 0.5 or l_hip[2] < 0.5 or r_hip[2] < 0.5:
        return None
    
    shoulder_y = (l_sh[1] + r_sh[1]) / 2
    hip_y = (l_hip[1] + r_hip[1]) / 2
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
    
    # 측정할 거리 목록 (1m ~ 5m)
    # 정확도를 위해 1.5m, 2.5m 등 중간값도 찍으면 더 좋습니다.
    target_distances = [1.0, 2.0, 3.0, 4.0, 5.0]
    current_idx = 0
    
    print("\n" + "="*40)
    print("파라미터 자동 산출 모드")
    print("화면의 지시에 따라 거리를 맞추고 'Space' 키를 누르세요.")
    print("="*40)
    
    try:
        while current_idx < len(target_distances):
            frame = picam2.capture_array()
            target_dist = target_distances[current_idx]
            
            # 가이드 텍스트
            msg = f"Distance: {target_dist}m -> Press SPACE"
            cv2.putText(frame, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 현재 수집된 데이터 개수 표시
            cv2.putText(frame, f"Collected: {len(torso_data)}/{len(target_distances)}", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Parameter Update", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32: # Spacebar
                results = model(frame, verbose=False)
                
                # 가장 크고 선명한 사람 찾기
                max_torso = 0
                found = False
                
                for result in results:
                    if result.keypoints is not None:
                        for i in range(len(result.keypoints)):
                            # 단일 객체 키포인트 래핑 처리
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
                    time.sleep(0.5) # 중복 입력 방지
                else:
                    print("사람을 찾지 못했습니다. 다시 시도하세요.")
            
            elif key == ord('q'):
                print("중단됨.")
                break
        
        picam2.stop()
        cv2.destroyAllWindows()

        # --------------------------------------------
        # ★ [핵심] 데이터 분석 및 계수 산출
        # --------------------------------------------
        if len(torso_data) >= 5:
            print("\n데이터 분석 중...")
            
            X = np.array(torso_data)
            y = np.array(real_dist_data)
            
            # 1. 1차 공식 계수 (Alpha, Beta) 찾기
            popt, _ = curve_fit(inverse_func, X, y)
            alpha, beta = popt
            
            # 2. 1차 공식으로 예측값 생성
            y_pred_1st = (alpha / X) + beta
            
            # 3. 2차 보정 계수 (a, b, c) 찾기
            # 입력: 1차 예측값(y_pred_1st), 정답: 실제 거리(y)
            # y_real = a * (y_pred)^2 + b * (y_pred) + c
            coeffs = np.polyfit(y_pred_1st, y, 2)
            a, b, c = coeffs
            
            # --------------------------------------------
            # 결과 출력 (복사하기 좋게 포맷팅)
            # --------------------------------------------
            print("\n" + "="*50)
            print("[최종 결과] 아래 내용을 main.py에 복사해 넣으세요!")
            print("="*50)
            print("# [1] 통계 공식 계수 (1차)")
            print(f"ALPHA = {alpha:.2f}")
            print(f"BETA  = {beta:.2f}")
            print("\n# [2] 2차 곡선 보정 계수")
            print(f"CORRECT_A = {a:.6f}")
            print(f"CORRECT_B = {b:.6f}")
            print(f"CORRECT_C = {c:.6f}")
            print("="*50)
            
        else:
            print("데이터가 부족하여 계산할 수 없습니다.")

    except Exception as e:
        print(f"에러 발생: {e}")
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
