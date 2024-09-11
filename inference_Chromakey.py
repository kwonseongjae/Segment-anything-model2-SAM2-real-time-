import os
import torch
import numpy as np
import cv2
import time

# GPU 설정
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# SAM2 모델 불러오기
from sam2.build_sam import build_sam2_camera_predictor

sam2_checkpoint = "C:/Users/BT/segment-anything-2/checkpoints/sam2_hiera_b+.pt"
model_cfg = "C:/Users/BT/segment-anything-2/sam2_configs/sam2_hiera_b+.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# 카메라 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 배경 RGB 색상 설정 (chromakey color)
background_rgb = (0, 175, 57)

if_init = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (1920, 1080))
    width, height = frame.shape[1], frame.shape[0]

    # 시간 측정 시작
    start = time.time()

    # 모션 블러 제거 및 반전 처리
    preprocess_start = time.time()
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frame = cv2.flip(frame, 1)
    preprocess_end = time.time()

    if not if_init:
        init_start = time.time()
        predictor.load_first_frame(frame)
        if_init = True

        # 객체 감지 초기 설정
        ann_frame_idx = 0
        ann_obj_id = 1
        points = np.array([[width * 1 // 2, height * 2 // 3]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        bbox = np.array([[width * 1 // 4, height * 1 // 3], [width * 3 // 4, height]], dtype=np.float32)

        # 프롬프트 추가
        _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
            frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels, bbox=bbox
        )
        init_end = time.time()

        # 초기화 단계에서의 소요 시간 출력
        print(f"Initialization Time: {init_end - init_start:.4f} seconds")
    else:
        tracking_start = time.time()
        out_obj_ids, out_mask_logits = predictor.track(frame)
        tracking_end = time.time()

        all_mask = np.zeros((height, width, 1), dtype=np.uint8)
        for i in range(len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
            all_mask = cv2.bitwise_or(all_mask, out_mask)

        all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)

        # Replace the non-object areas with the background color
        background_frame = np.full_like(frame, background_rgb, dtype=np.uint8)
        frame = np.where(all_mask == 0, background_frame, frame)

        postprocess_start = time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        postprocess_end = time.time()

        # 각 단계별 소요 시간 출력
        print(f"Preprocessing Time: {preprocess_end - preprocess_start:.4f} seconds")
        print(f"Tracking Time: {tracking_end - tracking_start:.4f} seconds")
        print(f"Postprocessing Time: {postprocess_end - postprocess_start:.4f} seconds")

    # FPS 및 Latency 계산
    end_time = time.time()
    fps = 1 / (end_time - start)
    latency = (end_time - start) * 1000

    # 프레임에 FPS 및 Latency 표시
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Latency: {latency:.2f} ms", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
