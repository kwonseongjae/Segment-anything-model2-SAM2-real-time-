#SAM2 실시간 객체 인식 + FPS,Latency
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import time


# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
# select the device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

from sam2.build_sam import build_sam2_camera_predictor
import time

sam2_checkpoint = "C:/Users/BT/segment-anything-2/checkpoints/sam2_hiera_tiny.pt"
model_cfg = "C:/Users/BT/segment-anything-2/sam2_configs/sam2_hiera_t.yaml"

predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)


cap = cv2.VideoCapture(0)
if_init = False
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
fps_list = []
while True:
    
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not ret:
        break

    frame = cv2.resize(frame, (1920, 1080))
    width, height = frame.shape[1], frame.shape[0]
    #width, height = frame.shape[:2][::-1]
    start=time.time()
    # 모션 블러 제거 추가 (가우시안 블러 적용)
    #frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
    # **반전 제거**
    frame = cv2.flip(frame, 1)  # 좌우 반전을 제거합니다.
    if not if_init:

        predictor.load_first_frame(frame)
        if_init = True

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
        # Let's add a positive click at (x, y) = (210, 350) to get started

        # for labels, `1` means positive click and `0` means negative click
        # points = np.array([[660, 267]], dtype=np.float32)
        # labels = np.array([1], dtype=np.int32)

        # _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
        # )

        # add bbox
        points = np.array([[width * 1//2, height * 2//3]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        
        bbox = np.array([
            [width * 1 // 4, height * 1//3 ],  # 좌상단
            [width * 3 // 4, height]         # 우하단
        ], dtype=np.float32)
        
        #bbox = np.array([[100, 100], [600, 500]], dtype=np.float32)#
        _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
            frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels, bbox=bbox
        )

    else:
        out_obj_ids, out_mask_logits = predictor.track(frame)

        all_mask = np.zeros((height, width, 1), dtype=np.uint8)
        # print(all_mask.shape)
        for i in range(0, len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                np.uint8
            ) * 255

            all_mask = cv2.bitwise_or(all_mask, out_mask)

        all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
        frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
         
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    end=time.time()-start
    fps=1/end
    latency=1000*end
    fps_list.append(fps)
    if len(fps_list) > 30:  # 최근 30 프레임의 평균을 사용
        fps_list.pop(0)
    avg_fps = sum(fps_list) / len(fps_list)
    print("fps: " ,fps) 
    print("Latency: " ,latency)
    cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Latency: {latency:.2f} ms", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
# gif = imageio.mimsave("./result.gif", frame_list, "GIF", duration=0.00085)
