import cv2
from face_det_module.src.faceboxes_detector import *
import face_det_module.src.faceboxes_detector as faceboxes_detector
from PIL import Image
from PIL import Image
from face_det_module.src import faceboxes_detector  # ensure this import is correct

def crop(image, preprocess, input_size, use_gpu, device):
    detector = faceboxes_detector.FaceBoxesDetector(
        'FaceBoxes', 
        '/home/face/Desktop/LangAgent/face_det_module/weights/FaceBoxesV2.pth', 
        use_gpu, device
    )

    det_box_scale = 1.2
    image_height, image_width, _ = image.shape
    detections, check = detector.detect(image, 600, 0.8, 'max', 1)
    
    if check:
        det_xmin = detections[0][2]
        det_ymin = detections[0][3]
        det_width = detections[0][4]
        det_height = detections[0][5]
        det_xmax = det_xmin + det_width - 1
        det_ymax = det_ymin + det_height - 1

        det_xmin -= int(det_width * (det_box_scale-1)/2)
        det_ymin += int(det_height * (det_box_scale-1)/2)
        det_xmax += int(det_width * (det_box_scale-1)/2)
        det_ymax += int(det_height * (det_box_scale-1)/2)
        det_xmin = max(det_xmin, 0)
        det_ymin = max(det_ymin, 0)
        det_xmax = min(det_xmax, image_width-1)
        det_ymax = min(det_ymax, image_height-1)

        det_width = det_xmax - det_xmin + 1
        det_height = det_ymax - det_ymin + 1

        if det_width > det_height:
            buffer = int((det_width - det_height)/2)
            det_ymin = max(0, det_ymin - buffer)
            det_ymax = min(image_height - 1, det_ymax + buffer)
        elif det_width < det_height:
            buffer = int((det_height - det_width)/2)
            det_xmin = max(0, det_xmin - buffer)
            det_xmax = min(image_width - 1, det_xmax + buffer)

        box = [det_xmin, det_ymin, det_xmax, det_ymax]

        det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :]
        det_crop = cv2.resize(det_crop, (input_size, input_size))

        inputs = Image.fromarray(det_crop.astype('uint8'), 'RGB')
        inputs = preprocess(inputs).to(device).unsqueeze(0)

        #print(f'face box : {check}')
        return inputs, check, box

    else:
        print(f'face box : {check}')
        return None, check, None

            




def crop_qt(image, preprocess, input_size, use_gpu, device):
    detector = FaceBoxesDetector(
        'FaceBoxes',
        '/home/face/Desktop/NLA/face_det_module/src/weights/FaceBoxesV2.pth',
        use_gpu,
        device
    )

    det_box_scale = 1.2
    image_height, image_width, _ = image.shape
    detections, check = detector.detect(image, 600, 0.8, 'max', 1)

    if check:
        det_xmin = detections[0][2]
        det_ymin = detections[0][3]
        det_width = detections[0][4]
        det_height = detections[0][5]
        det_xmax = det_xmin + det_width - 1
        det_ymax = det_ymin + det_height - 1

        det_xmin -= int(det_width * (det_box_scale - 1) / 2)
        det_ymin += int(det_height * (det_box_scale - 1) / 2)
        det_xmax += int(det_width * (det_box_scale - 1) / 2)
        det_ymax += int(det_height * (det_box_scale - 1) / 2)
        det_xmin = max(det_xmin, 0)
        det_ymin = max(det_ymin, 0)
        det_xmax = min(det_xmax, image_width - 1)
        det_ymax = min(det_ymax, image_height - 1)

        det_width = det_xmax - det_xmin + 1
        det_height = det_ymax - det_ymin + 1

        if det_width > det_height:
            buffer = int((det_width - det_height) / 2)
            det_ymin = max(0, det_ymin - buffer)
            det_ymax = min(image_height - 1, det_ymax + buffer)
        elif det_width < det_height:
            buffer = int((det_height - det_width) / 2)
            det_xmin = max(0, det_xmin - buffer)
            det_xmax = min(image_width - 1, det_xmax + buffer)

        det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :]
        det_crop = cv2.resize(det_crop, (input_size, input_size))
        inputs = Image.fromarray(det_crop.astype('uint8'), 'RGB')
        inputs = preprocess(inputs).to(device).unsqueeze(0)

        print(f'face box : {check}')
        return inputs, check, (det_xmin, det_ymin, det_xmax, det_ymax)

    else:
        print(f'face box : {check}')
        return check, check, None


# face_crop.py 에 정의되어 있어야 함
import cv2
import numpy as np
from PIL import Image
from face_det_module.src.faceboxes_detector import FaceBoxesDetector_multi


import cv2
import numpy as np
from PIL import Image
from face_det_module.src.faceboxes_detector import FaceBoxesDetector_multi


def compute_iou(box1, box2):
    """ box = (x1, y1, x2, y2) 형태 """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1 + 1)
    inter_h = max(0, y2 - y1 + 1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = inter_area / float(area1 + area2 - inter_area + 1e-6)
    return iou

def crop_multi(image, preprocess, input_size, use_gpu, device, threshold=0.8, iou_thresh=0.6, pad_ratio=0.1):
    """
    이미지에서 다중 얼굴을 감지하고, 중복 제거 및 padding 포함 crop 후 모델 입력 텐서로 반환

    Returns:
        faces: List of (tensor, (x1, y1, x2, y2), score)
    """
    if image is None or image.shape[0] == 0 or image.shape[1] == 0:
        print("Invalid image passed to crop_multi.")
        return []

    detector = FaceBoxesDetector_multi(
        'FaceBoxes',
        '/home/face/Desktop/NLA/face_det_module/src/weights/FaceBoxesV2.pth',
        use_gpu,
        device
    )

    image_height, image_width, _ = image.shape
    detections, check = detector.detect(image, thresh_xy=600, thresh=threshold)
    faces = []

    if not check:
        return []

    # 1. Confidence 기준 내림차순 정렬
    raw_boxes = []
    for det in detections:
        _, score, x, y, w, h = det
        x1, y1 = int(x), int(y)
        x2, y2 = x1 + int(w), y1 + int(h)
        raw_boxes.append((score, (x1, y1, x2, y2)))

    raw_boxes = sorted(raw_boxes, key=lambda x: x[0], reverse=True)

    # 2. IoU 기반 중복 제거 (NMS 유사)
    selected = []
    for score, box in raw_boxes:
        overlapped = False
        for _, sel_box in selected:
            if compute_iou(box, sel_box) > iou_thresh:
                overlapped = True
                break
        if not overlapped:
            selected.append((score, box))

    # 3. crop + 정사각형 보정 + padding + resize
    for score, (x1, y1, x2, y2) in selected:
        # 정사각형 보정
        w, h = x2 - x1, y2 - y1
        if w > h:
            diff = w - h
            y1 = max(0, y1 - diff // 2)
            y2 = min(image_height - 1, y2 + diff - diff // 2)
        elif h > w:
            diff = h - w
            x1 = max(0, x1 - diff // 2)
            x2 = min(image_width - 1, x2 + diff - diff // 2)

        # padding 비율 적용
        pad_w = int((x2 - x1) * pad_ratio)
        pad_h = int((y2 - y1) * pad_ratio)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(image_width - 1, x2 + pad_w)
        y2 = min(image_height - 1, y2 + pad_h)

        crop = image[y1:y2, x1:x2]
        if crop.shape[0] <= 0 or crop.shape[1] <= 0:
            continue

        crop = cv2.resize(crop, (input_size, input_size))
        tensor = preprocess(Image.fromarray(crop)).to(device).unsqueeze(0)
        faces.append((tensor, (x1, y1, x2, y2), score))

    return faces


