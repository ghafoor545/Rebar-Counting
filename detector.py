import os
import uuid
import base64
from typing import Optional

import cv2
import numpy as np
import requests
import onnxruntime as ort
import streamlit as st

from config import MODEL_PATH, DET_DIR, THUMB_DIR
from db import get_conn
from utils import utc_now_iso


# ------------------------------
# Model loading (ONNX Runtime)
# ------------------------------
def _parse_in_hw(onnx_input_shape, default_hw=(640, 640)):
    def _to_int(v, default):
        try:
            iv = int(v)
            return iv if iv > 0 else default
        except Exception:
            return default

    if isinstance(onnx_input_shape, (list, tuple)) and len(onnx_input_shape) == 4:
        H = _to_int(onnx_input_shape[2], default_hw[0])
        W = _to_int(onnx_input_shape[3], default_hw[1])
        return (H, W)
    return default_hw


def _validate_model_path(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found at: {model_path}")
    if not model_path.lower().endswith(".onnx"):
        raise ValueError(
            f"Expected a .onnx file, got: {os.path.splitext(model_path)[1]}"
        )


def _create_session(model_path: str):
    _validate_model_path(model_path)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model at {model_path}. Error: {e}")
    in_name = sess.get_inputs()[0].name
    in_hw = _parse_in_hw(sess.get_inputs()[0].shape, default_hw=(640, 640))
    return {"sess": sess, "in_name": in_name, "in_hw": in_hw, "path": model_path}


if hasattr(st, "cache_resource"):
    @st.cache_resource
    def load_model():
        return _create_session(MODEL_PATH)
else:
    @st.cache(allow_output_mutation=True)
    def load_model():
        return _create_session(MODEL_PATH)


model = load_model()


# ------------------------------
# Detection helpers
# ------------------------------
def fetch_snapshot(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None, f"Failed to fetch snapshot: HTTP {r.status_code}"
        nparr = np.frombuffer(r.content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return None, "Failed to decode image from snapshot."
        return image, None
    except Exception as e:
        return None, f"Snapshot fetch error: {e}"


def to_hd_1080p(image_bgr, background=(18, 24, 31)):
    target_w, target_h = 1920, 1080
    h, w = image_bgr.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), background, dtype=np.uint8)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return im, r, (left, top)


def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def scale_boxes(boxes, r, dwdh, orig_shape):
    left, top = dwdh
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top
    boxes[:, :4] /= r
    h, w = orig_shape[:2]
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    return boxes


def iou_one_to_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1]) + 1e-6
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) + 1e-6
    return inter / (area1 + area2 - inter + 1e-6)


def nms(boxes, scores, iou_thres=0.45, max_det=10000):
    if len(boxes) == 0:
        return []
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if len(keep) >= max_det or idxs.size == 1:
            break
        ious = iou_one_to_many(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thres]
    return keep


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def maybe_sigmoid(arr):
    a = np.asarray(arr)
    if a.size == 0:
        return a
    mn, mx = np.min(a), np.max(a)
    if 0.0 <= mn and mx <= 1.0:
        return a
    return sigmoid(a)


def parse_onnx_outputs(outs):
    arrs = [np.asarray(o) for o in outs]

    # YOLO-NAS / ultralytics NMS-like outputs
    for a in arrs:
        if a.ndim == 3 and a.shape[-1] == 6:
            return "nms", (a[0] if a.shape[0] == 1 else a)
        if a.ndim == 2 and a.shape[-1] == 6:
            return "nms", a

    # "boxes + scores + classes" style
    num = None
    boxes = None
    scores = None
    classes = None

    for a in arrs:
        if a.ndim == 1 and a.size == 1:
            try:
                num = int(a.reshape(-1)[0])
            except Exception:
                pass

    for a in arrs:
        if a.ndim >= 2 and a.shape[-1] == 4:
            b = a[0] if a.ndim == 3 else a
            boxes = b.astype(np.float32)

    for a in arrs:
        if a.ndim >= 1 and a.shape[-1] != 4 and a.dtype.kind in "fc":
            s = a[0] if a.ndim == 2 else a
            if s.ndim == 1:
                scores = s.astype(np.float32)

    for a in arrs:
        if a.ndim >= 1 and a.shape[-1] != 4:
            c = a[0] if a.ndim == 2 else a
            if c.ndim == 1:
                classes = np.round(c).astype(np.int32)

    if boxes is not None and scores is not None and classes is not None:
        N = min(len(boxes), len(scores), len(classes))
        if num is not None and 0 < num <= N:
            N = num
        dets = np.zeros((N, 6), dtype=np.float32)
        dets[:, :4] = boxes[:N]
        dets[:, 4] = scores[:N]
        dets[:, 5] = classes[:N]
        return "nms", dets

    # Fallback: raw YOLO-style outputs [B, anchors, C]
    z = arrs[0]
    if z.ndim == 3:
        if z.shape[1] < z.shape[2]:
            z = np.transpose(z, (0, 2, 1))
        z = z[0]
    elif z.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected ONNX output shape: {z.shape}")
    return "raw", z


def preprocess_for_onnx(image_bgr, in_hw):
    in_h, in_w = in_hw
    lb, r, dwdh = letterbox(image_bgr, (in_h, in_w))
    img = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img, r, dwdh


def draw_centered_ids(image_bgr, boxes):
    """
    Draw perfectly centered white IDs on detected boxes
    with adaptive font size for both close-up and far shots
    """
    img = image_bgr.copy()
    img_height, img_width = img.shape[:2]
    
    # Sort boxes by y-coordinate (top to bottom), then by x (left to right)
    sorted_boxes = sorted(boxes, key=lambda b: (int(b[1]), int(b[0])))
    
    for idx, box in enumerate(sorted_boxes, start=1):
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Calculate box dimensions
        box_width = x2 - x1
        box_height = y2 - y1
        box_size = min(box_width, box_height)
        
        # Smart adaptive font scaling based on box size
        if box_size < 25:  # Very small boxes (far shots)
            font_scale = 0.25
            thickness = 1
        elif box_size < 40:  # Small boxes
            font_scale = 0.3
            thickness = 1
        elif box_size < 70:  # Medium boxes
            font_scale = 0.4
            thickness = 1
        elif box_size < 120:  # Large boxes
            font_scale = 0.5
            thickness = 2
        elif box_size < 200:  # Very large boxes
            font_scale = 0.7
            thickness = 2
        else:  # Extremely large boxes (very close-up shots)
            font_scale = 0.9
            thickness = 3
        
        # Center coordinates
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # ID text
        id_text = str(idx)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size for perfect centering
        (text_width, text_height), baseline = cv2.getTextSize(
            id_text, font, font_scale, thickness
        )
        
        # Ensure text fits within box (reduce font if needed)
        max_text_width = box_width * 0.8
        max_text_height = box_height * 0.8
        
        if text_width > max_text_width or text_height > max_text_height:
            # Reduce font size proportionally
            width_ratio = max_text_width / text_width if text_width > 0 else 1
            height_ratio = max_text_height / text_height if text_height > 0 else 1
            reduction_factor = min(width_ratio, height_ratio) * 0.9
            
            font_scale = font_scale * reduction_factor
            thickness = max(1, int(thickness * reduction_factor))
            
            # Recalculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                id_text, font, font_scale, thickness
            )
        
        # Calculate perfect centered position
        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2
        
        # Ensure text stays within image bounds
        text_x = max(0, min(text_x, img_width - text_width))
        text_y = max(text_height, min(text_y, img_height))
        
        # Draw subtle black background for better readability
        padding = max(2, int(font_scale * 3))
        bg_x1 = max(0, text_x - padding)
        bg_y1 = max(0, text_y - text_height - padding)
        bg_x2 = min(img_width, text_x + text_width + padding)
        bg_y2 = min(img_height, text_y + padding)
        
        # Draw background rectangle
        cv2.rectangle(
            img,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            (0, 0, 0),  # Black
            -1  # Filled
        )
        
        # Draw white ID text (perfectly centered)
        cv2.putText(
            img,
            id_text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),  # White
            thickness,
            cv2.LINE_AA
        )
    
    return img, sorted_boxes


def detect_rebars(image_bgr, model, class_id=0, conf=0.6, iou=0.5, max_det=10000):
    try:
        sess = model["sess"]
        in_name = model["in_name"]
        in_hw = model["in_hw"]
        in_h, in_w = in_hw

        # Preprocess
        blob, r, dwdh = preprocess_for_onnx(image_bgr, in_hw)

        # Inference
        outs = sess.run(None, {in_name: blob})
        kind, data = parse_onnx_outputs(outs)

        dets_xyxy = []

        if kind == "nms":
            # Already-NMSed outputs [N,6] = [x1,y1,x2,y2,score,class]
            d = data.reshape(-1, 6).astype(np.float32)
            cls = np.round(d[:, 5]).astype(np.int32)
            scr = d[:, 4]

            mask = (scr >= conf) & (cls == int(class_id))
            if np.any(mask):
                b = d[mask, :4]

                # If boxes are normalized to [0,1], scale them
                if np.nanmax(b) <= 1.5:
                    b[:, [0, 2]] *= float(in_w)
                    b[:, [1, 3]] *= float(in_h)

                b = scale_boxes(b, r, dwdh, image_bgr.shape)
                dets_xyxy = b.tolist()

        else:
            # Raw YOLO-style predictions
            C = data.shape[1]
            boxes = data[:, :4].astype(np.float32)

            scores_mat_a = None
            scores_mat_b = None

            if C - 4 > 0:
                # case: boxes + per-class scores
                scores_mat_a = maybe_sigmoid(data[:, 4:])
            if C - 5 > 0:
                # case: boxes + obj + class => combine
                obj = maybe_sigmoid(data[:, 4:5])
                cls_b = maybe_sigmoid(data[:, 5:])
                scores_mat_b = obj * cls_b

            def count_above(m, t=0.25):
                return int((m is not None) and (m.max(axis=1) >= t).sum())

            cnt_a = count_above(scores_mat_a)
            cnt_b = count_above(scores_mat_b)

            scores_mat = scores_mat_a if (
                scores_mat_a is not None and (scores_mat_b is None or cnt_a >= cnt_b)
            ) else scores_mat_b

            if scores_mat is None or scores_mat.shape[1] <= class_id:
                annotated = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                return annotated, 0, "Model scores not found or class_id out of range."

            scores = scores_mat[:, class_id]

            # If boxes are normalized [0,1], scale them to input size
            if np.nanmax(boxes) <= 1.5:
                boxes[:, [0, 2]] *= float(in_w)
                boxes[:, [1, 3]] *= float(in_h)

            # Try interpreting as xywh -> xyxy
            b_xyxy_from_xywh = xywh2xyxy(boxes)
            valid1 = (
                (b_xyxy_from_xywh[:, 2] > b_xyxy_from_xywh[:, 0])
                & (b_xyxy_from_xywh[:, 3] > b_xyxy_from_xywh[:, 1])
            ).sum()

            # Or as already xyxy
            b_xyxy_as_is = boxes.copy()
            valid2 = (
                (b_xyxy_as_is[:, 2] > b_xyxy_as_is[:, 0])
                & (b_xyxy_as_is[:, 3] > b_xyxy_as_is[:, 1])
            ).sum()

            b_xyxy = b_xyxy_from_xywh if valid1 >= valid2 else b_xyxy_as_is

            keep_mask = scores >= float(conf)
            if np.any(keep_mask):
                b_xyxy = b_xyxy[keep_mask]
                s = scores[keep_mask]

                # De-letterbox back to original image
                b_xyxy = scale_boxes(b_xyxy, r, dwdh, image_bgr.shape)

                # Filter out tiny boxes
                wh = (b_xyxy[:, 2:4] - b_xyxy[:, 0:2]).clip(min=0)
                small = (wh[:, 0] < 4.0) | (wh[:, 1] < 4.0)
                if np.any(small):
                    b_xyxy = b_xyxy[~small]
                    s = s[~small]

                # Limit to max_det
                if len(s) > max_det:
                    topk = np.argsort(-s)[:max_det]
                    b_xyxy = b_xyxy[topk]
                    s = s[topk]

                keep = nms(b_xyxy, s, iou_thres=float(iou), max_det=max_det)
                if keep:
                    b_xyxy = b_xyxy[keep]
                    dets_xyxy = b_xyxy.tolist()

        # Draw bounding boxes AND centered white IDs
        if dets_xyxy:
            # First draw boxes (optional - you can keep this or remove)
            annotated = image_bgr.copy()
            for (x1, y1, x2, y2) in dets_xyxy:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 250, 180), 1)
            
            # Draw centered white IDs
            annotated, sorted_boxes = draw_centered_ids(annotated, dets_xyxy)
            count = len(sorted_boxes)
        else:
            annotated = image_bgr.copy()
            count = 0

        # Compose into 1080p frame with banner
        hd = to_hd_1080p(annotated, background=(18, 24, 31))
        banner_height = 100
        img_h, img_w, _ = hd.shape
        heading = f"Rebars detected: {count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        size, _ = cv2.getTextSize(heading, font, 2, 4)

        # Draw white banner
        cv2.rectangle(hd, (0, 0), (img_w, banner_height), (255, 255, 255), -1)
        
        # Draw black text on white banner
        cv2.putText(
            hd,
            heading,
            ((img_w - size[0]) // 2, (banner_height + size[1]) // 2),
            font,
            2,
            (0, 0, 0),  # Black text
            4,
            lineType=cv2.LINE_AA,
        )

        # Return RGB for Streamlit
        return cv2.cvtColor(hd, cv2.COLOR_BGR2RGB), count, None

    except Exception as e:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), 0, f"Detection error: {e}"


# ------------------------------
# Detection record helpers (DB)
# ------------------------------
def save_image_files(image_rgb: np.ndarray, det_id: str):
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    img_path = os.path.join(DET_DIR, f"{det_id}.jpg")
    thumb_path = os.path.join(THUMB_DIR, f"{det_id}.jpg")

    cv2.imwrite(img_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    h, w = img_bgr.shape[:2]
    tw = 360
    th = int(h * (tw / w))
    thumb = cv2.resize(img_bgr, (tw, th), interpolation=cv2.INTER_AREA)
    cv2.imwrite(thumb_path, thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    return img_path, thumb_path


def record_detection(
    user_id: int, processed_rgb: np.ndarray, count: int, stream_url: str, snapshot_url: str
):
    det_id = str(uuid.uuid4())
    img_path, thumb_path = save_image_files(processed_rgb, det_id)
    h, w = processed_rgb.shape[:2]

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO detections (
            id, user_id, timestamp, stream_url, snapshot_url,
            image_path, thumb_path, count, width, height
        )
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (
            det_id,
            user_id,
            utc_now_iso(),
            stream_url,
            snapshot_url,
            img_path,
            thumb_path,
            count,
            w,
            h,
        ),
    )
    conn.commit()
    conn.close()
    return det_id


def list_detections(user_id: int, page: int, per_page: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM detections WHERE user_id=?", (user_id,))
    total = cur.fetchone()["c"]
    offset = (page - 1) * per_page
    cur.execute(
        "SELECT * FROM detections WHERE user_id=? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        (user_id, per_page, offset),
    )
    rows = cur.fetchall()
    conn.close()
    return rows, total


def get_detection(det_id: str, user_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM detections WHERE id=? AND user_id=? LIMIT 1", (det_id, user_id)
    )
    row = cur.fetchone()
    conn.close()
    return row


def delete_detection(det_id: str, user_id: int):
    row = get_detection(det_id, user_id)
    if not row:
        return False

    try:
        if os.path.exists(row["image_path"]):
            os.remove(row["image_path"])
        if os.path.exists(row["thumb_path"]):
            os.remove(row["thumb_path"])
    except Exception:
        pass

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM detections WHERE id=? AND user_id=?", (det_id, user_id))
    conn.commit()
    conn.close()
    return True


# ------------------------------
# Image → data URI helpers
# ------------------------------
def img_to_data_uri(
    image_rgb: np.ndarray, quality: int = 90, max_w: Optional[int] = None
) -> Optional[str]:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if max_w is not None:
        h, w = bgr.shape[:2]
        if w > max_w:
            new_h = int(h * (max_w / w))
            bgr = cv2.resize(bgr, (max_w, new_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def file_to_data_uri(path: str, max_w: int = 120, quality: int = 85) -> Optional[str]:
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * (max_w / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
