# detector/utils.py
import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch

# -------- LOAD YOLO MODEL ---------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "best.pt")
MODEL = YOLO(MODEL_PATH)    
print(">>> YOLO Device:", MODEL.device)
print(">>> Torch CUDA available:", torch.cuda.is_available())

# -------- DRAW DETECTIONS ---------
def draw_detections(image_bgr, results):
    """Draw bounding boxes and labels on a BGR image (cv2 format)."""
    annotated = image_bgr.copy()
    for box in results.boxes:
        # box.xyxy is tensor-like; convert to ints
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = results.names[cls_id]
        # green for fresh, red for rotten, blue otherwise
        color = (34, 197, 94) if "fresh" in label.lower() else (239, 68, 68)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        txt = f"{label} {conf:.2f}"
        # put text background for readability
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, txt, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    return annotated

# -------- COMPUTE STATS ----------
def compute_stats(result):
    """
    From an Ultralytics 'result' object compute:
      - counts per class
      - total boxes
      - fresh_count / rotten_count
      - average confidence per class (0-100)
      - average confidence for fresh / rotten groups (0-100)
    Returns a dict with keys:
      counts, total, fresh_count, rotten_count, class_conf_avg (dict), fresh_conf_avg, rotten_conf_avg
    """
    counts = {}
    conf_sums = {}
    total = 0
    fresh_count = 0
    rotten_count = 0
    fresh_conf_sum = 0.0
    rotten_conf_sum = 0.0

    for box in result.boxes:
        cls_name = result.names[int(box.cls[0])]
        conf = float(box.conf[0])  # 0..1
        counts[cls_name] = counts.get(cls_name, 0) + 1
        conf_sums[cls_name] = conf_sums.get(cls_name, 0.0) + conf
        total += 1
        if "fresh" in cls_name.lower():
            fresh_count += 1
            fresh_conf_sum += conf
        else:
            rotten_count += 1
            rotten_conf_sum += conf

    # compute class-wise avg confidences in percent
    class_conf_avg = {}
    for cls, cnt in counts.items():
        avg_conf = (conf_sums.get(cls, 0.0) / max(cnt, 1)) * 100.0
        class_conf_avg[cls] = round(avg_conf, 2)

    # group averages
    fresh_conf_avg = round((fresh_conf_sum / max(fresh_count, 1)) * 100.0, 2) if fresh_count else 0.0
    rotten_conf_avg = round((rotten_conf_sum / max(rotten_count, 1)) * 100.0, 2) if rotten_count else 0.0

    return {
        "counts": counts,
        "total": total,
        "fresh_count": fresh_count,
        "rotten_count": rotten_count,
        "class_conf_avg": class_conf_avg,
        "fresh_conf_avg": fresh_conf_avg,
        "rotten_conf_avg": rotten_conf_avg
    }

# -------- PERCENTAGES -------------
def compute_percentages(stats):
    """
    Given stats dict from compute_stats, compute:
      - fresh/rotten percent by count
      - fresh/rotten avg confidence (already in stats)
    Returns dict with keys:
      fresh_pct_count, rotten_pct_count, fresh_avg_conf, rotten_avg_conf
    """
    total = max(stats.get("total", 1), 1)
    fresh_pct_count = (stats.get("fresh_count", 0) / total) * 100.0
    rotten_pct_count = (stats.get("rotten_count", 0) / total) * 100.0

    return {
        "fresh_pct_count": round(fresh_pct_count, 2),
        "rotten_pct_count": round(rotten_pct_count, 2),
        "fresh_avg_conf": stats.get("fresh_conf_avg", 0.0),
        "rotten_avg_conf": stats.get("rotten_conf_avg", 0.0),
        "class_conf_avg": stats.get("class_conf_avg", {})
    }

# -------- SUGGESTIONS -------------
def get_suggestion_for_class(cls_name, avg_conf_percent):
    """
    Return a human-friendly suggestion string based on:
      - cls_name (e.g. 'fresh apple' or 'rotten banana')
      - avg_conf_percent (0..100)
    Rules (example):
      - fresh & high confidence => storage + recipe suggestions
      - fresh but low confidence => re-check / store carefully
      - rotten low-medium => use for processing (jam/smoothie)
      - rotten high => compost/organic fertilizer / safe disposal
    """
    cls = cls_name.lower()
    # choose base item name (remove 'fresh ' / 'rotten ')
    item = cls.replace("fresh ", "").replace("rotten ", "").strip().replace("_", " ")

    conf = avg_conf_percent
    # Fresh cases
    if "fresh" in cls:
        if conf >= 85:
            return f"✅ {item.title()} appears fresh (confidence {conf}%). Suggest storing at recommended temperature and shelf-life: keep cool, use within standard shelf-life. Try recipes: fresh salads, raw consumption."
        if 60 <= conf < 85:
            return f"🟢 {item.title()} likely fresh (confidence {conf}%). Store in a cool, dry place or refrigerator (if applicable). Good for cooking or salads within 2–5 days."
        if 30 <= conf < 60:
            return f"🟡 {item.title()} borderline freshness (confidence {conf}%). Consider refrigerating and using soon; inspect for small blemishes. Good for cooking (soups, sautés) or roasting."
        return f"🔎 {item.title()} low-confidence fresh ({conf}%). Consider visual re-check; avoid long storage, cook or use immediately."

    # Rotten cases
    else:
        if conf >= 85:
            return f"⚠️ {item.title()} is highly rotten (confidence {conf}%). Not safe for consumption. Consider composting, organic fertilizer processing, or safe disposal. Avoid feeding to pets or livestock unless processed safely."
        if 65 <= conf < 85:
            return f"🟠 {item.title()} significantly rotten (confidence {conf}%). Use for industrial processing (biogas) or compost. Not recommended for direct consumption."
        if 40 <= conf < 65:
            return f"🟡 {item.title()} partly spoiled (confidence {conf}%). Can be salvaged for cooked recipes (purees, jams, pickles) after removing bad portions. Consider refrigerator and quick use."
        if 20 <= conf < 40:
            return f"🔵 {item.title()} minor spoilage (confidence {conf}%). Trim bad parts; can be used in smoothies, baking (banana breads), or pickling to avoid waste."
        return f"⚪ {item.title()} low-confidence rotten ({conf}%). Inspect manually — may be ok after trimming; if unsure, discard."

# optional helper to produce a short summary suggestion
def get_overall_advice(stats, percentages):
    """
    Returns a short overall advice string combining group counts and confidences.
    """
    fresh_pct = percentages.get("fresh_pct_count", 0.0)
    rotten_pct = percentages.get("rotten_pct_count", 0.0)
    fresh_conf = percentages.get("fresh_avg_conf", 0.0)
    rotten_conf = percentages.get("rotten_avg_conf", 0.0)

    if rotten_pct > 50 and rotten_conf > 60:
        return "Warning: majority detected as rotten — avoid selling/serving. Consider composting or processing."
    if fresh_pct > 70 and fresh_conf > 70:
        return "Good: most items are fresh — safe for sale/consumption with normal storage."
    return "Mixed results — review per-item suggestions below and handle accordingly."
