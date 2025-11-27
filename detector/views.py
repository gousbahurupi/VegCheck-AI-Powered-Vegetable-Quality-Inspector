# detector/views.py
import os
import cv2
import numpy as np
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.http import StreamingHttpResponse, JsonResponse
from pathlib import Path
from datetime import datetime
from .utils import MODEL, draw_detections, compute_stats, compute_percentages, get_suggestion_for_class

# Ensure MEDIA_ROOT exists
Path(settings.MEDIA_ROOT).mkdir(parents=True, exist_ok=True)

# Simple home/upload pages
def home(request):
    return render(request, 'detector/home.html')

def upload(request):
    # show same upload form as earlier
    return render(request, 'detector/upload.html')

def about(request):
    return render(request, 'detector/about.html')

# Process single image upload
def process_image_upload(request):
    """
    POST endpoint for single image upload -> runs YOLO -> returns result page
    """
    if request.method != "POST":
        return redirect("upload")

    image_file = request.FILES.get("image")
    if not image_file:
        return redirect("upload")

    fs = FileSystemStorage()
    # Save to media root under date folder
    date_folder = datetime.now().strftime("%d%B%Y")
    save_dir = os.path.join(settings.MEDIA_ROOT, date_folder)
    os.makedirs(save_dir, exist_ok=True)
    filename = fs.save(os.path.join(date_folder, image_file.name), image_file)
    abs_path = fs.path(filename)

    # Read image with cv2 (BGR)
    img_bgr = cv2.imread(abs_path)
    if img_bgr is None:
        return render(request, 'detector/error.html', {"message": "Unable to read uploaded image."})

    # YOLO inference on numpy image (Ultralytics supports numpy arrays)
    results = MODEL.predict(source=img_bgr, imgsz=640, conf=0.25, verbose=False)
    res = results[0]  # single image result

    # Annotate image and save annotated image
    annotated = draw_detections(img_bgr, res)
    out_annot_path = os.path.join(settings.MEDIA_ROOT, date_folder, f"annot_{image_file.name}")
    cv2.imwrite(out_annot_path, annotated)

    # Statistics
    stats = compute_stats(res)
    pct = compute_percentages(stats)

    # Suggestions: one per class detected (top few)
    suggestions = {}
    for cls_name, cnt in stats.get("counts", {}).items():
        suggestions[cls_name] = get_suggestion_for_class(cls_name)

    context = {
        "original": fs.url(filename),
        "annotated": fs.url(os.path.join(date_folder, f"annot_{image_file.name}")),
        "counts": stats.get("counts", {}),
        "total": stats.get("total", 0),
        "fresh_count": stats.get("fresh_count", 0),
        "rotten_count": stats.get("rotten_count", 0),
        "fresh_pct_count": round(pct["fresh_pct_count"], 2),
        "rotten_pct_count": round(pct["rotten_pct_count"], 2),
        "fresh_pct_conf": round(pct["fresh_pct_conf"], 2),
        "rotten_pct_conf": round(pct["rotten_pct_conf"], 2),
        "suggestions": suggestions,
    }

    return render(request, 'detector/result.html', context)

# Live camera stream (server-side inference)
def gen_live_frames(device_index=0, imgsz=416, conf=0.25):
    """
    Generator that yields multipart JPEG frames with detections drawn.
    """
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera device")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # run inference on resized frame (Ultralytics accepts numpy arrays)
            results = MODEL.predict(source=frame, imgsz=imgsz, conf=conf, verbose=False)
            res = results[0]
            annotated = draw_detections(frame, res)
            # encode to JPEG
            ret2, buf = cv2.imencode('.jpg', annotated)
            if not ret2:
                continue
            frame_bytes = buf.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

def live_feed(request):
    return StreamingHttpResponse(gen_live_frames(device_index=0), content_type='multipart/x-mixed-replace; boundary=frame')

def live_page(request):
    # simple page that shows <img src="/detector/live_feed/">
    return render(request, 'detector/live.html')

def process_image_upload(request):
    """
    POST endpoint for single image upload -> runs YOLO -> returns result page
    """
    if request.method != "POST":
        return redirect("detector:upload")

    image_file = request.FILES.get("image")
    if not image_file:
        return redirect("detector:upload")

    fs = FileSystemStorage()
    # Save to media root under date folder
    date_folder = datetime.now().strftime("%d%B%Y")
    save_dir = os.path.join(settings.MEDIA_ROOT, date_folder)
    os.makedirs(save_dir, exist_ok=True)
    filename = fs.save(os.path.join(date_folder, image_file.name), image_file)
    abs_path = fs.path(filename)

    # Read image with cv2 (BGR)
    img_bgr = cv2.imread(abs_path)
    if img_bgr is None:
        return render(request, 'detector/error.html', {"message": "Unable to read uploaded image."})

    # YOLO inference on numpy image (Ultralytics supports numpy arrays)
    # Keep imgsz lower for faster inference if needed (change to 320/640)
    results = MODEL.predict(source=img_bgr, imgsz=640, conf=0.25, verbose=False)
    res = results[0]  # single image result

    # Annotate image and save annotated image
    annotated = draw_detections(img_bgr, res)
    out_annot_path = os.path.join(settings.MEDIA_ROOT, date_folder, f"annot_{image_file.name}")
    cv2.imwrite(out_annot_path, annotated)

    # Statistics (counts + per-class avg confidence)
    stats = compute_stats(res)
    pct = compute_percentages(stats)

    # Suggestions: based on class avg confidence
    suggestions = {}
    for cls_name, cnt in stats.get("counts", {}).items():
        avg_conf = stats.get("class_conf_avg", {}).get(cls_name, 0.0)
        suggestions[cls_name] = get_suggestion_for_class(cls_name, avg_conf)

    # overall advice
    try:
        from .utils import get_overall_advice
        overall_advice = get_overall_advice(stats, pct)
    except Exception:
        overall_advice = ""

    context = {
        "original": fs.url(filename),
        "annotated": fs.url(os.path.join(date_folder, f"annot_{image_file.name}")),
        "counts": stats.get("counts", {}),
        "total": stats.get("total", 0),
        "fresh_count": stats.get("fresh_count", 0),
        "rotten_count": stats.get("rotten_count", 0),
        "fresh_pct_count": pct["fresh_pct_count"],
        "rotten_pct_count": pct["rotten_pct_count"],
        "fresh_avg_conf": pct.get("fresh_avg_conf", 0.0),
        "rotten_avg_conf": pct.get("rotten_avg_conf", 0.0),
        "class_conf_avg": stats.get("class_conf_avg", {}),
        "suggestions": suggestions,
        "overall_advice": overall_advice,
    }

    return render(request, 'detector/result.html', context)
