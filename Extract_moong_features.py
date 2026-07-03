# ****************************************************************************************************************************************************************
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
import os
import glob


# ══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION 
# ══════════════════════════════════════════════════════════════

def extract_features(image_path: str, debug_out_dir: str = None) -> dict:
    img = cv2.imread(image_path)
    if img is None: 
        return None

    h_img, w_img = img.shape[:2]
    max_dim = 1200
    if max(h_img, w_img) > max_dim:
        scale = max_dim / max(h_img, w_img)
        img = cv2.resize(img, (int(w_img * scale), int(h_img * scale)), interpolation=cv2.INTER_AREA)

    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: 
        return None

    min_area = 50  
    max_area = (h_img * w_img) * 0.98 
    valid_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

    if not valid_contours: 
        return None

    main_cnt = max(valid_contours, key=cv2.contourArea)
    area_val = cv2.contourArea(main_cnt)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [main_cnt], -1, 255, -1)

    # ── Shape / Morphology ────────────────────────────────────
    area      = float(area_val)
    perimeter = float(cv2.arcLength(main_cnt, True))
    rect = cv2.minAreaRect(main_cnt)
    _, (rw, rh), _ = rect
    length = float(max(rw, rh))
    width  = float(min(rw, rh))
    aspect_ratio = length / width if width > 0 else 0.0
    roundness    = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    compactness  = area / (length * width) if (length * width) > 0 else 0.0

    if len(main_cnt) >= 5:
        ellipse = cv2.fitEllipse(main_cnt)
        ea, eb  = ellipse[1]
        eccentricity = float(np.sqrt(1 - (min(ea, eb) / max(ea, eb)) ** 2)) if max(ea, eb) > 0 else 0.0
    else:
        eccentricity = 0.0

    # ── Color Analysis ───────────────────────────────
    seed_pixels = img[mask == 255]
    if len(seed_pixels) == 0: 
        return None

    mean_b, mean_g, mean_r = [float(np.mean(seed_pixels[:, i])) for i in range(3)]
    seed_h = hsv[:, :, 0][mask == 255].astype(float)
    seed_s = hsv[:, :, 1][mask == 255].astype(float)
    seed_v = hsv[:, :, 2][mask == 255].astype(float)

    mean_hue        = float(np.mean(seed_h))
    mean_saturation = float(np.mean(seed_s))
    mean_value      = float(np.mean(seed_v)) 
    color_variance  = float(np.var(seed_pixels))
    
    very_dark_pixels = np.sum(seed_v < 50)  
    total_pixels = len(seed_v)
    dark_spots_ratio = float(very_dark_pixels / total_pixels)
    extreme_dark_ratio = float(np.sum(seed_v < 30) / total_pixels)
    yellow_ratio = float(np.sum((seed_h >= 15) & (seed_h <= 35)) / len(seed_h))
    brown_mask = ((seed_h >= 5) & (seed_h <= 20)) & (seed_v < 100)
    brown_ratio = float(np.sum(brown_mask) / total_pixels)

    # ── Texture (GLCM) ────────────────────────────────────────
    x_b, y_b, bw, bh = cv2.boundingRect(main_cnt)
    roi_gray = gray[y_b:y_b + bh, x_b:x_b + bw]
    roi_mask = mask[y_b:y_b + bh, x_b:x_b + bw]
    
    if roi_gray.size == 0: 
        roi_gray = gray
        roi_mask = mask

    roi_gray_masked = roi_gray.copy()
    roi_gray_masked[roi_mask == 0] = 0

    roi_8_glcm = roi_gray_masked.copy()
    if roi_8_glcm.shape[0] > 300 or roi_8_glcm.shape[1] > 300:
        scale_g = 300 / max(roi_8_glcm.shape)
        roi_8_glcm = cv2.resize(roi_8_glcm, 
                                (int(roi_8_glcm.shape[1] * scale_g), 
                                 int(roi_8_glcm.shape[0] * scale_g)), 
                                interpolation=cv2.INTER_AREA)

    glcm = graycomatrix(roi_8_glcm, distances=[1], angles=[0, np.pi/4, np.pi/2], 
                        levels=256, symmetric=True, normed=True)
    texture_contrast    = float(np.mean(graycoprops(glcm, 'contrast')))
    texture_homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
    texture_energy      = float(np.mean(graycoprops(glcm, 'energy')))
    texture_correlation = float(np.mean(graycoprops(glcm, 'correlation')))

    flat = roi_8_glcm[roi_8_glcm > 0].flatten()
    if len(flat) > 0:
        hist, _ = np.histogram(flat, bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        texture_entropy = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0
    else:
        texture_entropy = 0.0

    # ── Damage Detection ─────────────────────────────
    roi_for_cracks = roi_gray_masked.copy()
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    tophat = cv2.morphologyEx(roi_for_cracks, cv2.MORPH_TOPHAT, kernel_small)
    blackhat = cv2.morphologyEx(roi_for_cracks, cv2.MORPH_BLACKHAT, kernel_small)
    enhanced = cv2.add(roi_for_cracks, tophat)
    enhanced = cv2.subtract(enhanced, blackhat)
    
    blur_roi = cv2.GaussianBlur(enhanced, (3, 3), 0)
    edges = cv2.Canny(blur_roi, 30, 90)
    edges = cv2.bitwise_and(edges, roi_mask)
    edges = cv2.dilate(edges, kernel_small, iterations=1)
    
    crack_cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crack_count = 0
    total_crack_length = 0
    for c in crack_cnts:
        length_crack = cv2.arcLength(c, False)
        if length_crack > 20: 
            crack_count += 1
            total_crack_length += length_crack
    
    _, hole_thresh1 = cv2.threshold(roi_for_cracks, 40, 255, cv2.THRESH_BINARY_INV)
    hole_thresh2 = cv2.adaptiveThreshold(roi_for_cracks, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 21, 5)
    hole_thresh = cv2.bitwise_or(hole_thresh1, hole_thresh2)
    hole_thresh = cv2.bitwise_and(hole_thresh, roi_mask)
    hole_thresh = cv2.morphologyEx(hole_thresh, cv2.MORPH_OPEN, kernel_small, iterations=2)
    hole_thresh = cv2.morphologyEx(hole_thresh, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    
    hole_cnts, _ = cv2.findContours(hole_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hole_count = 0
    total_hole_area = 0
    for c in hole_cnts:
        hole_area = cv2.contourArea(c)
        if 25 < hole_area < (area * 0.20):
            hole_count += 1
            total_hole_area += hole_area
            
    hole_area_ratio = float(total_hole_area / area) if area > 0 else 0.0
    
    hull = cv2.convexHull(main_cnt)
    hull_area = float(cv2.contourArea(hull))
    broken_edge_ratio = float(1 - area / hull_area) if hull_area > 0 else 0.0
    
    hull_perimeter = cv2.arcLength(hull, True)
    edge_irregularity = float((perimeter - hull_perimeter) / hull_perimeter) if hull_perimeter > 0 else 0.0

    M = cv2.moments(main_cnt)
    if M["m00"] > 0:
        cx_m = M["m10"] / M["m00"]
        cy_m = M["m01"] / M["m00"]
        dists = [np.hypot(pt[0][0] - cx_m, pt[0][1] - cy_m) for pt in main_cnt]
        wrinkle_index = float(np.std(dists))
    else:
        wrinkle_index = 0.0

    hull_indices = cv2.convexHull(main_cnt, returnPoints=False)
    max_defect_depth = 0.0
    defect_count = 0
    
    if hull_indices is not None and len(hull_indices) > 3 and len(main_cnt) > 3:
        try:
            defects = cv2.convexityDefects(main_cnt, hull_indices)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    depth = d / 256.0
                    if depth > max_defect_depth:
                        max_defect_depth = depth
                    if depth > width * 0.05:
                        defect_count += 1
        except Exception:
            pass
            
    chip_ratio = float(max_defect_depth / width) if width > 0 else 0.0
    
    severe_damage_ratio = (
        dark_spots_ratio * 0.4 +
        hole_area_ratio * 0.3 +
        broken_edge_ratio * 0.2 +
        chip_ratio * 0.1
    )
    
    estimated_volume = float((4 / 3) * np.pi * (length / 2) * (width / 2) ** 2)
    density_estimate = float(estimated_volume / area) if area > 0 else 0.0

    # ── DEBUG VISUALIZATION ───────────────────────────────────
    if debug_out_dir:
        os.makedirs(debug_out_dir, exist_ok=True)
        debug_img = img.copy()
        cv2.drawContours(debug_img, [main_cnt], -1, (0, 255, 0), 2)
        
        debug_holes = img.copy()
        if hole_count > 0:
            cv2.drawContours(debug_holes, hole_cnts, -1, (0, 0, 255), -1)
            debug_img = cv2.addWeighted(debug_img, 0.7, debug_holes, 0.3, 0)
        
        base_name = os.path.basename(image_path)
        cv2.imwrite(os.path.join(debug_out_dir, f"analyzed_{base_name}"), debug_img)
        cv2.imwrite(os.path.join(debug_out_dir, f"cracks_{base_name}"), edges)
        cv2.imwrite(os.path.join(debug_out_dir, f"holes_{base_name}"), hole_thresh)

    # ── COMPILE FEATURES ──────────────────────────────────────
    feat = {
        "area":                round(area, 2),
        "perimeter":           round(perimeter, 2),
        "length":              round(length, 2),
        "width":               round(width, 2),
        "aspect_ratio":        round(aspect_ratio, 4),
        "roundness":           round(roundness, 4),
        "compactness":         round(compactness, 4),
        "eccentricity":        round(eccentricity, 4),
        "mean_red":            round(mean_r, 2),
        "mean_green":          round(mean_g, 2),
        "mean_blue":           round(mean_b, 2),
        "mean_hue":            round(mean_hue, 2),
        "mean_saturation":     round(mean_saturation, 2),
        "mean_value":          round(mean_value, 2),
        "color_variance":      round(color_variance, 2),
        "dark_spots_ratio":    round(dark_spots_ratio, 4),
        "extreme_dark_ratio":  round(extreme_dark_ratio, 4),
        "yellow_ratio":        round(yellow_ratio, 4),
        "brown_ratio":         round(brown_ratio, 4),
        "texture_contrast":    round(texture_contrast, 4),
        "texture_entropy":     round(texture_entropy, 4),
        "texture_homogeneity": round(texture_homogeneity, 4),
        "texture_energy":      round(texture_energy, 4),
        "texture_correlation": round(texture_correlation, 4),
        "crack_count":         crack_count,
        "total_crack_length":  round(total_crack_length, 2),
        "hole_count":          hole_count,
        "hole_area_ratio":     round(hole_area_ratio, 4),
        "broken_edge_ratio":   round(broken_edge_ratio, 4),
        "edge_irregularity":   round(edge_irregularity, 4),
        "wrinkle_index":       round(wrinkle_index, 4),
        "chip_ratio":          round(chip_ratio, 4),
        "defect_count":        defect_count,
        "severe_damage_ratio": round(severe_damage_ratio, 4),
        "estimated_volume":    round(estimated_volume, 2),
        "density_estimate":    round(density_estimate, 4),
    }

    return feat


# ══════════════════════════════════════════════════════════════
# PROCESS IMAGE FOLDERS
# ══════════════════════════════════════════════════════════════

SUPPORTED_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif")

def process_image_folder(folder: str, start_id: int = 1, debug_dir: str = None) -> list:
    files = []
    for ext in SUPPORTED_EXTS:
        files.extend(glob.glob(os.path.join(folder, ext)))
        files.extend(glob.glob(os.path.join(folder, ext.upper())))
    files = sorted(set(files))

    if not files:
        print(f"  No images found in: {folder}")
        return []

    print(f"\n  Processing Folder: {os.path.basename(folder)}")
    print(f"  Found {len(files)} image(s). Extracting features...\n")
    
    records = []
    for i, fp in enumerate(files, 0):
        feat = extract_features(fp, debug_out_dir=debug_dir)

        if feat:
            ordered_feat = {
                "seed_id": f"MOONG_{start_id + i:03d}",
                "folder": os.path.basename(folder),
                "filename": os.path.basename(fp)
            }
            ordered_feat.update(feat)
            
            records.append(ordered_feat)
            print(f"   {start_id + i:3d}. {os.path.basename(fp):45s}  EXTRACTED")
        else:
            print(f"   {start_id + i:3d}. {os.path.basename(fp):45s}  SKIPPED (seed not detected)")

    return records


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  MOONG DAL SEED FEATURE EXTRACTION  —  EXPORT TO CSV")
    print("=" * 70)

    
    IMAGE_FOLDERS = [
        r"D:\ML_Ai\moong_images",
        r"D:\ML_Ai\Hp",
        r"D:\ML_Ai\Vp"
    ]
    

    OUTPUT_PATH  = "moong_dal_features_only.csv"  
    DEBUG_FOLDER = "moong_contours_output"                 

    all_records = []
    current_id = 1
    
    for folder in IMAGE_FOLDERS:
        if os.path.isdir(folder):
            records = process_image_folder(folder, start_id=current_id, debug_dir=DEBUG_FOLDER)
            all_records.extend(records)
            current_id += len(records)
        else:
            print(f"\n  ⚠ Directory not found, skipping: '{folder}'")

    if all_records:
        print("\n  Saving features to CSV...")
        df = pd.DataFrame(all_records)
       
        if os.path.dirname(OUTPUT_PATH):
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            
        df.to_csv(OUTPUT_PATH, index=False)

        print(f"\n  ┌─ EXTRACTION RESULTS ──────────────────────┐")
        print(f"  │ Total seeds processed : {len(all_records):<4d}              │")
        print(f"  │ File Saved as         : {os.path.basename(OUTPUT_PATH):<17s} │")
        print(f"  └───────────────────────────────────────────┘")
        
        print(f"\n  ✓ Successfully exported {len(all_records)} rows to {OUTPUT_PATH}")
        print()
    else:
        print("\n  ⚠ No seeds were processed. Please check your folder paths.")