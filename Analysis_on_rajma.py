"""
Simple Step-by-Step Grain Analysis
Easy to understand and modify for beginners
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# STEP 1: LOAD IMAGE........................................................................
def load_image(image_path):
    """Load and display the image"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Display
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    

    plt.subplot(1, 2, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    plt.show()
    
    return img, img_rgb, gray


# STEP 2: DETECT GRAINS ........................................................................
def detect_grains(gray_image):
    """Detect individual grains using image processing"""
    
    # 2.1: Blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # 2.2: Thresholding to separate grains from background
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    
    # 2.3: Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 2.4: Find contours (grain boundaries)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 2.5: Filter out very small or very large contours (noise)
    min_area = 50  
    max_area = 10000 
    filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    
    # Visualize
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(blurred, cmap='gray')
    plt.title('Blurred Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(binary, cmap='gray')
    plt.title('Binary Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    temp_img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(temp_img, filtered_contours, -1, (0, 0,255 ), 2)
    plt.imshow(temp_img)
    plt.title(f'Detected Grains: {len(filtered_contours)}')
    plt.axis('off')
    plt.show()
    
    return filtered_contours


# STEP 3: ANALYZE COLOR........................................................................
def analyze_color(img_bgr, contours):
    """Analyze the color of each grain"""
    
    grain_colors = []
    color_labels = []
    
    for contour in contours:
        # Create a mask for this grain
        mask = np.zeros(img_bgr.shape[:2], np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Calculate mean color (BGR format)
        mean_bgr = cv2.mean(img_bgr, mask=mask)[:3]
        b, g, r = mean_bgr
        grain_colors.append([b, g, r])
        
        # Classify color
        if r > 150 and g > 150 and b < 100:
            color_labels.append('Yellow')
        elif g > r and g > b and g > 100:
            color_labels.append('Green')
        elif r < 80 and g < 80 and b < 80:
            color_labels.append('Black')
        elif r > 100 and g > 80 and b < 100:
            color_labels.append('Brown')
        else:
            color_labels.append('Other')
    
    # Count colors
    color_counts = pd.Series(color_labels).value_counts()
    
    print("COLOR ANALYSIS:")
    print(color_counts)
    print(f"\nDominant Color: {color_counts.index[0]}")
    
    # Visualize
    plt.figure(figsize=(10, 5))
    color_counts.plot(kind='bar', color=['yellow', 'green', 'black', 'brown', 'gray'])
    plt.title('Color Distribution')
    plt.ylabel('Number of Grains')
    plt.xlabel('Color')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return color_labels, grain_colors


# STEP 4: ANALYZE SHAPE........................................................................
def analyze_shape(contours):
    """Analyze the shape of each grain"""
    
    shapes = []
    circularities = []
    aspect_ratios = []
    
    for contour in contours:
        # Calculate area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Circularity = 4π × area / perimeter²
        # Circularity = 1 for perfect circle, < 1 for other shapes
        
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            circularities.append(circularity)
        else:
            circularities.append(0)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        aspect_ratios.append(aspect_ratio)
 
        if circularity > 0.8:
            shapes.append('Round')
        elif aspect_ratio > 1.5 or aspect_ratio < 0.67:
            shapes.append('Oval')
        else:
            shapes.append('Split')
    
    # Count shapes
    shape_counts = pd.Series(shapes).value_counts()
    
    print("\nSHAPE ANALYSIS:")
    print(shape_counts)
    print(f"\nDominant Shape: {shape_counts.index[0]}")
    print(f"Average Circularity: {np.mean(circularities):.3f}")
    print(f"Average Aspect Ratio: {np.mean(aspect_ratios):.3f}")
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    shape_counts.plot(kind='bar', color='skyblue')
    plt.title('Shape Distribution')
    plt.ylabel('Number of Grains')
    plt.xlabel('Shape Type')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.hist(circularities, bins=20, color='coral', edgecolor='black')
    plt.title('Circularity Distribution')
    plt.xlabel('Circularity (1 = perfect circle)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return shapes, circularities


# STEP 5: ANALYZE SIZE........................................................................
def analyze_size(contours):
    """Analyze the size of grains"""
    
    areas = []
    widths = []
    heights = []
    
    for contour in contours:

        area = cv2.contourArea(contour)
        areas.append(area)
        
        x, y, w, h = cv2.boundingRect(contour)
        widths.append(w)
        heights.append(h)
    
    print("\nSIZE ANALYSIS:")
    print(f"Average Area: {np.mean(areas):.2f} pixels²")
    print(f"Min Area: {np.min(areas):.2f} pixels²")
    print(f"Max Area: {np.max(areas):.2f} pixels²")
    print(f"Standard Deviation: {np.std(areas):.2f} pixels²")
    print(f"\nAverage Width: {np.mean(widths):.2f} pixels")
    print(f"Average Height: {np.mean(heights):.2f} pixels")
    
    # Uniformity (1 = all grains same size, 0 = very different)
    uniformity = 1 - (np.std(areas) / np.mean(areas))
    print(f"Size Uniformity: {uniformity:.3f}")
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(areas, bins=20, color='green', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(areas), color='red', linestyle='--', linewidth=2, label='Mean')
    plt.title('Grain Size Distribution')
    plt.xlabel('Area (pixels²)')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(widths, heights, alpha=0.5, color='purple')
    plt.title('Width vs Height')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    
    plt.tight_layout()
    plt.show()
    
    return areas



# STEP 6: ANALYZE TEXTURE........................................................................
def analyze_texture(gray_image, contours):
    """Analyze texture of grains"""
    
    smoothness_values = []
    std_values = []
    
    for contour in contours:
        # Create mask
        mask = np.zeros(gray_image.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Get pixel values inside this grain
        grain_pixels = gray_image[mask == 255]
        
        if len(grain_pixels) > 0:
            # Standard deviation (higher = rougher texture)
            std = np.std(grain_pixels)
            std_values.append(std)
            
            # Smoothness (0 = rough, 1 = smooth)
            smoothness = 1 - (1 / (1 + std))
            smoothness_values.append(smoothness)
    
    print("\nTEXTURE ANALYSIS:")
    print(f"Average Smoothness: {np.mean(smoothness_values):.3f} (0=rough, 1=smooth)")
    print(f"Average Std Dev: {np.mean(std_values):.2f}")
    
    # Visualize
    plt.figure(figsize=(10, 5))
    plt.hist(smoothness_values, bins=20, color='orange', edgecolor='black')
    plt.title('Grain Smoothness Distribution')
    plt.xlabel('Smoothness (1 = very smooth)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    
    return smoothness_values



# STEP 7: COUNT GRAINS........................................................................
def count_grains(contours):
    """Simple grain counting"""
    total_grains = len(contours)
    print("\nGRAIN COUNT:")
    print(f"Total Grains Detected: {total_grains}")
    return total_grains


# STEP 8: CALCULATE COVERAGE........................................................................
def calculate_coverage(img_shape, contours):
    """Calculate how much area is covered by grains"""
    
    # Total image area
    total_image_area = img_shape[0] * img_shape[1]
    
    # Total grain area
    total_grain_area = sum([cv2.contourArea(cnt) for cnt in contours])
    
    # Coverage percentage
    coverage_percent = (total_grain_area / total_image_area) * 100
    
    print("\nCOVERAGE ANALYSIS:")
    print(f"Image Area: {total_image_area} pixels²")
    print(f"Grain Area: {total_grain_area:.2f} pixels²")
    print(f"Coverage: {coverage_percent:.2f}%")
    
    # Visualize
    plt.figure(figsize=(6, 6))
    plt.pie([total_grain_area, total_image_area - total_grain_area], 
            labels=['Grains', 'Background'],
            autopct='%1.1f%%',
            colors=['wheat', 'lightgray'],
            startangle=90)
    plt.title('Area Coverage')
    plt.show()
    
    return coverage_percent


# STEP 9: CALCULATE DENSITY........................................................................
def calculate_density(img_shape, contours):
    """Calculate grain density"""
    
    total_area = img_shape[0] * img_shape[1]
    grain_count = len(contours)
    
    # Density = grains per 1000 square pixels
    density = (grain_count / total_area) * 1000
    
    # Average spacing between grains
    avg_spacing = total_area / grain_count if grain_count > 0 else 0
    
    print("\nDENSITY ANALYSIS:")
    print(f"Grain Density: {density:.3f} grains per 1000 pixels²")
    print(f"Average Spacing: {avg_spacing:.2f} pixels² per grain")
    
    return density



# STEP 10: CREATE SUMMARY REPORT........................................................................
def create_summary_report(all_results):
    """Create a summary table of all results"""
    
    data = {
        'Metric': [
            'Total Grains',
            'Dominant Color',
            'Dominant Shape',
            'Avg Area (px²)',
            'Avg Circularity',
            'Coverage (%)',
            'Density (grains/1000px²)',
            'Avg Smoothness'
        ],
        'Value': [
            all_results['count'],
            all_results['dominant_color'],
            all_results['dominant_shape'],
            f"{all_results['avg_area']:.2f}",
            f"{all_results['avg_circularity']:.3f}",
            f"{all_results['coverage']:.2f}",
            f"{all_results['density']:.3f}",
            f"{all_results['avg_smoothness']:.3f}"
        ]
    }
    
    df = pd.DataFrame(data)
    print("\n" + "="*60)
    print("COMPLETE GRAIN ANALYSIS SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    return df


# Main........................................................................
if __name__ == "__main__":
    
    # image path.....
    # IMAGE_PATH = "D:\\ML_Ai\\f1\\IMG_20260309_230308.jpg.jpeg" 
    IMAGE_PATH = "d:\\ML_Ai\images\\WhatsApp Image 2026-02-06 at 6.59.51 PM.jpeg"  
    
    print("Starting Grain Analysis...\n")
    
    # Step 1: Load image
    result = load_image(IMAGE_PATH)
    if result is None:
        print("Please update IMAGE_PATH with your grain image")
        exit()
    
    img_bgr, img_rgb, gray = result
    
    # Step 2: Detect grains
    contours = detect_grains(gray)
    
    # Step 3: Analyze color
    color_labels, grain_colors = analyze_color(img_bgr, contours)
    
    # Step 4: Analyze shape
    shapes, circularities = analyze_shape(contours)
    
    # Step 5: Analyze size
    areas = analyze_size(contours)
    
    # Step 6: Analyze texture
    smoothness = analyze_texture(gray, contours)
    
    # Step 7: Count
    grain_count = count_grains(contours)
    
    # Step 8: Coverage
    coverage = calculate_coverage(gray.shape, contours)
    
    # Step 9: Density
    density = calculate_density(gray.shape, contours)
    
    # Step 10: Summary
    all_results = {
        'count': grain_count,
        'dominant_color': pd.Series(color_labels).value_counts().index[0] if color_labels else 'Unknown',
        'dominant_shape': pd.Series(shapes).value_counts().index[0] if shapes else 'Unknown',
        'avg_area': np.mean(areas) if areas else 0,
        'avg_circularity': np.mean(circularities) if circularities else 0,
        'coverage': coverage,
        'density': density,
        'avg_smoothness': np.mean(smoothness) if smoothness else 0
    }
