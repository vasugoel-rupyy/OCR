import cv2
import sys
import os
import argparse
from pathlib import Path
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ocr_pipeline.templates.library import TEMPLATE_LIBRARY
from ocr_pipeline.utils import load_image

def visualize_template(image_path: str, template_name: str = None):
    """
    Draw template regions on the image.
    If template_name is provided, uses that specific template.
    Otherwise, tries to match based on Aspect Ratio logic.
    """
    image = load_image(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    h, w = image.shape[:2]
    aspect_ratio = w / h
    print(f"Image Aspect Ratio: {aspect_ratio:.2f}")

    # Find template
    selected_template = None
    
    if template_name:
        for t in TEMPLATE_LIBRARY:
            if t.name == template_name:
                selected_template = t
                break
        if not selected_template:
            print(f"Error: Template '{template_name}' not found.")
            return
    else:
        # Match by AR
        print("Auto-matching template by Aspect Ratio...")
        best_diff = 100
        for t in TEMPLATE_LIBRARY:
            diff = abs(t.width_height_ratio - aspect_ratio)
            if diff < best_diff:
                best_diff = diff
                selected_template = t
        
        if selected_template and best_diff < 0.2:
            print(f"Auto-selected template: {selected_template.name} (Diff: {best_diff:.3f})")
        else:
            print("No suitable template matched by Aspect Ratio.")
            return

    # Draw regions
    vis_image = image.copy()
    
    # Draw Document Boundary (implicit full image)
    cv2.putText(vis_image, f"Template: {selected_template.name}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for region in selected_template.regions:
        x1, y1, x2, y2 = region.to_absolute(w, h)
        
        # Color: Green
        color = (0, 255, 0)
        thickness = 2
        
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
        
        # Label
        label = f"{region.name} ({region.type})"
        cv2.putText(vis_image, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        print(f"Region '{region.name}': ({x1}, {y1}) -> ({x2}, {y2}) [Rel: {region.coords}]")

    # Save output
    filename = Path(image_path).name
    output_path = f"debug_calibration_{filename}"
    cv2.imwrite(output_path, vis_image)
    print(f"\nVisualization saved to: {output_path}")
    print("Open this image to verify alignment.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Template Regions for Calibration")
    parser.add_argument("image_path", help="Path to sample image")
    parser.add_argument("--template", help="Specific template name (optional)", default=None)
    
    args = parser.parse_args()
    visualize_template(args.image_path, args.template)
