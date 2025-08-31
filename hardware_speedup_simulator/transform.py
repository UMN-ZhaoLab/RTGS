#!/usr/bin/env python3
"""
Transform script to convert point_cloud.json to simulator-compatible format
Converts 3D point cloud data to 2D pixel mapping with Gaussian counts
"""

import json
import numpy as np
import argparse
from typing import Dict, List, Tuple

def load_point_cloud(json_file: str) -> List[Dict]:
    """Load point cloud data from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['vertex']

def project_3d_to_2d(points: List[Dict], 
                     width: int = 1752, 
                     height: int = 1160,
                     fov: float = 60.0,
                     camera_distance: float = 10.0) -> Dict[str, List]:
    """
    Project 3D points to 2D pixel coordinates
    
    Args:
        points: List of 3D points with x, y, z coordinates
        width: Output image width
        height: Output image height
        fov: Field of view in degrees
        camera_distance: Distance from camera to scene center
    
    Returns:
        Dictionary mapping "u_v" coordinates to list of Gaussian indices
    """
    pixels = {}
    
    # Convert FOV to radians
    fov_rad = np.radians(fov)
    
    # Calculate focal length
    focal_length = width / (2 * np.tan(fov_rad / 2))
    
    for i, point in enumerate(points):
        x, y, z = point['x'], point['y'], point['z']
        
        # Simple perspective projection
        # Move camera to positive z and project
        z_proj = z + camera_distance
        
        if z_proj > 0:  # Only project points in front of camera
            # Project to 2D
            u = int((x * focal_length / z_proj) + width / 2)
            v = int((y * focal_length / z_proj) + height / 2)
            
            # Clamp to image boundaries
            u = max(0, min(u, width - 1))
            v = max(0, min(v, height - 1))
            
            # Create key in "u_v" format
            key = f"{u}_{v}"
            
            if key not in pixels:
                pixels[key] = []
            pixels[key].append(i)  # Store Gaussian index
    
    return pixels

def save_simulator_format(pixels: Dict[str, List], output_file: str):
    """Save data in simulator-compatible format"""
    data = {"pixels": pixels}
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Saved simulator-compatible format to {output_file}")
    print(f"📊 Total pixels with Gaussians: {len(pixels)}")
    
    # Count total Gaussians
    total_gaussians = sum(len(gaussian_list) for gaussian_list in pixels.values())
    print(f"🎯 Total Gaussians: {total_gaussians}")

def main():
    parser = argparse.ArgumentParser(description='Transform point cloud to simulator format')
    parser.add_argument('input', help='Input point_cloud.json file')
    parser.add_argument('output', help='Output simulator-compatible JSON file')
    parser.add_argument('--width', type=int, default=1752, help='Output image width')
    parser.add_argument('--height', type=int, default=1160, help='Output image height')
    parser.add_argument('--fov', type=float, default=60.0, help='Field of view in degrees')
    parser.add_argument('--camera-distance', type=float, default=10.0, help='Camera distance')
    
    args = parser.parse_args()
    
    print("🔄 Loading point cloud data...")
    points = load_point_cloud(args.input)
    print(f"📈 Loaded {len(points)} 3D points")
    
    print("🎥 Projecting 3D to 2D...")
    pixels = project_3d_to_2d(
        points, 
        args.width, 
        args.height, 
        args.fov, 
        args.camera_distance
    )
    
    print("💾 Saving simulator format...")
    save_simulator_format(pixels, args.output)
    
    print("🎉 Transformation completed successfully!")

if __name__ == "__main__":
    main()
