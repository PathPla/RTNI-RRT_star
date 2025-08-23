#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate RTNI_RRT* datasets for validating RTNI_RRT* effectiveness
"""

import json
import os
import time
import yaml
import random

import cv2
import numpy as np
import gc
import psutil
from os.path import join, exists
from os import makedirs
import matplotlib.pyplot as plt
from path_planning_utils.Astar_with_clearance import generate_start_goal_points, AStar

# Global configuration
SKIP_ASTAR_FOR_LARGE_MAPS = True  # Whether to skip A* validation for large maps
LARGE_MAP_THRESHOLD = 1000  # Large map threshold
MEDIUM_MAP_THRESHOLD = 750   # Medium map threshold to start using optimization strategies
MEMORY_LIMIT_GB = 95.0       # Memory usage limit (GB)


def save_paths_as_txt(path_list, output_dir, dataset_name, env_idx):
    """
    Save path list as txt files
    
    Args:
        path_list: List of paths, each path is a list of coordinate points
        output_dir: Output root directory
        dataset_name: Dataset name (e.g., 'scaled_maps')
        env_idx: Environment index
    """
    # Create path directory
    path_dir = join(output_dir, dataset_name, 'paths')
    makedirs(path_dir, exist_ok=True)
    
    # Save each path as a separate txt file
    for path_idx, path in enumerate(path_list):
        if path is not None and len(path) > 0:
            path_np = np.array(path)
            path_file = join(path_dir, f"{env_idx}_{path_idx}.txt")
            np.savetxt(path_file, path_np, fmt='%d', delimiter=',')


def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024 / 1024  # Return in GB


def memory_cleanup():
    """Force memory cleanup"""
    gc.collect()
    
    
def check_memory_limit(limit_gb=8.0):
    """Check if memory usage exceeds limit"""
    current_memory = get_memory_usage()
    if current_memory > limit_gb:
        print(f"⚠️  Memory usage too high: {current_memory:.1f}GB > {limit_gb:.1f}GB, cleaning...")
        memory_cleanup()
        return True
    return False


def simple_line_collision_check(binary_env, start, goal, clearance=3):
    """
    Simple line connectivity check to avoid A* memory consumption for large maps
    Uses Bresenham line algorithm to check for obstacles along the path
    """
    x0, y0 = start
    x1, y1 = goal
    
    # Bresenham line algorithm
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    
    # Check if points on path have obstacles
    height, width = binary_env.shape
    for x, y in points:
        # Check boundaries
        if (x - clearance < 0 or x + clearance >= width or 
            y - clearance < 0 or y + clearance >= height):
            return False
        
        # Check if there are obstacles within clearance range around this point
        if np.any(binary_env[y-clearance:y+clearance+1, x-clearance:x+clearance+1] == 0):
            return False
    
    return True


def safe_astar_with_timeout(binary_env, s_start, s_goal, clearance=3, timeout_seconds=30):
    """
    Safe A* path planning with timeout and memory limits
    """
    start_time = time.time()
    
    try:
        # Check memory usage
        if check_memory_limit(6000):  # 6GB limit
            return None, -1
            
        astar = AStar(s_start, s_goal, binary_env, clearance, "euclidean")
        
        # Modify A* search to support timeout
        astar.timeout = timeout_seconds
        astar.start_time = start_time
        
        path, visited = astar.searching()
        path = astar.get_path_from_start_to_goal(path)
        path_success = astar.check_success(path)
        exec_time = time.time() - start_time
        
        if path_success:
            return path, exec_time
        else:
            return None, exec_time
            
    except Exception as e:
        print(f"A* algorithm error: {e}")
        return None, time.time() - start_time
    finally:
        # Force memory cleanup
        memory_cleanup()


def generate_env(
    img_height,
    img_width,
    rectangle_width_range,
    circle_radius_range,
    num_rectangles_range,
    num_circles_range,
):
    """
    Generate environment
    """
    env_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    env_dims = (img_height, img_width)
    num_rectangles = random.randint(num_rectangles_range[0], num_rectangles_range[1])
    num_circles = random.randint(num_circles_range[0], num_circles_range[1])
    rectangle_obstacles = []
    circle_obstacles = []
    
    # Draw random black rectangles
    for i in range(num_rectangles):
        x = random.randint(0, img_width - 50)  # Leave boundary
        y = random.randint(0, img_height - 50)
        w = random.randint(rectangle_width_range[0], rectangle_width_range[1])
        h = random.randint(rectangle_width_range[0], rectangle_width_range[1])
        # Ensure within boundaries
        w = min(w, img_width - x)
        h = min(h, img_height - y)
        cv2.rectangle(env_img, (x, y), (x + w, y + h), (0, 0, 0), -1)
        rectangle_obstacles.append([x, y, w, h])
    
    # Draw random circular obstacles
    for i in range(num_circles):
        r = random.randint(circle_radius_range[0], circle_radius_range[1])
        x = random.randint(r, img_width - r)
        y = random.randint(r, img_height - r)
        cv2.circle(env_img, (x, y), r, (0, 0, 0), -1)
        circle_obstacles.append([x, y, r])
    
    # Create binary environment mask
    binary_env = np.zeros(env_dims).astype(int)
    binary_env[env_img[:, :, 0] != 0] = 1
    
    return env_img, binary_env, env_dims, rectangle_obstacles, circle_obstacles


def generate_astar_path(
    binary_env,
    s_start,
    s_goal,
    clearance=3,
):
    """
    Generate A* path (intelligent memory management version)
    Uses different strategies based on map size and memory usage:
    - Ultra-large maps (>=1000): Skip A*, use estimated path
    - Medium maps (>=750): Use memory optimization strategies (downsampling, etc.)
    - Small maps (<750): Normal A* algorithm
    """
    map_size = max(binary_env.shape)
    current_memory = get_memory_usage()
    
    # Intelligent strategy selection
    if SKIP_ASTAR_FOR_LARGE_MAPS and map_size >= LARGE_MAP_THRESHOLD:
        # Ultra-large maps: Skip A*, use estimated path
        print(f"  Skipping A* validation (map size: {map_size}x{map_size}, memory limit)")
        return _generate_estimated_path(binary_env, s_start, s_goal, clearance)
        
    elif map_size >= MEDIUM_MAP_THRESHOLD or current_memory > MEMORY_LIMIT_GB * 0.7:
        # Medium maps or high memory usage: Use optimization strategies
        print(f"  Using memory-optimized A* (map size: {map_size}x{map_size}, memory: {current_memory:.1f}GB)")
        return _generate_optimized_astar_path(binary_env, s_start, s_goal, clearance)
        
    else:
        # Small maps: Normal A*
        timeout = 15 if map_size < 500 else 30
        return safe_astar_with_timeout(binary_env, s_start, s_goal, clearance, timeout)


def _generate_estimated_path(binary_env, s_start, s_goal, clearance):
    """Generate estimated path for ultra-large maps"""
    # Simple line connectivity check
    if simple_line_collision_check(binary_env, s_start, s_goal, clearance):
        # Estimate path length (Euclidean distance * 1.2 as approximation)
        estimated_path_length = np.sqrt((s_goal[0] - s_start[0])**2 + (s_goal[1] - s_start[1])**2) * 1.2
        estimated_time = estimated_path_length / 100.0  # Assume average speed
        return [(s_start[0], s_start[1]), (s_goal[0], s_goal[1])], estimated_time
    else:
        return None, 0.0


def _generate_optimized_astar_path(binary_env, s_start, s_goal, clearance):
    """Use memory-optimized A* strategy for medium maps"""
    try:
        # Strategy 1: Downsampling strategy - find path on low-resolution map first
        scale_factor = 2 if binary_env.shape[0] > 750 else 1
        
        if scale_factor > 1:
            print(f"    Using downsampling strategy (scale_factor={scale_factor})")
            # Downsample map
            small_env = binary_env[::scale_factor, ::scale_factor]
            small_start = (s_start[0] // scale_factor, s_start[1] // scale_factor)
            small_goal = (s_goal[0] // scale_factor, s_goal[1] // scale_factor)
            
            # Ensure start and goal points are within downsampled map range
            small_start = (min(small_start[0], small_env.shape[1]-1), min(small_start[1], small_env.shape[0]-1))
            small_goal = (min(small_goal[0], small_env.shape[1]-1), min(small_goal[1], small_env.shape[0]-1))
            
            # Run A* on small map
            path_small, exec_time = safe_astar_with_timeout(small_env, small_start, small_goal, clearance//scale_factor, 20)
            
            if path_small is not None and len(path_small) > 1:
                # Map path back to original size and smooth path
                path_full = []
                for x, y in path_small:
                    # Map back to original size, add some random offset to avoid being too straight
                    full_x = int(x * scale_factor + np.random.randint(-scale_factor//2, scale_factor//2 + 1))
                    full_y = int(y * scale_factor + np.random.randint(-scale_factor//2, scale_factor//2 + 1))
                    # Ensure within boundaries
                    full_x = max(0, min(full_x, binary_env.shape[1]-1))
                    full_y = max(0, min(full_y, binary_env.shape[0]-1))
                    path_full.append((full_x, full_y))
                
                return path_full, exec_time * (scale_factor ** 1.5)  # Adjust execution time estimate
            else:
                print("    Downsampling A* failed, trying direct connectivity check")
                return _generate_estimated_path(binary_env, s_start, s_goal, clearance)
        else:
            # Strategy 2: Increase timeout but limit search range
            print("    Using extended timeout A*")
            return safe_astar_with_timeout(binary_env, s_start, s_goal, clearance, 60)
            
    except Exception as e:
        print(f"    Optimized A* failed, using estimated path: {e}")
        return _generate_estimated_path(binary_env, s_start, s_goal, clearance)


def generate_scaled_maps(map_sizes=[224, 500, 750, 1000, 1500], num_per_size=100, 
                        output_dir='data', resume=True, save_images=False, 
                        skip_astar_for_large_maps=True):
    """
    Generate Multi-scale Environments test cases
    Supports checkpoint resume
    """
    if resume:
        # Try to load existing configurations
        existing_configs = load_existing_configs(output_dir, 'scaled_maps')
        all_env_configs = existing_configs.copy()
        
        # Count generated environments
        generated_counts = {}
        for config in existing_configs:
            map_size = config['map_size']
            generated_counts[map_size] = generated_counts.get(map_size, 0) + 1
        
        print("Already generated environments statistics:")
        for map_size in map_sizes:
            count = generated_counts.get(map_size, 0)
            print(f"{map_size}x{map_size}: {count}/{num_per_size}")
    else:
        all_env_configs = []
        generated_counts = {}
    
    for map_size in map_sizes:
        # Check if this size is completed
        current_count = generated_counts.get(map_size, 0)
        if current_count >= num_per_size:
            print(f"{map_size}x{map_size} completed, skipping")
            continue
            
        print(f"Generating {map_size}x{map_size} maps, need {num_per_size - current_count} more")
        
        # Adjust obstacle parameters based on map size
        scale_factor = map_size / 224
        rectangle_width_range = [int(16 * scale_factor), int(24 * scale_factor)]
        circle_radius_range = [int(16 * scale_factor), int(24 * scale_factor)]
        num_rectangles_range = [int(8 * scale_factor), int(12 * scale_factor)]
        num_circles_range = [int(8 * scale_factor), int(12 * scale_factor)]
        path_clearance = 3
        start_goal_dim_distance_limit = int(50 * scale_factor)
        
        env_count = current_count  # Start from existing count
        invalid_env_count = 0
        total_env_count = 0
        
        while env_count < num_per_size:
            total_env_count += 1
            
            # Memory check and cleanup
            current_memory = get_memory_usage()
            if total_env_count % 5 == 0:
                print(f"  Current memory usage: {current_memory:.1f}GB")
                if current_memory > 4.0:  # 4GB threshold
                    print("  Performing memory cleanup...")
                    memory_cleanup()
            
            # Generate environment
            try:
                env_img, binary_env, env_dims, rectangle_obstacles, circle_obstacles = generate_env(
                    map_size,  # img_height
                    map_size,  # img_width
                    rectangle_width_range,
                    circle_radius_range,
                    num_rectangles_range,
                    num_circles_range,
                )
            except MemoryError:
                print(f"❌ Memory insufficient when generating environment, skipping...")
                invalid_env_count += 1
                continue
            
            valid_env = True
            num_samples_per_env = 4  # Generate 4 start-goal pairs per environment
            s_start_list, s_goal_list, path_list, exec_time_list = [], [], [], []
            
            for _ in range(num_samples_per_env):
                # Generate start and goal points (avoiding obstacles)
                s_start, s_goal = generate_start_goal_points(
                    binary_env,
                    clearance=path_clearance,
                    distance_lower_limit=start_goal_dim_distance_limit,
                    max_attempt_count=100,
                )
                
                if s_start is None:
                    valid_env = False
                    break
                
                # Generate A* path to validate connectivity
                try:
                    path, exec_time = generate_astar_path(
                        binary_env,
                        s_start,
                        s_goal,
                        clearance=path_clearance,
                    )
                    
                    if path is None or exec_time < 0:  # Timeout or failure
                        valid_env = False
                        break
                except Exception as e:
                    print(f"  A* path generation exception: {e}")
                    valid_env = False
                    break
                
                s_start_list.append(s_start)
                s_goal_list.append(s_goal)
                path_list.append(path)
                exec_time_list.append(exec_time)
            
            if not valid_env:
                invalid_env_count += 1
                if invalid_env_count % 10 == 0:
                    print(f"Invalid environments: {invalid_env_count}/{total_env_count}")
                
                # If too many consecutive failures, adjust parameters
                if invalid_env_count > 50 and invalid_env_count > total_env_count * 0.8:
                    print(f"⚠️  Invalid environment ratio too high ({invalid_env_count}/{total_env_count}), adjusting parameters...")
                    # Reduce obstacle density
                    num_rectangles_range = [max(1, num_rectangles_range[0]-2), max(2, num_rectangles_range[1]-2)]
                    num_circles_range = [max(1, num_circles_range[0]-2), max(2, num_circles_range[1]-2)]
                    # Increase start-goal distance tolerance
                    start_goal_dim_distance_limit = max(20, start_goal_dim_distance_limit - 10)
                    print(f"  New parameters: rectangles({num_rectangles_range}), circles({num_circles_range}), distance({start_goal_dim_distance_limit})")
                
                continue
            
            # Create environment dictionary (following original format)
            env_dict = {
                'env_dims': env_dims,
                'rectangle_obstacles': rectangle_obstacles,
                'circle_obstacles': circle_obstacles,
                'start': s_start_list,
                'goal': s_goal_list,
                'exec_time': exec_time_list,
                'paths': path_list,  # Add path information
                'map_size': map_size,
                'scale_factor': scale_factor,
                'test_id': f'scale_{map_size}_{env_count}'
            }
            
            all_env_configs.append(env_dict)
            env_count += 1
            
            # Periodic save (every 10 environments)
            if env_count % 10 == 0:
                print(f"Generated {env_count}/{num_per_size} {map_size}x{map_size} environments")
                if resume:  # If checkpoint resume is enabled, save periodically
                    save_env_configs_incremental(all_env_configs, output_dir, 'scaled_maps')
                    print("  -> Progress saved")

            # Optional: Save rendered image for this environment (aligned with envs list index)
            if save_images:
                try:
                    save_env_image_incremental(env_dict, output_dir, 'scaled_maps', index=len(all_env_configs)-1)
                except Exception as e:
                    print(f"Failed to save environment image (scaled_maps, idx={len(all_env_configs)-1}): {e}")
    
    # Final save of all generated environment configurations and paths
    if all_env_configs:
        save_env_configs_incremental(all_env_configs, output_dir, 'scaled_maps')
        print(f"Finally saved {len(all_env_configs)} scaled_maps environment configurations")
    
    return all_env_configs


def generate_long_distance_navigation(map_size=1000, distances=[200, 400, 600, 800, 950], 
                                     complexities=[0.1, 0.25, 0.4], num_per_config=50,
                                     output_dir='data', resume=True, save_images=False):
    """
    Generate Distance-Varying Navigation test cases
    Supports checkpoint resume
    """
    if resume:
        # Try to load existing configurations
        existing_configs = load_existing_configs(output_dir, 'long_distance')
        all_env_configs = existing_configs.copy()
        
        # Count generated environments
        generated_counts = {}
        for config in existing_configs:
            key = (config['target_distance'], config['complexity'])
            generated_counts[key] = generated_counts.get(key, 0) + 1
        
        print("Already generated Distance-Varying Navigation environments statistics:")
        for distance in distances:
            for complexity in complexities:
                key = (distance, complexity)
                count = generated_counts.get(key, 0)
                print(f"  Distance {distance}, Complexity {complexity}: {count}/{num_per_config}")
    else:
        all_env_configs = []
        generated_counts = {}
    
    for distance in distances:
        for complexity in complexities:
            # Check if this configuration is completed
            key = (distance, complexity)
            current_count = generated_counts.get(key, 0)
            if current_count >= num_per_config:
                print(f"  Distance {distance}, Complexity {complexity} completed, skipping")
                continue
                
            print(f"Generating distance={distance}, complexity={complexity} cases, need {num_per_config - current_count} more")
            
            # Set obstacle parameters based on complexity
            base_obstacle_count = int(complexity * 20)  # Base obstacle count
            rectangle_width_range = [20, 60]
            circle_radius_range = [15, 35]
            num_rectangles_range = [base_obstacle_count, base_obstacle_count + 5]
            num_circles_range = [base_obstacle_count, base_obstacle_count + 5]
            path_clearance = 10
            
            env_count = current_count  # Start from existing count
            invalid_env_count = 0
            total_env_count = 0
            
            while env_count < num_per_config:
                total_env_count += 1
                
                # Memory check
                if total_env_count % 5 == 0:
                    current_memory = get_memory_usage()
                    print(f"  Long distance navigation memory: {current_memory:.1f}GB")
                    if current_memory > 4.0:
                        memory_cleanup()
                
                # Generate environment
                try:
                    env_img, binary_env, env_dims, rectangle_obstacles, circle_obstacles = generate_env(
                        map_size,
                        map_size,
                        rectangle_width_range,
                        circle_radius_range,
                        num_rectangles_range,
                        num_circles_range,
                    )
                except MemoryError:
                    print("❌ Long distance navigation environment generation memory insufficient")
                    invalid_env_count += 1
                    continue
                
                valid_env = True
                s_start_list, s_goal_list, path_list, exec_time_list = [], [], [], []
                
                # Generate one start-goal pair with specified distance for each environment
                attempts = 0
                max_attempts = 100
                
                while attempts < max_attempts:
                    # Randomly generate start point
                    s_start, _ = generate_start_goal_points(
                        binary_env,
                        clearance=path_clearance,
                        distance_lower_limit=10,  # Temporary small value
                        max_attempt_count=50,
                    )
                    
                    if s_start is None:
                        attempts += 1
                        continue
                    
                    # Generate goal point within specified distance range
                    found_goal = False
                    for angle_attempt in range(36):  # Try different angles
                        angle = angle_attempt * 10 * np.pi / 180  # Every 10 degrees
                        x_goal = int(s_start[0] + distance * np.cos(angle))
                        y_goal = int(s_start[1] + distance * np.sin(angle))
                        
                        # Check if goal point is valid
                        if (path_clearance <= x_goal < map_size - path_clearance and 
                            path_clearance <= y_goal < map_size - path_clearance):
                            # Check if area around goal point is obstacle-free
                            if not np.any(binary_env[y_goal-path_clearance:y_goal+path_clearance+1,
                                                    x_goal-path_clearance:x_goal+path_clearance+1] == 0):
                                s_goal = (x_goal, y_goal)
                                actual_distance = np.sqrt((x_goal - s_start[0])**2 + (y_goal - s_start[1])**2)
                                
                                # Check if distance meets requirement (allow ±10% error)
                                if abs(actual_distance - distance) <= distance * 0.1:
                                    # Validate path connectivity
                                    try:
                                        path, exec_time = generate_astar_path(
                                            binary_env,
                                            s_start,
                                            s_goal,
                                            clearance=path_clearance,
                                        )
                                        
                                        if path is not None and exec_time >= 0:
                                            s_start_list.append(s_start)
                                            s_goal_list.append(s_goal)
                                            path_list.append(path)
                                            exec_time_list.append(exec_time)
                                            found_goal = True
                                            break
                                    except Exception as e:
                                        print(f"  Long distance navigation A* exception: {e}")
                                        continue
                    
                    if found_goal:
                        break
                    
                    attempts += 1
                
                if not found_goal:
                    invalid_env_count += 1
                    continue
                
                # Create environment dictionary
                env_dict = {
                    'env_dims': env_dims,
                    'rectangle_obstacles': rectangle_obstacles,
                    'circle_obstacles': circle_obstacles,
                    'start': s_start_list,
                    'goal': s_goal_list,
                    'exec_time': exec_time_list,
                    'paths': path_list,  # Add path information
                    'map_size': map_size,
                    'target_distance': distance,
                    'complexity': complexity,
                    'test_id': f'dist_{distance}_complex_{complexity}_{env_count}'
                }
                
                all_env_configs.append(env_dict)
                env_count += 1
                
                # Periodic save
                if env_count % 10 == 0:
                    print(f"Generated {env_count}/{num_per_config} distance={distance}, complexity={complexity} environments")
                    if resume:
                        save_env_configs_incremental(all_env_configs, output_dir, 'long_distance')
                        print("  -> Progress saved")

                # Optional: Save environment image
                if save_images:
                    try:
                        save_env_image_incremental(env_dict, output_dir, 'long_distance', index=len(all_env_configs)-1)
                    except Exception as e:
                        print(f"Failed to save environment image (long_distance, idx={len(all_env_configs)-1}): {e}")
    
    # Final save of all generated environment configurations and paths
    if all_env_configs:
        save_env_configs_incremental(all_env_configs, output_dir, 'long_distance')
        print(f"Finally saved {len(all_env_configs)} long_distance environment configurations")
    
    return all_env_configs


def load_existing_configs(output_dir, dataset_name):
    """
    Load existing environment configurations (for checkpoint resume)
    """
    dataset_dir = join(output_dir, dataset_name)
    json_path = join(dataset_dir, "envs.json")
    
    if exists(json_path):
        try:
            with open(json_path, 'r') as f:
                existing_configs = json.load(f)
            print(f"Found existing configuration file: {len(existing_configs)} environments")
            return existing_configs
        except Exception as e:
            print(f"Failed to load existing configuration: {e}")
            return []
    return []


def save_env_configs_incremental(env_configs, output_dir, dataset_name):
    """
    Incrementally save environment configurations, supports checkpoint resume
    Also saves path files
    """
    # Create directory structure
    dataset_dir = join(output_dir, dataset_name)
    if not exists(dataset_dir):
        makedirs(dataset_dir)
    
    # Save as JSON format
    with open(join(dataset_dir, "envs.json"), "w") as f:
        json.dump(env_configs, f)
    
    # Save all path files
    for env_idx, env_config in enumerate(env_configs):
        if 'paths' in env_config and env_config['paths']:
            try:
                save_paths_as_txt(env_config['paths'], output_dir, dataset_name, env_idx)
            except Exception as e:
                print(f"Failed to save path file ({dataset_name}, env={env_idx}): {e}")
       
    print(f"Saved {len(env_configs)} environment configurations to {dataset_dir}/")


def save_env_configs(env_configs, output_dir, dataset_name):
    """
    Save environment configurations
    """
    return save_env_configs_incremental(env_configs, output_dir, dataset_name)


def visualize_env_config(env_config, save_path=None):
    """
    Visualize environment configuration (following original format)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Rebuild environment image for visualization
    height, width = env_config['env_dims']
    env_img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw rectangular obstacles
    for rect in env_config['rectangle_obstacles']:
        x, y, w, h = rect
        cv2.rectangle(env_img, (x, y), (x + w, y + h), (0, 0, 0), -1)
    
    # Draw circular obstacles
    for circle in env_config['circle_obstacles']:
        x, y, r = circle
        cv2.circle(env_img, (x, y), r, (0, 0, 0), -1)
    
    # Display environment
    ax.imshow(env_img, origin='lower')
    
    # Mark start and goal points
    if 'start' in env_config and 'goal' in env_config:
        for i, (start, goal) in enumerate(zip(env_config['start'], env_config['goal'])):
            ax.plot(start[0], start[1], 'go', markersize=8, alpha=0.8, label='Start' if i == 0 else "")
            ax.plot(goal[0], goal[1], 'ro', markersize=8, alpha=0.8, label='Goal' if i == 0 else "")
    
    # Display information
    title = f"Test Case: {env_config.get('test_id', 'Unknown')}\n"
    title += f"Map Size: {env_config['env_dims'][1]}x{env_config['env_dims'][0]}"
    if 'target_distance' in env_config:
        title += f", Target Distance: {env_config['target_distance']}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def build_env_image(env_config):
    """
    Rebuild environment image from environment configuration (without paths, only obstacle rendering)
    Returns: np.ndarray (H, W, 3) uint8
    """
    height, width = env_config['env_dims']
    env_img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Draw rectangular obstacles
    for rect in env_config['rectangle_obstacles']:
        x, y, w, h = rect
        cv2.rectangle(env_img, (x, y), (x + w, y + h), (0, 0, 0), -1)

    # Draw circular obstacles
    for circle in env_config['circle_obstacles']:
        x, y, r = circle
        cv2.circle(env_img, (x, y), r, (0, 0, 0), -1)

    return env_img


def save_env_image_incremental(env_config, output_dir, dataset_name, index):
    """
    Render environment as PNG image and save to env_imgs/{index}.png.
    Skip if file already exists.
    """
    dataset_dir = join(output_dir, dataset_name)
    img_dir = join(dataset_dir, 'env_imgs')
    if not exists(img_dir):
        makedirs(img_dir)

    img_path = join(img_dir, f"{index}.png")
    if exists(img_path):
        return

    env_img = build_env_image(env_config)
    # OpenCV saves as BGR, currently RGB white background + black obstacles, no visual impact
    cv2.imwrite(img_path, env_img)


def generate_all_visualizations(scaled_configs, long_distance_configs, complex_configs, output_dir):
    """
    Generate selective visualizations for all environment types
    
    Parameters:
    - scaled_configs: List of different sized map configurations
    - long_distance_configs: List of long distance navigation configurations
    - complex_configs: List of complex environment configurations (deprecated, kept for compatibility)
    - output_dir: Output directory
    
    Returns:
    - vis_stats: Dictionary of visualization statistics
    """
    vis_dir = join(output_dir, 'visualizations')
    if exists(vis_dir):
        print("Visualization directory already exists, skipping visualization generation")
        return {}
    
    makedirs(vis_dir)
    print("\n=== Generating visualization examples ===")
    
    vis_stats = {}
    
    # 1. For scaled_maps environments, select 4 environments for visualization
    print("Generating scaled_maps visualizations...")
    scaled_vis_count = 0
    for i in range(0, len(scaled_configs), max(1, len(scaled_configs) // 4)):
        if scaled_vis_count >= 4:
            break
        if i < len(scaled_configs):
            visualize_env_config(scaled_configs[i], 
                               join(vis_dir, f'scaled_example_{i}.png'))
            scaled_vis_count += 1
    vis_stats['scaled_maps'] = scaled_vis_count
    
    # 2. For long_distance environments, select one environment for each of 5 distances and 3 complexities
    print("Generating long_distance visualizations...")
    distances = [200, 400, 600, 800, 950]
    complexities = [0.1, 0.25, 0.4]
    
    # Group long_distance_configs by distance and complexity
    grouped_configs = {}
    for config in long_distance_configs:
        distance = config.get('target_distance')
        complexity = config.get('complexity')
        key = (distance, complexity)
        if key not in grouped_configs:
            grouped_configs[key] = []
        grouped_configs[key].append(config)
    
    # Select one from each combination for visualization
    long_distance_vis_count = 0
    for distance in distances:
        for complexity in complexities:
            key = (distance, complexity)
            if key in grouped_configs and len(grouped_configs[key]) > 0:
                config = grouped_configs[key][0]  # Select the first one
                filename = f'long_distance_d{distance}_c{complexity}.png'
                visualize_env_config(config, join(vis_dir, filename))
                long_distance_vis_count += 1
    vis_stats['long_distance'] = long_distance_vis_count
    
    # 3. complex_environments has been removed, skip
    vis_stats['complex_environments'] = 0
    
    # 4. For block and gap environments, select 1 environment for each of 5 sizes
    print("Generating block and gap environment visualizations...")
    
    # Load block_gap configurations
    block_gap_config_path = join('data/block_gap', 'block_gap_configs.json')
    block_vis_count = 0
    gap_vis_count = 0
    
    if exists(block_gap_config_path):
        with open(block_gap_config_path, 'r') as f:
            block_gap_configs_data = json.load(f)
        
        # Block environments - Select 1 for each of 5 size ratios [2,3,4,5,6]
        block_configs = block_gap_configs_data.get('block', [])
        block_ratios = [2, 3, 4, 5, 6]  # l_dgoal_ratio values
        
        # Group by ratio
        block_grouped = {}
        for config in block_configs:
            ratio = config['img_height'] // config['d_goal']  # Calculate ratio
            if ratio not in block_grouped:
                block_grouped[ratio] = []
            block_grouped[ratio].append(config)
        
        # Select the first one for each ratio
        for ratio in block_ratios:
            if ratio in block_grouped and len(block_grouped[ratio]) > 0:
                config = block_grouped[ratio][0]
                # Rebuild environment dictionary from configuration for visualization
                img_h, img_w = config['img_height'], config['img_width']
                w = config['w']
                d_goal = config['d_goal']
                
                rec_obs_x = img_w // 2 - w // 2
                rec_obs_y = img_h // 2 - w // 2
                rectangle_obstacles = [[rec_obs_x, rec_obs_y, w, w]]
                circle_obstacles = []
                x_start = (img_w // 2 - d_goal // 2, img_h // 2)
                x_goal = (img_w // 2 + d_goal // 2, img_h // 2)
                
                env_dict = {
                    'env_dims': (img_h, img_w),
                    'rectangle_obstacles': rectangle_obstacles,
                    'circle_obstacles': circle_obstacles,
                    'start': [x_start],
                    'goal': [x_goal],
                }
                
                filename = f'block_ratio{ratio}_w{w}.png'
                visualize_env_config(env_dict, join(vis_dir, filename))
                block_vis_count += 1
        
        # Gap environments - Select 1 for each of 5 gap heights [7,6,5,4,3]
        gap_configs = block_gap_configs_data.get('gap', [])
        gap_heights = [7, 6, 5, 4, 3]  # h_g values
        
        # Group by gap height
        gap_grouped = {}
        for config in gap_configs:
            h_g = config['h_g']
            if h_g not in gap_grouped:
                gap_grouped[h_g] = []
            gap_grouped[h_g].append(config)
        
        # Select the first one for each gap height
        for h_g in gap_heights:
            if h_g in gap_grouped and len(gap_grouped[h_g]) > 0:
                config = gap_grouped[h_g][0]
                # Rebuild environment dictionary from configuration for visualization
                img_h, img_w = config['img_height'], config['img_width']
                h = config['h']
                t = config['t']
                y_g = config['y_g']
                d_goal = config['d_goal']
                
                rectangle_obstacles = []
                # Upper wall
                rec_obs_x = img_w // 2 - t // 2
                rec_obs_y = img_h // 2 - h // 2
                rec_obs_w = t
                rec_obs_h = h - h_g - y_g
                rectangle_obstacles.append([rec_obs_x, rec_obs_y, rec_obs_w, rec_obs_h])
                # Lower wall
                rec_obs_x2 = img_w // 2 - t // 2
                rec_obs_y2 = rec_obs_y + (h - y_g)
                rec_obs_w2 = t
                rec_obs_h2 = y_g
                rectangle_obstacles.append([rec_obs_x2, rec_obs_y2, rec_obs_w2, rec_obs_h2])
                
                circle_obstacles = []
                x_start = (img_w // 2 - d_goal // 2, img_h // 2)
                x_goal = (img_w // 2 + d_goal // 2, img_h // 2)
                
                env_dict = {
                    'env_dims': (img_h, img_w),
                    'rectangle_obstacles': rectangle_obstacles,
                    'circle_obstacles': circle_obstacles,
                    'start': [x_start],
                    'goal': [x_goal],
                }
                
                filename = f'gap_height{h_g}_pos{y_g}.png'
                visualize_env_config(env_dict, join(vis_dir, filename))
                gap_vis_count += 1
    
    vis_stats['block_environments'] = block_vis_count
    vis_stats['gap_environments'] = gap_vis_count
    
    # Output statistics
    print(f"Visualization generation completed, generated:")
    print(f"  - scaled_maps: {vis_stats['scaled_maps']}")
    print(f"  - long_distance: {vis_stats['long_distance']}")
    # complex_environments has been removed
    print(f"  - block environments: {vis_stats['block_environments']}")
    print(f"  - gap environments: {vis_stats['gap_environments']}")
    
    return vis_stats


def generate_block_gap_configs(output_root_dir='data/block_gap', save_images=True):
    """
    Generate center block and narrow passage dataset configurations
    Output path: {output_root_dir}/block_gap_configs.json
    block: 5 sizes (d_goal=60, map edge length is 2, 3, 4, 5, 6 times) × 100 block widths = 500
    gap: 5 different gap heights × 100 different passage positions = 500
    """
    os.makedirs(output_root_dir, exist_ok=True)

    block_gap_configs = {"block": [], "gap": []}

    # Generate real time block dataset
    block_dir = join(output_root_dir, 'block')
    os.makedirs(block_dir, exist_ok=True)
    block_envs = []
    block_img_dir = join(block_dir, 'env_imgs')
    os.makedirs(block_img_dir, exist_ok=True)
    # block dataset
    num_envs = 100
    d_goal = 60
    block_widths = np.random.randint(10, 50, num_envs)
    for l_dgoal_ratio in [2, 3, 4, 5, 6]:
        img_height, img_width = d_goal * l_dgoal_ratio, d_goal * l_dgoal_ratio
        for block_width in block_widths:
            img_h, img_w = int(img_height), int(img_width)
            w = int(block_width)
            # Obstacle (center block)
            rec_obs_x = img_w // 2 - w // 2
            rec_obs_y = img_h // 2 - w // 2
            rectangle_obstacles = [[rec_obs_x, rec_obs_y, w, w]]
            circle_obstacles = []
            # Start and goal points
            x_start = (img_w // 2 - d_goal // 2, img_h // 2)
            x_goal = (img_w // 2 + d_goal // 2, img_h // 2)
            best_path_len = block_width + (((d_goal - block_width) // 2) ** 2 + (block_width // 2) ** 2) ** 0.5 + \
                             (((d_goal - block_width) - (d_goal - block_width) // 2) ** 2 + (block_width // 2) ** 2) ** 0.5
            env_dict = {
                'env_dims': (img_h, img_w),
                'rectangle_obstacles': rectangle_obstacles,
                'circle_obstacles': circle_obstacles,
                'start': [x_start],
                'goal': [x_goal],
                'best_path_len': float(best_path_len),
            }
            block_envs.append(env_dict)
            
            block_env_config = {
                'w': int(block_width),
                'd_goal': d_goal,
                'img_height': img_h,
                'img_width': img_w,
                'best_path_len': float(best_path_len),
            }
            block_gap_configs['block'].append(block_env_config)

            if save_images:
                img = build_env_image(env_dict)
                cv2.imwrite(join(block_img_dir, f"{len(block_envs)-1}.png"), img)

    with open(join(block_dir, 'envs.json'), 'w') as f:
        json.dump(block_envs, f)

    # Generate GAP anytime dataset
    gap_dir = join(output_root_dir, 'gap')
    os.makedirs(gap_dir, exist_ok=True)
    gap_envs = []
    gap_img_dir = join(gap_dir, 'env_imgs')
    os.makedirs(gap_img_dir, exist_ok=True)
    # gap dataset
    num_envs = 100
    img_height, img_width = 224, 224
    h = 90
    t = 20
    d_goal = 60
    flank_path_len = t + 2 * (((d_goal - t) / 2) ** 2 + (h / 2) ** 2) ** 0.5
    for h_g in [7, 6, 5, 4, 3]:
        for y_g in np.random.randint(20, 70, num_envs):
            img_h, img_w = int(img_height), int(img_width)
            h = int(h)
            t = int(t)
            h_g = int(h_g)
            y_g = int(y_g)
            d_goal = int(d_goal)
            # Upper and lower wall obstacles
            rectangle_obstacles = []
            # Upper wall
            rec_obs_x = img_w // 2 - t // 2
            rec_obs_y = img_h // 2 - h // 2
            rec_obs_w = t
            rec_obs_h = h - h_g - y_g
            rectangle_obstacles.append([rec_obs_x, rec_obs_y, rec_obs_w, rec_obs_h])
            # Lower wall
            rec_obs_x2 = img_w // 2 - t // 2
            rec_obs_y2 = rec_obs_y + (h - y_g)
            rec_obs_w2 = t
            rec_obs_h2 = y_g
            rectangle_obstacles.append([rec_obs_x2, rec_obs_y2, rec_obs_w2, rec_obs_h2])
            circle_obstacles = []
            # Start and goal points
            x_start = (img_w // 2 - d_goal // 2, img_h // 2)
            x_goal = (img_w // 2 + d_goal // 2, img_h // 2)
            env_dict = {
                'env_dims': (img_h, img_w),
                'rectangle_obstacles': rectangle_obstacles,
                'circle_obstacles': circle_obstacles,
                'start': [x_start],
                'goal': [x_goal],
            }
            gap_envs.append(env_dict)

            gap_env_config = {
                'h': h,
                't': t,
                'h_g': h_g,
                'y_g': y_g,
                'd_goal': d_goal,
                'img_height': img_h,
                'img_width': img_w,
                'flank_path_len': float(flank_path_len),
            }
            block_gap_configs['gap'].append(gap_env_config)

            if save_images:
                img = build_env_image(env_dict)
                cv2.imwrite(join(gap_img_dir, f"{len(gap_envs)-1}.png"), img)

    with open(join(gap_dir, 'envs.json'), 'w') as f:
        json.dump(gap_envs, f)

    with open(join(output_root_dir, "block_gap_configs.json"), 'w') as f:
        json.dump(block_gap_configs, f)
    print(f"Saved center block / narrow passage datasets to: {output_root_dir}")


def main(resume=True, output_dir='data'):
    """
    Generate all test cases
    Supports checkpoint resume
    """
    # Create output directory
    if not exists(output_dir):
        makedirs(output_dir)
    
    print(f"\n=== Starting test case generation (checkpoint resume: {'enabled' if resume else 'disabled'}) ===")
    
    try:
        # 0. Generate center block & narrow passage configurations (one-time, idempotent), and save to data dataset with images
        print("\n=== Generating Center Block & Narrow Passage dataset configurations ===")
        generate_block_gap_configs(output_root_dir='data/block_gap', save_images=True)

        # 1. Generate Multi-scale Environments (multi-scale environment testing)
        print("\n=== Generating Multi-scale Environments test cases ===")
        
        scaled_configs = generate_scaled_maps(
            map_sizes=[224, 500, 750, 1000],
            num_per_size=100,
            output_dir=output_dir,
            resume=resume,
            save_images=True
        )
        
        # 2. Generate Distance-Varying Navigation test cases
        print("\n=== Generating Distance-Varying Navigation test cases ===")
        long_distance_configs = generate_long_distance_navigation(
            map_size=1000,
            distances=[200, 400, 600, 800, 950],
            complexities=[0.1, 0.25, 0.4],
            num_per_config=50,
            output_dir=output_dir,
            resume=resume,
            save_images=True
        )
        
        # Generate visualizations (call separate visualization function)
        generate_all_visualizations(scaled_configs, long_distance_configs, [], output_dir)
        
        # Final statistics
        print("\n=== Test case generation completed ===")
        print(f"Multi-scale Environments: {len(scaled_configs)} environments")
        print(f"Distance-Varying Navigation: {len(long_distance_configs)} environments") 

        
        # Count block_gap environments
        block_gap_total = 0
        block_gap_config_path = join('data/block_gap', 'block_gap_configs.json')
        if exists(block_gap_config_path):
            with open(block_gap_config_path, 'r') as f:
                block_gap_data = json.load(f)
            block_count = len(block_gap_data.get('block', []))
            gap_count = len(block_gap_data.get('gap', []))
            block_gap_total = block_count + gap_count
            print(f"Block environments: {block_count}")
            print(f"Gap environments: {gap_count}")
        
        # Calculate total start-goal pairs
        total_samples = 0
        for configs in [scaled_configs, long_distance_configs]:
            for config in configs:
                total_samples += len(config.get('start', []))
        
        # Add block_gap samples (1 start-goal pair per environment)
        total_samples += block_gap_total
        
        print(f"Total: {len(scaled_configs) + len(long_distance_configs) + block_gap_total} environments")
        print(f"Total: {total_samples} start-goal pairs")
        print(f"\nDatasets saved in: {output_dir}/")
        print(f"Visualizations saved in: {join(output_dir, 'visualizations')}/")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n=== User interrupted ===")
        print("Progress has been saved, you can continue generation by running the script again later")
        return False
    except Exception as e:
        print(f"\n\n=== Error during generation ===")
        print(f"Error: {e}")
        print("Progress has been saved, you can continue generation by running the script again later")
        return False


def parse_args_and_run():
    """Parse command line arguments and run main function"""
    global LARGE_MAP_THRESHOLD, MEDIUM_MAP_THRESHOLD, MEMORY_LIMIT_GB
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate large-scale map test cases with checkpoint resume support')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='Enable checkpoint resume (enabled by default)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                       help='Disable checkpoint resume, start generation from scratch')
    parser.add_argument('--output-dir', default='data',
                       help='Output directory (default: data)')

    parser.add_argument('--large-map-threshold', type=int, default=1000,
                       help='Large map threshold, skip A* validation above this size (default: 1000)')
    parser.add_argument('--medium-map-threshold', type=int, default=750,
                       help='Medium map threshold, use optimization strategies above this size (default: 750)')
    parser.add_argument('--memory-limit', type=float, default=95.0,
                       help='Memory usage limit (GB, default: 95.0)')
    
    args = parser.parse_args()
    
    # Update global configuration
    LARGE_MAP_THRESHOLD = args.large_map_threshold
    MEDIUM_MAP_THRESHOLD = args.medium_map_threshold
    MEMORY_LIMIT_GB = args.memory_limit
    
    # Update output directory
    if args.output_dir != 'data':
        print(f"Using output directory: {args.output_dir}")
    
    # Display configuration
    print(f"💡 Intelligent memory management mode:")
    print(f"   - Ultra-large map threshold: {LARGE_MAP_THRESHOLD} (skip A*)")
    print(f"   - Medium map threshold: {MEDIUM_MAP_THRESHOLD} (optimization strategies)")
    print(f"   - Memory limit: {MEMORY_LIMIT_GB}GB")
    
    success = main(resume=args.resume, output_dir=args.output_dir)
    if success:
        print("\n🎉 All test cases generated successfully!")
    else:
        print("\n⚠️  Generation process was interrupted or encountered errors, but progress has been saved")


if __name__ == "__main__":
    parse_args_and_run()