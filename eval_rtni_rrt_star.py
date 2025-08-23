#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTNI-RRT* Algorithm Evaluation Script
Evaluates different RRT-based algorithms on various environments and generates results for plotting
"""

import os
import sys
import json
import pickle
import time
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from path_planning_classes.rrt_star_2d import RRTStar2D
from path_planning_classes.irrt_star_2d import IRRTStar2D
from path_planning_classes.nrrt_star_png_2d import NRRTStarPNG2D
from path_planning_classes.rtni_rrt_star_2d import RTNI_RRT_2D
from path_planning_classes.rtni_rrt_start_cb_2d import RTNI_RRT_2D_CB
from path_planning_utils.rrt_env import Env
from datasets.point_cloud_mask_utils import get_point_cloud_mask_around_points


class AlgorithmEvaluator:
    """Evaluates different RRT-based algorithms on various environments"""
    
    def __init__(self, args):
        self.args = args
        self.results = {}
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Algorithm mapping
        self.algorithm_classes = {
            'rrt_star': RRTStar2D,
            'irrt_star': IRRTStar2D,
            'nrrt_star': NRRTStarPNG2D,
            'rtni_rrt_star': RTNI_RRT_2D,
            'rtni_rrt_star_cb': RTNI_RRT_2D_CB
        }
        
        # Environment types
        self.env_types = [
            'scaled_maps',
            'long_distance_navigation',
            'block_gap'
        ]
        
        # Create output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_environment_data(self, env_type):
        """Load environment data from the specified type"""
        env_path = Path('data') / env_type
        
        if not env_path.exists():
            print(f"Warning: Environment path {env_path} does not exist")
            return []
            
        # Try to load from different file formats
        config_files = []
        
        # Look for JSON config files
        json_files = list(env_path.glob('*.json'))
        if json_files:
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        configs = json.load(f)
                        if isinstance(configs, dict):
                            # Handle different JSON structures
                            if 'configs' in configs:
                                config_files.extend(configs['configs'])
                            elif 'block' in configs and 'gap' in configs:
                                config_files.extend(configs['block'])
                                config_files.extend(configs['gap'])
                            else:
                                config_files.append(configs)
                        elif isinstance(configs, list):
                            config_files.extend(configs)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
                    continue
        
        # Look for PKL files
        pkl_files = list(env_path.glob('*.pkl'))
        if pkl_files:
            for pkl_file in pkl_files:
                try:
                    with open(pkl_file, 'rb') as f:
                        configs = pickle.load(f)
                        if isinstance(configs, list):
                            config_files.extend(configs)
                        elif isinstance(configs, dict):
                            config_files.append(configs)
                except Exception as e:
                    print(f"Error loading {pkl_file}: {e}")
                    continue
        
        print(f"Loaded {len(config_files)} configurations from {env_type}")
        return config_files
    
    def create_path_planner(self, algorithm_name, problem_config):
        """Create a path planner instance for the specified algorithm"""
        if algorithm_name not in self.algorithm_classes:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
            
        algorithm_class = self.algorithm_classes[algorithm_name]
        
        # Common parameters
        common_params = {
            'x_start': problem_config['x_start'],
            'x_goal': problem_config['x_goal'],
            'step_len': self.args.step_len,
            'search_radius': problem_config.get('search_radius', 20),
            'iter_max': self.args.iter_max,
            'clearance': self.args.clearance
        }
        
        if algorithm_name in ['rrt_star', 'irrt_star']:
            # Basic RRT algorithms
            env = Env(problem_config['env_dict'])
            planner = algorithm_class(**common_params, env=env)
            # Set robot parameters for execution time calculation
            planner.robot_params = {
                'max_velocity': self.args.robot_max_velocity,
                'max_acceleration': self.args.robot_max_acceleration,
                'max_angular_velocity': self.args.robot_max_angular_velocity
            }
            return planner
            
        elif algorithm_name in ['nrrt_star']:
            # Neural RRT algorithms
            planner = algorithm_class(**common_params, env=problem_config['env_dict'])
            # Set robot parameters for execution time calculation
            planner.robot_params = {
                'max_velocity': self.args.robot_max_velocity,
                'max_acceleration': self.args.robot_max_acceleration,
                'max_angular_velocity': self.args.robot_max_angular_velocity
            }
            return planner
            
        elif algorithm_name in ['rtni_rrt_star', 'rtni_rrt_star_cb']:
            # RTNI-RRT algorithms
            # Create a dummy neural wrapper for evaluation
            class DummyNeuralWrapper:
                def __init__(self):
                    self.model = None
                    
                def predict(self, *args, **kwargs):
                    # Return random predictions for evaluation
                    return np.random.random((self.args.pc_n_points, 2))
            
            neural_wrapper = DummyNeuralWrapper()
            
            rtni_params = {
                **common_params,
                'env_dict': problem_config['env_dict'],
                'png_wrapper': neural_wrapper,
                'binary_mask': problem_config.get('binary_mask', np.ones((100, 100))),
                'pc_n_points': self.args.pc_n_points,
                'pc_over_sample_scale': self.args.pc_over_sample_scale,
                'pc_sample_rate': self.args.pc_sample_rate,
                'pc_update_cost_ratio': self.args.pc_update_cost_ratio,
                'trajectory_length': self.args.trajectory_length,
                'to_travel': self.args.to_travel,
                'robot_max_velocity': self.args.robot_max_velocity,
                'robot_max_acceleration': self.args.robot_max_acceleration,
                'robot_max_angular_velocity': self.args.robot_max_angular_velocity
            }
            
            return algorithm_class(**rtni_params)
        
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}")
    
    def evaluate_algorithm_on_environment(self, algorithm_name, env_configs, env_type):
        """Evaluate a specific algorithm on a specific environment type"""
        print(f"\nEvaluating {algorithm_name} on {env_type}...")
        
        results = []
        
        # Limit number of evaluations if specified
        max_evals = min(len(env_configs), self.args.max_evaluations)
        env_configs = env_configs[:max_evals]
        
        for i, config in enumerate(tqdm(env_configs, desc=f"{algorithm_name} on {env_type}")):
            try:
                # Create path planner
                planner = self.create_path_planner(algorithm_name, config)
                
                # Run evaluation
                if hasattr(planner, 'run_anytime_eval'):
                    result = planner.run_anytime_eval()
                else:
                    # Fallback to basic planning for algorithms without anytime evaluation
                    start_time = time.time()
                    planner.planning()
                    end_time = time.time()
                    
                    result = {
                        'success': len(planner.path) > 0,
                        'total_time': end_time - start_time,
                        'planning_time': end_time - start_time,
                        'execution_time': 0.0,
                        'first_solution_time': end_time - start_time,
                        'path_length': planner.get_path_len(planner.path) if planner.path else np.inf,
                        'total_iterations': planner.iter_max,
                        'step_count': 1,
                        'planning_execution_ratio': 1.0,
                        'avg_planning_per_step': end_time - start_time,
                        'avg_execution_per_step': 0.0
                    }
                
                # Add metadata
                result['algorithm'] = algorithm_name
                result['environment'] = env_type
                result['config_index'] = i
                result['env_config'] = config
                
                results.append(result)
                
            except Exception as e:
                print(f"Error evaluating {algorithm_name} on {env_type} config {i}: {e}")
                # Add error result
                results.append({
                    'algorithm': algorithm_name,
                    'environment': env_type,
                    'config_index': i,
                    'env_config': config,
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    def run_evaluation(self):
        """Run the complete evaluation"""
        print("Starting RTNI-RRT* algorithm evaluation...")
        print(f"Algorithms: {list(self.args.algorithms)}")
        print(f"Environments: {list(self.args.environments)}")
        print(f"Output directory: {self.output_dir}")
        
        all_results = {}
        
        for env_type in self.args.environments:
            print(f"\n{'='*50}")
            print(f"Processing environment: {env_type}")
            print(f"{'='*50}")
            
            # Load environment configurations
            env_configs = self.load_environment_data(env_type)
            if not env_configs:
                print(f"No configurations found for {env_type}, skipping...")
                continue
            
            env_results = {}
            
            for algorithm in self.args.algorithms:
                try:
                    # Evaluate algorithm on this environment
                    algorithm_results = self.evaluate_algorithm_on_environment(
                        algorithm, env_configs, env_type
                    )
                    env_results[algorithm] = algorithm_results
                    
                except Exception as e:
                    print(f"Error evaluating {algorithm} on {env_type}: {e}")
                    env_results[algorithm] = []
            
            all_results[env_type] = env_results
        
        # Save results
        self.save_results(all_results)
        
        print(f"\nEvaluation completed! Results saved to {self.output_dir}")
        return all_results
    
    def save_results(self, results):
        """Save evaluation results to files"""
        # Save detailed results
        results_file = self.output_dir / f'eval_results_{self.timestamp}.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Detailed results saved to: {results_file}")
        
        # Save summary statistics
        summary = self.generate_summary_statistics(results)
        summary_file = self.output_dir / f'eval_summary_{self.timestamp}.pkl'
        with open(summary_file, 'wb') as f:
            pickle.dump(summary, f)
        print(f"Summary statistics saved to: {summary_file}")
        
        # Save results in plot.py compatible format
        plot_results = self.convert_to_plot_format(results)
        plot_file = self.output_dir / 'results.pkl'
        with open(plot_file, 'wb') as f:
            pickle.dump(plot_results, f)
        print(f"Plot-compatible results saved to: {plot_file}")
    
    def generate_summary_statistics(self, results):
        """Generate summary statistics from detailed results"""
        summary = {}
        
        for env_type, env_results in results.items():
            summary[env_type] = {}
            
            for algorithm, algorithm_results in env_results.items():
                if not algorithm_results:
                    continue
                
                # Filter out error results
                valid_results = [r for r in algorithm_results if 'error' not in r]
                if not valid_results:
                    continue
                
                # Calculate statistics
                success_rate = np.mean([r['success'] for r in valid_results])
                total_times = [r['total_time'] for r in valid_results if r['success']]
                planning_times = [r['planning_time'] for r in valid_results if r['success']]
                path_lengths = [r['path_length'] for r in valid_results if r['success']]
                
                summary[env_type][algorithm] = {
                    'success_rate': success_rate,
                    'num_evaluations': len(valid_results),
                    'num_successful': sum([r['success'] for r in valid_results]),
                    'avg_total_time': np.mean(total_times) if total_times else np.inf,
                    'avg_planning_time': np.mean(planning_times) if planning_times else np.inf,
                    'avg_path_length': np.mean(path_lengths) if path_lengths else np.inf,
                    'std_total_time': np.std(total_times) if total_times else 0,
                    'std_planning_time': np.std(planning_times) if planning_times else 0,
                    'std_path_length': np.std(path_lengths) if path_lengths else 0
                }
        
        return summary
    
    def convert_to_plot_format(self, results):
        """Convert results to the format expected by plot.py"""
        plot_data = {}
        
        # Map algorithm names to plot.py format
        algorithm_mapping = {
            'rrt_star': 'RRT*',
            'irrt_star': 'IRRT*',
            'nrrt_star': 'NRRT*',
            'rtni_rrt_star': 'RTNI-RRT*(Ct)',
            'rtni_rrt_star_cb': 'RTNI-RRT*(CtCb)'
        }
        
        # Map environment names to plot.py format
        env_mapping = {
            'scaled_maps': 'multi_scale',
            'long_distance_navigation': 'distance_varying',
            'block_gap': 'center_obstacle'
        }
        
        for env_type, env_results in results.items():
            plot_env = env_mapping.get(env_type, env_type)
            plot_data[plot_env] = {}
            
            for algorithm, algorithm_results in env_results.items():
                plot_algorithm = algorithm_mapping.get(algorithm, algorithm)
                plot_data[plot_env][plot_algorithm] = {}
                
                if not algorithm_results:
                    continue
                
                # Filter out error results
                valid_results = [r for r in algorithm_results if 'error' not in r and r['success']]
                if not valid_results:
                    continue
                
                # Extract metrics for plotting
                planning_exec_ratios = [r['planning_execution_ratio'] for r in valid_results]
                total_times = [r['total_time'] for r in valid_results]
                path_lengths = [r['path_length'] for r in valid_results]
                
                plot_data[plot_env][plot_algorithm] = {
                    'planning_exec_ratio': np.array(planning_exec_ratios),
                    'total_time': np.array(total_times),
                    'path_length': np.array(path_lengths),
                    'success_rate': len(valid_results) / len(algorithm_results)
                }
        
        return plot_data


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate RTNI-RRT* algorithms on various environments'
    )
    
    # Algorithm selection
    parser.add_argument(
        '--algorithms', 
        nargs='+', 
        default=['rrt_star', 'irrt_star', 'nrrt_star', 'rtni_rrt_star', 'rtni_rrt_star_cb'],
        choices=['rrt_star', 'irrt_star', 'nrrt_star', 'rtni_rrt_star', 'rtni_rrt_star_cb'],
        help='Algorithms to evaluate'
    )
    
    # Environment selection
    parser.add_argument(
        '--environments', 
        nargs='+', 
        default=['scaled_maps', 'long_distance_navigation', 'block_gap'],
        choices=['scaled_maps', 'long_distance_navigation', 'block_gap'],
        help='Environment types to evaluate on'
    )
    
    # Algorithm parameters
    parser.add_argument('--step_len', type=float, default=20.0, help='Step length for RRT')
    parser.add_argument('--iter_max', type=int, default=5000, help='Maximum iterations')
    parser.add_argument('--clearance', type=int, default=3, help='Clearance for collision checking')
    
    # RTNI-RRT specific parameters
    parser.add_argument('--pc_n_points', type=int, default=1024, help='Number of point cloud points')
    parser.add_argument('--pc_over_sample_scale', type=float, default=1.5, help='Point cloud oversample scale')
    parser.add_argument('--pc_sample_rate', type=float, default=0.8, help='Point cloud sample rate')
    parser.add_argument('--pc_update_cost_ratio', type=float, default=0.1, help='Point cloud update cost ratio')
    parser.add_argument('--trajectory_length', type=int, default=10, help='Trajectory length for anytime planning')
    parser.add_argument('--to_travel', type=float, default=0.8, help='Travel ratio for anytime planning')
    parser.add_argument('--robot_max_velocity', type=float, default=1.0, help='Robot max velocity')
    parser.add_argument('--robot_max_acceleration', type=float, default=2.0, help='Robot max acceleration')
    parser.add_argument('--robot_max_angular_velocity', type=float, default=1.0, help='Robot max angular velocity')
    
    # Evaluation parameters
    parser.add_argument('--max_evaluations', type=int, default=50, help='Maximum evaluations per environment')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='Output directory for results')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Create evaluator and run evaluation
    evaluator = AlgorithmEvaluator(args)
    results = evaluator.run_evaluation()
    
    print("\nEvaluation completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print("You can now use plot.py to generate publication-ready figures")


if __name__ == "__main__":
    main()
