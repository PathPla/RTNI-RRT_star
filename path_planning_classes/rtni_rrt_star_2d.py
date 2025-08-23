import numpy as np
import threading
import time
import math
from copy import deepcopy

from path_planning_utils.rrt_env import Env
from path_planning_classes.irrt_star_2d import IRRTStar2D
from path_planning_classes.rrt_base_2d import RRTBase2D
from path_planning_classes.rrt_visualizer_2d import NIRRTStarVisualizer
from datasets.point_cloud_mask_utils import get_point_cloud_mask_around_points, \
    generate_rectangle_point_cloud, ellipsoid_point_cloud_sampling


class RTNI_RRT_2D(IRRTStar2D):
    """
    RTNI-RRT*(ct) algorithm specifically designed for paper evaluation
    Separates computation time and execution time, no visualization overhead
    """
    def __init__(
        self,
        x_start,
        x_goal,
        step_len,
        search_radius,
        iter_max,
        env_dict,
        png_wrapper,
        binary_mask,
        clearance,
        pc_n_points,
        pc_over_sample_scale,
        pc_sample_rate,
        pc_update_cost_ratio,
        trajectory_length,
        to_travel,
        robot_max_velocity,
        robot_max_acceleration,
        robot_max_angular_velocity,
    ):
        RRTBase2D.__init__(
            self,
            x_start,
            x_goal,
            step_len,
            search_radius,
            iter_max,
            Env(env_dict),
            clearance,
            "RTNI-RRT* Eval 2D",
        )
        
        self.png_wrapper = png_wrapper
        self.binary_mask = binary_mask
        self.pc_n_points = pc_n_points
        self.pc_over_sample_scale = pc_over_sample_scale
        self.pc_sample_rate = pc_sample_rate
        self.pc_neighbor_radius = self.step_len
        self.visualizer = NIRRTStarVisualizer(self.x_start, self.x_goal, self.env)
        self.pc_update_cost_ratio = pc_update_cost_ratio
        self.path_solutions = []
        self.env_dict = env_dict
        self.visualizer = NIRRTStarVisualizer(self.x_start, self.x_goal, self.env)
        
        # Anytime-related parameters
        self.agent_pos = self.x_start
        self.trajectory_length = trajectory_length
        self.to_travel = to_travel
        self.stop_event = threading.Event()
        self.sampling_thread = None
        self.find_goal = False
        self.total_cost = 0
        self.sampling_iter = 0
        
        # Evaluation-related parameters
        self.robot_params = {
            'max_velocity': robot_max_velocity,        # m/s
            'max_acceleration': robot_max_acceleration,    # m/s²
            'max_angular_velocity': robot_max_angular_velocity # rad/s
        }
        
        # Time statistics
        self.planning_time = 0
        self.execution_time = 0
        self.first_solution_time = 0
        self.total_iterations = 0
        self.path_segments = []  # Record all path segments
        
        # Initialize path point cloud predictions as None (no visualization)
        self.path_point_cloud_pred = None
        self.path_point_cloud_other = None
        
        # Initialize c_max and c_update
        self.c_max = np.inf
        self.c_update = np.inf

    def init_pc(self):
        self.update_point_cloud(
            cmax=np.inf,
            cmin=None,
        )

    def update_point_cloud(
        self,
        cmax,
        cmin,
    ):
        if self.pc_sample_rate == 0:
            self.path_point_cloud_pred = None
            self.visualizer.set_path_point_cloud_pred(self.path_point_cloud_pred)
            return
        if cmax < np.inf:
            max_min_ratio = cmax/cmin
            pc = ellipsoid_point_cloud_sampling(
                self.x_start,
                self.x_goal,
                max_min_ratio,
                self.binary_mask,
                self.pc_n_points,
                n_raw_samples=self.pc_n_points*self.pc_over_sample_scale,
            )  # Sample elliptical region as point cloud
        else:
            pc = generate_rectangle_point_cloud(
                self.binary_mask,
                self.pc_n_points,
                self.pc_over_sample_scale,
            )  # Sample rectangular region as point cloud
        start_mask = get_point_cloud_mask_around_points(
            pc,
            self.x_start[np.newaxis,:],
            self.pc_neighbor_radius,
        ) # (n_points,)
        goal_mask = get_point_cloud_mask_around_points(
            pc,
            self.x_goal[np.newaxis,:],
            self.pc_neighbor_radius,
        ) # (n_points,)
        path_pred, path_score = self.png_wrapper.classify_path_points(
            pc.astype(np.float32),
            start_mask.astype(np.float32),
            goal_mask.astype(np.float32),
        )
        self.path_point_cloud_pred = pc[path_pred.nonzero()[0]] # (<pc_n_points, 2)
        self.visualizer.set_path_point_cloud_pred(self.path_point_cloud_pred)
        self.visualizer.set_path_point_cloud_other(pc[np.nonzero(path_pred==0)[0]])

    def sampling_method_no_viz(self, first_planning=False, stop_event=None):
        """
        Sampling method without visualization
        """
        sampling_start_time = time.time()
        
        # Call parent class sampling method core logic
        theta, start_goal_straightline_dist, x_center, C = self.init()
        c_best = self.c_max
        if first_planning:
            self.init_pc()
            self.c_update = self.c_max
        
        iterations = 0
        for k in range(self.iter_max):
            if stop_event and stop_event.is_set():
                if len(self.path_solutions)>0:
                    print(f"The {self.sampling_iter}th sampling cost {k} iterations")
                    return
                
            if len(self.path_solutions) > 0:
                if first_planning:
                    print(f"The {self.sampling_iter}th sampling cost {k} iterations")
                    break
                c_best, x_best = self.find_best_path_solution()
                self.c_max = c_best
                
            node_rand, self.c_update = self.generate_random_node(c_best, start_goal_straightline_dist, x_center, C, self.c_update) # * nirrt*
            node_nearest, node_nearest_index = self.nearest_neighbor(self.vertices[:self.num_vertices], node_rand)
            node_new = self.new_state(node_nearest, node_rand)
            if not self.utils.is_collision(node_nearest, node_new):
                if np.linalg.norm(node_new-node_nearest)<1e-8:
                    # * do not create a new node if it is actually the same point
                    node_new = node_nearest
                    node_new_index = node_nearest_index
                    curr_node_new_cost = self.cost(node_nearest_index)
                else:
                    if self.num_vertices == (self.iter_max+1):  # add by lx
                        node_new_index = np.where(np.isinf(self.vertices))[0][0]
                    else:
                        node_new_index = self.num_vertices
                        self.num_vertices += 1
                    self.vertices[node_new_index] = node_new
                    self.vertex_parents[node_new_index] = node_nearest_index
                    curr_node_new_cost = self.cost(node_nearest_index)+self.Line(node_nearest, node_new)
                neighbor_indices = self.find_near_neighbors(node_new, node_new_index)
                if len(neighbor_indices)>0:
                    self.choose_parent(node_new, neighbor_indices, node_new_index, curr_node_new_cost)
                    self.rewire(node_new, neighbor_indices, node_new_index)
                if self.InGoalRegion(node_new):
                    self.path_solutions.append(node_new_index)

                self.path = []
                                    
            iterations += 1
            
        self.total_iterations += iterations
        sampling_time = time.time() - sampling_start_time
        self.planning_time += sampling_time

    def generate_random_node(
        self,
        c_curr,
        c_min,
        x_center,
        C,
        c_update,
    ):
        '''
        - outputs
            - node_rand: np (2,)
            - c_update: scalar
        '''
        # * tested that np.inf < alpha*np.inf is False, alpha in (0,1]
        if c_curr < self.pc_update_cost_ratio*c_update:
            self.update_point_cloud(c_curr, c_min)
            c_update = c_curr
        if np.random.random() < self.pc_sample_rate:
            try:
                return self.SamplePointCloud(), c_update
            except:
                if c_curr < np.inf:
                    return self.SampleInformedSubset(
                        c_curr,
                        c_min,
                        x_center,
                        C,
                    ), c_update
                else:
                    return self.SampleFree(), c_update
        else:
            if c_curr < np.inf:
                return self.SampleInformedSubset(
                    c_curr,
                    c_min,
                    x_center,
                    C,
                ), c_update
            else:
                return self.SampleFree(), c_update

    def SamplePointCloud(self):
        return self.path_point_cloud_pred[np.random.randint(0,len(self.path_point_cloud_pred))]

    def calculate_execution_time(self, path_segment):
        """
        Calculate execution time based on real physical parameters
        """
        if len(path_segment) < 2:
            return 0
            
        total_time = 0
        
        for i in range(len(path_segment) - 1):
            # Calculate distance
            distance = np.linalg.norm(
                np.array(path_segment[i+1]) - np.array(path_segment[i])
            )
            
            # Time calculation considering acceleration constraints
            max_vel = self.robot_params['max_velocity']
            max_acc = self.robot_params['max_acceleration']
            
            time_to_max_speed = max_vel / max_acc
            distance_to_max_speed = 0.5 * max_acc * time_to_max_speed**2
            
            if distance <= 2 * distance_to_max_speed:
                # Short distance: only acceleration and deceleration phases
                segment_time = 2 * np.sqrt(distance / max_acc)
            else:
                # Long distance: acceleration-constant speed-deceleration
                constant_speed_distance = distance - 2 * distance_to_max_speed
                segment_time = 2 * time_to_max_speed + constant_speed_distance / max_vel
                
            total_time += segment_time

        print(f"execution_time: {total_time}")
            
        return total_time

    def move_along_path_no_viz(self, path_segment):
        """
        Path movement without visualization (only update position)
        """
        if len(path_segment) == 0:
            print("path_segment is empty")
            return
            
        # Calculate execution time
        execution_time = self.calculate_execution_time(path_segment)
        self.execution_time += execution_time
        
        # Update agent position to the end of path segment
        self.agent_pos = np.array(path_segment[-1])
        
        # Record path segment
        self.path_segments.append(path_segment.copy())
        
        # Check if goal is reached
        if np.linalg.norm(self.agent_pos - self.x_goal) < self.step_len:
            self.find_goal = True

    def run_anytime_eval(self):
        """
        Real-time algorithm specifically designed for evaluation (no visualization)
        """
        print("Starting RTNI-RRT* evaluation...")
        
        # Initialize statistics
        start_time = time.time()
        first_planning = True
        step_count = 0
        
        while not self.find_goal:
            step_start_time = time.time()

            if self.visualizer.path_point_cloud_pred is not None:
                self.visualizer.scatter_pred1.remove()
                del self.visualizer.scatter_pred1
                self.visualizer.scatter_pred2.remove()
                del self.visualizer.scatter_pred2
            
            # Stop current sampling thread
            if self.sampling_thread and self.sampling_thread.is_alive():
                self.stop_event.set()
                self.sampling_thread.join()
                self.sampling_iter += 1
            
            # First planning
            if first_planning:
                print("Executing initial planning...")
                self.sampling_method_no_viz(first_planning=True)
                self.sampling_iter += 1
                first_planning = False
                
                if len(self.path_solutions) == 0:
                    print("Initial planning failed, no path found")
                    break
                    
                print(f"Initial planning completed, solution found, time: {self.first_solution_time:.3f}s")
            
            # Get current optimal path segment
            try:
                current_path = self.get_global_path()
                if len(current_path) == 0:
                    print("Unable to get valid path")
                    break
                    
                path_segment = current_path[:min(self.trajectory_length + 1, len(current_path))]
            except:
                print("Path retrieval failed")
                break
            
            step_count += 1
            print(f"Step {step_count}: Executing path segment, length: {len(path_segment)}")

            # Prune tree and set new root node
            self.prune_tree_and_set_new_root(path_segment)

            self.path_solutions = []
            self.visualizer.x_start = self.x_start
            self.path_point_cloud_pred = None
            
            # Start background sampling (if not reached goal)
            if not self.find_goal:
                self.stop_event.clear()
                self.sampling_thread = threading.Thread(
                    target=self.sampling_method_no_viz, 
                    kwargs={"stop_event": self.stop_event}
                )
                self.sampling_thread.start()
            
            # Execute path segment (no visualization)
            self.move_along_path_no_viz(path_segment)
            
            # Step completed
            step_time = time.time() - step_start_time
            print(f"Step {step_count} completed, time: {step_time:.3f}s")
            
            # Safety check: avoid infinite loop
            if step_count > 1000:  # Set maximum step limit
                print("Reached maximum step limit, stopping execution")
                break
        
        # Stop the final sampling thread
        if self.sampling_thread and self.sampling_thread.is_alive():
            self.stop_event.set()
            self.sampling_thread.join()
        
        total_time = time.time() - start_time
        
        # Calculate final path length
        final_path_length = 0
        if self.path_segments:
            for segment in self.path_segments:
                for i in range(len(segment) - 1):
                    final_path_length += np.linalg.norm(
                        np.array(segment[i+1]) - np.array(segment[i])
                    )
        
        # Return detailed results
        result = {
            'success': self.find_goal,
            'total_time': total_time,
            'planning_time': self.planning_time,
            'execution_time': self.execution_time,
            'first_solution_time': self.first_solution_time,
            'path_length': final_path_length,
            'total_iterations': self.total_iterations,
            'step_count': step_count,
            'planning_execution_ratio': self.planning_time / max(self.execution_time, 0.001),
            'avg_planning_per_step': self.planning_time / max(step_count, 1),
            'avg_execution_per_step': self.execution_time / max(step_count, 1)
        }
        
        print(f"\n=== RTNI-RRT* Evaluation Results ===")
        print(f"Success: {result['success']}")
        print(f"Total time: {result['total_time']:.3f}s")
        print(f"Planning time: {result['planning_time']:.3f}s")
        print(f"Execution time: {result['execution_time']:.3f}s")
        print(f"First solution time: {result['first_solution_time']:.3f}s")
        print(f"Path length: {result['path_length']:.2f}")
        print(f"Total iterations: {result['total_iterations']}")
        print(f"Step count: {result['step_count']}")
        print(f"Planning/Execution ratio: {result['planning_execution_ratio']:.2f}")
        
        return result


def get_path_planner(
    args,
    problem,
    neural_wrapper,
):
    """
    Get evaluation version of path planner
    """
    return RTNI_RRT_2D(
        problem['x_start'],
        problem['x_goal'],
        args.step_len,
        problem['search_radius'],
        args.iter_max,
        problem['env_dict'],
        neural_wrapper,
        problem['binary_mask'],
        args.clearance,
        args.pc_n_points,
        args.pc_over_sample_scale,
        args.pc_sample_rate,
        args.pc_update_cost_ratio,
        args.trajectory_length,
        args.to_travel,
        args.robot_max_velocity,
        args.robot_max_acceleration,
        args.robot_max_angular_velocity,
    )