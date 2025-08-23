import numpy as np
import threading
import multiprocessing  
import time  

from path_planning_utils.rrt_env import Env
from path_planning_classes.rrt_base_2d import RRTBase2D
from path_planning_classes.rtni_rrt_star_2d import RTNI_RRT_2D
from path_planning_classes.rrt_visualizer_2d import NIRRTStarVisualizer
from datasets.point_cloud_mask_utils import generate_rectangle_point_cloud, \
    ellipsoid_point_cloud_sampling
from wrapper.pointnet_pointnet2.pointnet2_wrapper_connect_bfs import PNGWrapper


class RTNI_RRT_Star_C_2D(RTNI_RRT_2D):
    """
    RTNI-RRT*(ctcb) algorithm specifically designed for paper evaluation
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
        png_wrapper_connect,
        binary_mask,
        clearance,
        pc_n_points,
        pc_over_sample_scale,
        pc_sample_rate,
        pc_update_cost_ratio,
        connect_max_trial_attempts,
        trajectory_length,  
        to_travel,  
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
            "NIRRT*-PNG(C) 2D",
        )
        self.png_wrapper = png_wrapper_connect
        self.binary_mask = binary_mask
        self.pc_n_points = pc_n_points # * number of points in pc
        self.pc_over_sample_scale = pc_over_sample_scale
        self.pc_sample_rate = pc_sample_rate
        self.pc_neighbor_radius = self.step_len
        self.pc_update_cost_ratio = pc_update_cost_ratio
        self.path_solutions = [] # * a list of valid goal parent vertex indices
        self.env_dict = env_dict
        self.connect_max_trial_attempts = connect_max_trial_attempts
        self.visualizer = NIRRTStarVisualizer(self.x_start, self.x_goal, self.env)
        self.agent_pos = self.x_start  
        self.trajectory_length = trajectory_length  
        self.real_path_length = 0  
        self.lines_from_start = []  
        self.lines_from_current = []  
        self.lines_to_goal = []  
        self.stop_event = threading.Event()  
        self.sampling_thread = None  
        self.find_goal = False  
        self.total_cost = 0  
        self.sampling_iter = 0  
        self.start_time = 0  
        self.end_time = 0  
        self.c_max = np.inf  
        self.c_update = np.inf  
        self.to_travel = to_travel  
        
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
            )
        else:
            pc = generate_rectangle_point_cloud(
                self.binary_mask,
                self.pc_n_points,
                self.pc_over_sample_scale,
            )
        _, _, path_pred = self.png_wrapper.generate_connected_path_points(
            pc.astype(np.float32),
            self.x_start,
            self.x_goal,
            self.env_dict,
            neighbor_radius=self.pc_neighbor_radius,
            max_trial_attempts=self.connect_max_trial_attempts,
        )
        self.path_point_cloud_pred = pc[path_pred.nonzero()[0]] # (<pc_n_points, 2)
        self.visualizer.set_path_point_cloud_pred(self.path_point_cloud_pred)
        self.visualizer.set_path_point_cloud_other(pc[np.nonzero(path_pred==0)[0]])


def get_path_planner(
    args,
    problem,
    neural_wrapper,
):
    return RTNI_RRT_Star_C_2D(
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
        args.connect_max_trial_attempts,
        args.trajectory_length,  
        args.to_travel,  
    )