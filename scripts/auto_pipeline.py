#!/usr/bin/env python3
"""
Automated Pipeline for Robot Vision and Action
Sequentially executes camera capture, vision processing, and robot action
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import cv2
import warnings
from scipy.spatial.transform import Rotation as R
from ur_env.rotations import pose2quat

# ReKep imports
from rekep.keypoint_proposal import KeypointProposer
from rekep.constraint_generation import ConstraintGenerator
from rekep.perception.gdino import GroundingDINO
from rekep.environment import R2D2Env
from rekep.ik_solver import UR5IKSolver
from rekep.subgoal_solver import SubgoalSolver
from rekep.path_solver import PathSolver
from ur_env.ur5_env import RobotEnv
from rekep.utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_callable_grasping_cost_fn,
)

# RealSense import
import pyrealsense2 as rs

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="xFormers is not available")

class AutoPipeline:
    def __init__(self, execute_robot=False):
        self.execute_robot = execute_robot
        self.data_path = "./data/realsense_captures"
        self.script_dir = "./scripts"
        
    def run_camera_capture(self):
        """Run camera capture with live view and key press confirmation"""
        print("=== Step 1: Camera Capture ===")
        
        # Execute the camera script with modifications
        self._run_modified_camera()
        
    def _run_modified_camera(self):
        """Run modified camera capture with live view"""
        import pyrealsense2 as rs
        import numpy as np
        import cv2
        
        # Create save directory
        os.makedirs(self.data_path, exist_ok=True)
        
        # Configure streams
        pipeline = rs.pipeline()
        config = rs.config()
        
        target_serial = "819612070593"
        config.enable_device(target_serial)
        
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        pipeline.start(config)
        
        try:
            print("Camera started. Showing live view...")
            print("Press SPACE in the camera window to capture image, or press 'q' to quit")
            
            while True:
                # Wait for frames
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # Convert to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Apply colormap on depth image
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                # Combine images for display
                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = color_image.shape
                
                if depth_colormap_dim != color_colormap_dim:
                    resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                    images = np.hstack((resized_color_image, depth_colormap))
                else:
                    images = np.hstack((color_image, depth_colormap))
                
                # Add instruction text on the image
                instruction_text = "Press SPACE to capture, 'q' to quit"
                cv2.putText(images, instruction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Show images
                cv2.namedWindow('RealSense Live View', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense Live View', images)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting without capture...")
                    break
                elif key == ord(' '):  # Space key to capture
                    # Save images
                    color_path = os.path.join(self.data_path, 'varied_camera_raw.png')  
                    depth_path = os.path.join(self.data_path, 'varied_camera_depth.npy')
                    
                    cv2.imwrite(color_path, color_image)
                    np.save(depth_path, depth_image)
                    
                    print(f"Images captured and saved to {self.data_path}")
                    print(f"- RGB: varied_camera_raw.png")
                    print(f"- Depth: varied_camera_depth.npy")
                    break
                
        finally:
            cv2.destroyAllWindows()
            pipeline.stop()
    
    def run_vision_processing(self, instruction):
        """Run vision processing"""
        print("\n=== Step 2: Vision Processing ===")
        
        # Initialize vision components
        global_config = get_config(config_path="./configs/config.yaml")
        config = global_config['main']
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])

        keypoint_proposer = KeypointProposer(global_config['keypoint_proposer'])
        constraint_generator = ConstraintGenerator(global_config['constraint_generator'])
        
        # Load camera data
        color_path = os.path.join(self.data_path, 'varied_camera_raw.png')
        depth_path = os.path.join(self.data_path, 'varied_camera_depth.npy')
        
        print(f"\033[92mDebug: Looking for files at:\033[0m")
        print(f"\033[92mDebug: Color path: {color_path}\033[0m")
        print(f"\033[92mDebug: Depth path: {depth_path}\033[0m")
        
        bgr = cv2.imread(color_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path)

        print(f"\033[92mDebug: Input image shape: {rgb.shape}\033[0m")
        print(f"\033[92mDebug: Input depth shape: {depth.shape}\033[0m")
        
        # Object detection with DINO-X
        print(f"\033[92mDebug: Dino-X Detection mode\033[0m")
        gdino = GroundingDINO()
        predictions = gdino.get_dinox(color_path)
        _, masks = gdino.visualize_bbox_and_mask(predictions, color_path, './data/')
        masks = masks.astype(bool)
        masks = np.stack(masks, axis=0)

        print(f"\033[92mDebug: Generated {len(masks)} masks\033[0m")
        print(f"\033[92mDebug: masks shape: {masks[0].shape}\033[0m")
        print(f"\033[92mDebug: Type of masks: {type(masks)}\033[0m")

        # Generate point cloud from depth
        points = self._depth_to_pointcloud(depth)
        print(f"\033[92mDebug: Generated point cloud with shape: {points.shape}\033[0m")
        
        # Keypoint proposal and constraint generation
        keypoints, projected_img = keypoint_proposer.get_keypoints(rgb, points, masks)
        print(f'{bcolors.HEADER}Got {len(keypoints)} proposed keypoints{bcolors.ENDC}')
        self._show_keypoint_image(projected_img)
        
        metadata = {'init_keypoint_positions': keypoints, 'num_keypoints': len(keypoints)}
        rekep_program_dir = constraint_generator.generate(projected_img, instruction, metadata)
        print(f'{bcolors.HEADER}Constraints generated and saved in {rekep_program_dir}{bcolors.ENDC}')
        
        # Display images sequentially after vision processing is complete
        self._display_images_sequentially()
        
        return rekep_program_dir
    
    def _extract_rekep_dir_from_output(self, output):
        """Extract rekep program directory from vision script output"""
        lines = output.strip().split('\n')
        for line in lines:
            if 'rekep_program_dir:' in line:
                # Extract path from debug output
                return line.split('rekep_program_dir: ')[-1].strip()
        
        # Fallback: look for latest directory in vlm_query
        vlm_query_dir = "./vlm_query/"
        if os.path.exists(vlm_query_dir):
            vlm_dirs = [os.path.join(vlm_query_dir, d) for d in os.listdir(vlm_query_dir) 
                       if os.path.isdir(os.path.join(vlm_query_dir, d))]
            if vlm_dirs:
                return max(vlm_dirs, key=os.path.getmtime)
        
        return None
    
    def _display_images_sequentially(self):
        """Display three images sequentially with key press navigation"""
        import cv2
        import time
        
        print("\nWaiting for vision processing images to be generated...")
        
        # Try both .png and .jpg extensions and different locations
        image_files = [
            ('data/dinox_bbox', 'DINO-X Bounding Boxes'),
            ('data/dinox_mask', 'DINO-X Masks'),  
            ('data/rekep_with_keypoints', 'ReKep with Keypoints')
        ]
        
        # Wait a bit for files to be fully written
        time.sleep(2)
        
        for img_base, title in image_files:
            # Try different extensions and wait for file to exist
            img_path = None
            max_wait = 10  # seconds
            waited = 0
            
            while waited < max_wait:
                for ext in ['.png', '.jpg']:
                    test_path = img_base + ext
                    if os.path.exists(test_path) and os.path.getsize(test_path) > 0:
                        img_path = test_path
                        break
                
                if img_path:
                    break
                    
                time.sleep(1)
                waited += 1
            
            if img_path:
                print(f"\nFound image: {img_path}")
                print(f"Displaying: {title} (press any key in the window to continue)")
                img = cv2.imread(img_path)

                if img is None or not img.any():
                    print(f"Warning: {img_path} appears to be empty or corrupted")
                    continue

                h, w = img.shape[:2]
                w = w * 2
                h = h * 2
                img = cv2.resize(img, (w, h))
                if h > 1600 or w > 2400:
                    scale = min(1600/h, 2400/w)
                    img = cv2.resize(img, (int(w*scale), int(h*scale)))


                # Add instruction text on the image
                instruction_text = "Press any key to continue"
                cv2.putText(img, instruction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow(title, img)
                cv2.waitKey(0)  # Wait for any key press in the window
                cv2.destroyAllWindows()
            else:
                print(f"Warning: No image found for {title} after waiting {max_wait}s")
    
    def _get_task_instruction(self):
        """Get task instruction from user input"""
        # default_instruction = "Put the box cutter into the box."
        # default_instruction = "Put the toy bear in the box."
        # default_instruction = "Put the sponge in the tray."
        default_instruction = "Put the white cube into the box."
        
        print(f"\n=== Task Instruction Input ===")
        print(f"Default instruction: '{default_instruction}'")
        
        try:
            user_input = input("Enter task instruction (press Enter for default): ").strip()
            if user_input:
                return user_input
            else:
                return default_instruction
        except (EOFError, KeyboardInterrupt):
            print(f"\nUsing default instruction: '{default_instruction}'")
            return default_instruction
    
    def _depth_to_pointcloud(self, depth):
        """Convert depth image to point cloud"""
        # D435 default intrinsics
        fx, fy = 616.57, 616.52
        ppx, ppy = 322.57, 246.28
        depth_scale = 0.001

        height, width = depth.shape
        nx = np.linspace(0, width-1, width)
        ny = np.linspace(0, height-1, height)
        u, v = np.meshgrid(nx, ny)
        
        points = np.zeros((height * width, 3))
        valid_mask = depth > 0
        
        x = (u[valid_mask].flatten() - ppx) / fx
        y = (v[valid_mask].flatten() - ppy) / fy
        z = depth[valid_mask].flatten() * depth_scale
        
        x = np.multiply(x, z)
        y = np.multiply(y, z)

        valid_indices = np.where(valid_mask.flatten())[0]
        points[valid_indices] = np.stack((x, y, z), axis=-1)

        return points
    
    def _show_keypoint_image(self, idx_img):
        """Save keypoint image"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.imshow(idx_img)
        plt.axis('on')
        plt.title('Annotated Image with Keypoints')
        plt.savefig('./data/rekep_with_keypoints.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    def run_robot_action(self, rekep_program_dir, instruction):
        """Run robot action processing"""
        print("\n=== Step 3: Robot Action Planning ===")
        
        if not rekep_program_dir:
            print("Error: No rekep program directory provided")
            return None
        
        if self.execute_robot:
            print("Executing robot actions...")
        else:
            print("Generating action plan (robot execution disabled)...")
        
        # Initialize robot controller components
        global_config = get_config(config_path="./configs/config.yaml")
        config = global_config['main']
        
        # Set random seeds
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        
        # Setup robot components
        robot_env = RobotEnv()
        env = R2D2Env(global_config['env'])
        
        reset_joint_pos = np.array([
            0.19440510869026184, -1.9749982992755335, 1.5334253311157227, 
            5.154152870178223, -1.5606663862811487, 1.7688038349151611
        ])
        
        ik_solver = UR5IKSolver(reset_joint_pos=reset_joint_pos, world2robot_homo=None)
        subgoal_solver = SubgoalSolver(global_config['subgoal_solver'], ik_solver, reset_joint_pos)
        path_solver = PathSolver(global_config['path_solver'], ik_solver, reset_joint_pos)
        
        # Execute the robot planning
        all_subgoals_data = self._execute_robot_planning(
            rekep_program_dir, env, robot_env, subgoal_solver, path_solver, config
        )
        
        return all_subgoals_data
        
    def _execute_robot_planning(self, rekep_program_dir, env, robot_env, subgoal_solver, path_solver, config):
        """Execute robot planning and return all subgoals"""
        # Load program info and constraints
        program_info = self._load_program_info(rekep_program_dir)
        constraint_fns = self._load_constraints(rekep_program_dir, env)
        
        # Setup environment state
        env.register_keypoints(program_info['init_keypoint_positions'])
        scene_keypoints = env.get_keypoint_positions()
        world_keypoints = self._transform_keypoints_to_world(scene_keypoints, robot_env)
        
        keypoints = np.concatenate([[self._get_ee_position(robot_env)], world_keypoints], axis=0)
        curr_ee_pose = self._get_ee_pose(robot_env)
        curr_joint_pos = self._get_joint_positions(robot_env)
        sdf_voxels = env.get_sdf_voxels(config['sdf_voxel_size'])
        collision_points = env.get_collision_points()
        
        keypoint_movable_mask = np.zeros(program_info['num_keypoints'] + 1, dtype=bool)
        keypoint_movable_mask[0] = True
        
        # Store all subgoals
        all_subgoals = []
        
        # Process each stage
        num_stages = program_info['num_stages']
        print(f"Processing {num_stages} stages...")
        
        for stage in range(1, num_stages + 1):
            print(f"\n--- Processing Stage {stage} ---")
            
            # Update stage info
            is_grasp_stage = program_info['grasp_keypoints'][stage - 1] != -1
            is_release_stage = program_info['release_keypoints'][stage - 1] != -1
            
            # Generate subgoal
            subgoal_constraints = constraint_fns[stage]['subgoal']
            path_constraints = constraint_fns[stage]['path']
            
            subgoal_pose, _ = subgoal_solver.solve(
                curr_ee_pose, keypoints, keypoint_movable_mask,
                subgoal_constraints, path_constraints, sdf_voxels, collision_points,
                is_grasp_stage, curr_joint_pos, from_scratch=True
            )
            
            # Maintain current orientation
            subgoal_pose[3:7] = curr_ee_pose[3:7]
            
            print(f"Stage {stage} subgoal: {subgoal_pose}")
            
            # Store subgoal info
            subgoal_info = {
                "stage": stage,
                "subgoal_pose": subgoal_pose.tolist(),
                "current_ee_pose": curr_ee_pose.tolist(),
                "is_grasp_stage": is_grasp_stage,
                "is_release_stage": is_release_stage
            }
            all_subgoals.append(subgoal_info)
            
            # Update ee pose for next stage
            curr_ee_pose = subgoal_pose.copy()
            keypoints[0] = subgoal_pose[:3]
        
        # Save all subgoals
        os.makedirs('./outputs', exist_ok=True)
        all_subgoals_path = './outputs/all_subgoals.json'
        
        summary_data = {
            "num_stages": len(all_subgoals),
            "subgoals": all_subgoals
        }
        
        with open(all_subgoals_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        
        print(f"{bcolors.OKGREEN}All subgoals saved to {all_subgoals_path}{bcolors.ENDC}")
        return summary_data

    def _load_program_info(self, rekep_program_dir):
        """Load program information"""
        with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
            return json.load(f)

    def _load_constraints(self, rekep_program_dir, env):
        """Load constraints for all stages"""
        program_info = self._load_program_info(rekep_program_dir)
        constraint_fns = {}
        for stage in range(1, program_info['num_stages'] + 1):
            stage_dict = {}
            for constraint_type in ['subgoal', 'path']:
                load_path = os.path.join(rekep_program_dir, f'stage{stage}_{constraint_type}_constraints.txt')
                if os.path.exists(load_path):
                    get_grasping_cost_fn = get_callable_grasping_cost_fn(env)
                    stage_dict[constraint_type] = load_functions_from_txt(load_path, get_grasping_cost_fn)
                else:
                    stage_dict[constraint_type] = []
            constraint_fns[stage] = stage_dict
        return constraint_fns
    
    def _get_joint_positions(self, robot_env):
        """Get joint positions"""
        return robot_env.robot.get_joint_positions()
    
    def _get_ee_position(self, robot_env):
        """Get end effector position"""
        return robot_env.robot.get_tcp_pose()[:3]
    
    def _get_ee_pose(self, robot_env):
        """Get end effector pose"""
        ee_pos = robot_env.robot.get_tcp_pose()
        return pose2quat(ee_pos)

    def _transform_keypoints_to_world(self, keypoints, robot_env):
        """Transform keypoints from camera coordinate system to world coordinate system"""
        keypoints = np.array(keypoints)
        
        # Load camera extrinsics
        ee2camera = self._load_camera_extrinsics()
        
        # Convert to homogeneous coordinates
        keypoints_homogeneous = np.hstack((keypoints, np.ones((keypoints.shape[0], 1))))
        
        # Get end effector pose
        ee_pose = self._get_ee_pose(robot_env)
        quat = np.array([ee_pose[3], ee_pose[4], ee_pose[5], ee_pose[6]])
        rotation = R.from_quat(quat).as_matrix()
        
        # Create transformation matrix
        base2ee = np.eye(4)
        base2ee[:3, :3] = rotation
        base2ee[:3, 3] = ee_pose[:3]
        
        # Apply transformation
        camera_frame = base2ee @ ee2camera
        base_coords_homogeneous = (camera_frame @ keypoints_homogeneous.T).T
        
        # Convert to non-homogeneous coordinates
        return base_coords_homogeneous[:, :3] / base_coords_homogeneous[:, 3, np.newaxis]

    def _load_camera_extrinsics(self):
        """Load camera extrinsics from wrist_to_d435.tf"""
        extrinsics_path = 'src/pogs/calibration_outputs/wrist_to_d435.tf'
        with open(extrinsics_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Skip header lines
        data_lines = lines[2:]
        
        # Translation vector
        translation = np.array([float(x) for x in data_lines[0].split()])
        
        # 3x3 rotation matrix
        rotation = np.array([[float(x) for x in line.split()] for line in data_lines[1:4]])
        
        # Create 4x4 homogeneous transformation matrix
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = rotation
        extrinsics[:3, 3] = translation
        
        return extrinsics
    
    def _load_action_sequence(self):
        """Load the generated all_subgoals"""
        all_subgoals_path = "./outputs/all_subgoals.json"
        
        if os.path.exists(all_subgoals_path):
            with open(all_subgoals_path, 'r') as f:
                return json.load(f)
        
        return None
    
    def save_final_output(self, all_subgoals_data):
        """Display final output summary"""
        if all_subgoals_data:
            print(f"\n=== Final Output: all_subgoals.json ===")
            print(f"File location: ./outputs/all_subgoals.json")
            
            num_stages = all_subgoals_data.get('num_stages', 0)
            print(f"Total stages processed: {num_stages}")
            
            # Print each subgoal position
            for subgoal in all_subgoals_data.get('subgoals', []):
                stage = subgoal['stage']
                pos = subgoal['subgoal_pose'][:3]
                action = "GRASP" if subgoal['is_grasp_stage'] else ("RELEASE" if subgoal['is_release_stage'] else "MOVE")
                print(f"Stage {stage} ({action}): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        else:
            print("Warning: No subgoals data found")
    
    def run_complete_pipeline(self):
        """Run the complete pipeline"""
        print("Starting Automated Robot Pipeline")
        print(f"Robot execution: {'ENABLED' if self.execute_robot else 'DISABLED'}")
        
        # Get task instruction from user
        instruction = self._get_task_instruction()
        print(f"\nTask instruction: {instruction}")
        
        try:
            # Step 1: Camera capture
            self.run_camera_capture()
            
            # Step 2: Vision processing
            rekep_program_dir = self.run_vision_processing(instruction)
            
            # Step 3: Robot action
            all_subgoals_data = self.run_robot_action(rekep_program_dir, instruction)
            
            # Step 4: Display final output summary
            self.save_final_output(all_subgoals_data)
            
            print("\n=== Pipeline Completed Successfully ===")
            
        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Automated Robot Pipeline')
    parser.add_argument('--execute-robot', action='store_true', 
                       help='Actually execute robot actions (default: plan only)')
    args = parser.parse_args()
    
    pipeline = AutoPipeline(execute_robot=args.execute_robot)
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()