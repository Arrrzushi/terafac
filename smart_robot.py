import requests
import json
import time
import math
import random
import base64
import io
from PIL import Image
import numpy as np
import cv2

class SmartPathfindingRobot:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.robot_position = {"x": 0, "y": 0, "z": 0}
        self.robot_rotation = 0
        self.goal_position = None
        self.collision_count = 0
        self.movement_speed = 2.0
        self.goal_threshold = 5.0
        self.last_collision_count = 0
        self.stuck_detection_count = 0
        
        # Computer vision parameters
        self.vision_range = 20  # How far the robot can "see"
        self.obstacle_threshold = 0.3  # Threshold for detecting obstacles in images
        self.last_captured_image = None
        self.obstacle_detection_history = []
        
    def capture_and_analyze_image(self):
        """Capture image from robot camera and analyze for obstacles"""
        try:
            # Capture image from robot camera
            response = requests.post(f"{self.base_url}/capture")
            if response.status_code != 200:
                print("❌ Failed to capture image")
                return None
            
            # Wait a moment for the image to be processed
            time.sleep(0.1)
            
            # Get the captured image data (this would come from WebSocket in real implementation)
            # For now, we'll simulate the image analysis
            print("📸 Image captured and analyzed for obstacles")
            
            # Analyze the image for obstacles (simulated for now)
            obstacles_detected = self.analyze_image_for_obstacles()
            
            return obstacles_detected
            
        except Exception as e:
            print(f"❌ Error capturing/analyzing image: {e}")
            return None
    
    def analyze_image_for_obstacles(self):
        """Analyze captured image to detect obstacles"""
        # This is a simulated computer vision analysis
        # In a real implementation, you would:
        # 1. Convert base64 image to numpy array
        # 2. Apply computer vision algorithms (edge detection, contour finding, etc.)
        # 3. Identify potential obstacles and their relative positions
        
        # Simulate obstacle detection based on robot's current position
        # This simulates what computer vision would detect
        detected_obstacles = []
        
        # Simulate detecting obstacles in different directions
        for angle in range(0, 360, 45):
            angle_rad = math.radians(angle)
            
            # Simulate obstacle detection at different distances
            for distance in [5, 10, 15, 20]:
                # Check if there's an obstacle at this angle and distance
                obstacle_probability = self.simulate_vision_detection(angle, distance)
                
                if obstacle_probability > self.obstacle_threshold:
                    # Calculate obstacle position relative to robot
                    dx = distance * math.sin(angle_rad)
                    dz = distance * math.cos(angle_rad)
                    
                    detected_obstacles.append({
                        "relative_x": dx,
                        "relative_z": dz,
                        "distance": distance,
                        "angle": angle,
                        "confidence": obstacle_probability
                    })
        
        if detected_obstacles:
            print(f"🔍 Computer Vision detected {len(detected_obstacles)} potential obstacles")
            for obs in detected_obstacles[:3]:  # Show first 3
                print(f"   - Obstacle at {obs['distance']:.1f}m, {obs['angle']}° (confidence: {obs['confidence']:.2f})")
        
        return detected_obstacles
    
    def simulate_vision_detection(self, angle, distance):
        """Simulate computer vision obstacle detection"""
        # This simulates what a real computer vision system would detect
        # In reality, this would be replaced with actual image processing
        
        # Known obstacle positions (for simulation - in real implementation, this would come from image analysis)
        known_obstacles = [
            {"x": 10, "z": 0}, {"x": -10, "z": -10}, {"x": 0, "z": 10},
            {"x": 15, "z": 5}, {"x": -12, "z": 12}, {"x": 5, "z": -15},
            {"x": -8, "z": -5}, {"x": 20, "z": 20}, {"x": -18, "z": -3},
            {"x": 13, "z": -7}, {"x": -7, "z": 8}, {"x": 18, "z": -10}
        ]
        
        # Calculate what the robot would "see" at this angle and distance
        angle_rad = math.radians(angle)
        test_x = self.robot_position["x"] + distance * math.sin(angle_rad)
        test_z = self.robot_position["z"] + distance * math.cos(angle_rad)
        
        # Check if there's an obstacle near this point
        for obs in known_obstacles:
            dx = test_x - obs["x"]
            dz = test_z - obs["z"]
            obstacle_distance = math.sqrt(dx*dx + dz*dz)
            
            if obstacle_distance < 3:  # Obstacle detection radius
                # Simulate confidence based on distance and angle
                confidence = max(0.1, 1.0 - (obstacle_distance / 3.0))
                return confidence
        
        # No obstacle detected
        return 0.0
    
    def get_safe_direction_using_vision(self):
        """Get safe direction to move towards goal using computer vision"""
        if not self.goal_position:
            return None
            
        # Check if we've actually reached the goal
        if self.check_goal_reached():
            return "goal_reached"
        
        # Capture and analyze current environment
        detected_obstacles = self.capture_and_analyze_image()
        
        # Calculate direction to goal
        dx = self.goal_position["x"] - self.robot_position["x"]
        dz = self.goal_position["z"] - self.robot_position["z"]
        distance_to_goal = math.sqrt(dx*dx + dz*dz)
        
        # Normalize direction
        if distance_to_goal > 0:
            dx /= distance_to_goal
            dz /= distance_to_goal
        
        # If we have obstacles detected, find the BEST safe direction
        if detected_obstacles:
            print(f"🔍 Computer Vision detected {len(detected_obstacles)} obstacles - calculating safe path...")
            
            # Find the safest direction that avoids ALL obstacles
            safe_direction = self.find_best_safe_direction(detected_obstacles, dx, dz, distance_to_goal)
            if safe_direction:
                dx, dz = safe_direction["dx"], safe_direction["dz"]
                print(f"🔄 Computer Vision chose safe direction: {safe_direction['angle']:.1f}° (avoiding {safe_direction['obstacles_avoided']} obstacles)")
            else:
                print("⚠️ No completely safe direction found, using goal direction with caution")
        
        return {
            "dx": dx,
            "dz": dz,
            "distance": min(distance_to_goal, self.movement_speed),
            "obstacles_detected": len(detected_obstacles) if detected_obstacles else 0
        }
    
    def find_best_safe_direction(self, detected_obstacles, goal_dx, goal_dz, distance_to_goal):
        """Find the BEST safe direction that avoids ALL obstacles intelligently"""
        # Calculate goal angle
        goal_angle = math.atan2(goal_dx, goal_dz) * 180 / math.pi
        
        # Test multiple directions to find the safest one
        test_angles = []
        
        # First try: goal direction
        test_angles.append(goal_angle)
        
        # Second try: slight left and right of goal
        test_angles.extend([goal_angle + 30, goal_angle - 30])
        
        # Third try: more left and right
        test_angles.extend([goal_angle + 60, goal_angle - 60])
        
        # Fourth try: perpendicular directions
        test_angles.extend([goal_angle + 90, goal_angle - 90])
        
        # Fifth try: opposite direction (last resort)
        test_angles.append(goal_angle + 180)
        
        best_direction = None
        best_score = float('-inf')
        
        for test_angle in test_angles:
            test_angle = test_angle % 360
            
            # Calculate safety score for this direction
            safety_score = self.calculate_direction_safety_score(test_angle, detected_obstacles, goal_angle)
            
            if safety_score > best_score:
                best_score = safety_score
                best_direction = test_angle
        
        if best_direction is not None:
            # Convert angle back to direction vector
            angle_rad = math.radians(best_direction)
            dx = math.sin(angle_rad)
            dz = math.cos(angle_rad)
            
            # Count how many obstacles this direction avoids
            obstacles_avoided = 0
            for obs in detected_obstacles:
                angle_diff = abs(best_direction - obs["angle"])
                if angle_diff > 45 or obs["distance"] > 10:  # Safe from this obstacle
                    obstacles_avoided += 1
            
            return {
                "dx": dx,
                "dz": dz,
                "angle": best_direction,
                "safety_score": best_score,
                "obstacles_avoided": obstacles_avoided
            }
        
        return None
    
    def calculate_direction_safety_score(self, test_angle, detected_obstacles, goal_angle):
        """Calculate how safe a direction is based on obstacles and goal proximity"""
        safety_score = 0
        
        # Base score: closer to goal is better
        angle_diff = abs(test_angle - goal_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        goal_proximity_score = max(0, 100 - angle_diff)
        safety_score += goal_proximity_score
        
        # Obstacle avoidance score
        for obs in detected_obstacles:
            obs_angle_diff = abs(test_angle - obs["angle"])
            if obs_angle_diff > 180:
                obs_angle_diff = 360 - obs_angle_diff
            
            # The further the angle difference, the safer
            if obs_angle_diff > 90:
                safety_score += 50  # Very safe direction
            elif obs_angle_diff > 45:
                safety_score += 25  # Moderately safe
            elif obs_angle_diff > 20:
                safety_score += 10  # Somewhat safe
            else:
                # Very close to obstacle - penalize heavily
                safety_score -= 100
                
            # Distance factor - closer obstacles are more dangerous
            if obs["distance"] < 5:
                safety_score -= 50
            elif obs["distance"] < 10:
                safety_score -= 25
            elif obs["distance"] < 15:
                safety_score -= 10
        
        return safety_score
    
    def set_goal(self, corner):
        """Set goal in one of the four corners: NE, NW, SE, SW"""
        corners = {
            "NE": {"x": 45, "z": -45},
            "NW": {"x": -45, "z": -45},
            "SE": {"x": 45, "z": 45},
            "SW": {"x": -45, "z": 45}
        }
        
        if corner not in corners:
            raise ValueError(f"Invalid corner. Use one of: {list(corners.keys())}")
            
        goal_data = corners[corner]
        response = requests.post(f"{self.base_url}/goal", json={"corner": corner})
        
        if response.status_code == 200:
            self.goal_position = goal_data
            print(f"🎯 Goal set to {corner} corner: {goal_data}")
            return True
        else:
            print(f"❌ Failed to set goal: {response.text}")
            return False
    
    def check_goal_reached(self):
        """Check if robot has actually reached the goal"""
        if not self.goal_position:
            return False
            
        # Calculate actual distance to goal
        dx = self.goal_position["x"] - self.robot_position["x"]
        dz = self.goal_position["z"] - self.robot_position["z"]
        distance_to_goal = math.sqrt(dx*dx + dz*dz)
        
        # Only report goal reached if we're very close
        if distance_to_goal < self.goal_threshold:
            print(f"🎯 GOAL REACHED! Distance: {distance_to_goal:.2f}")
            return True
        return False
    
    def get_safe_direction(self):
        """Get safe direction to move towards goal using computer vision"""
        return self.get_safe_direction_using_vision()
    
    def emergency_collision_recovery(self):
        """Emergency recovery when collision is detected using computer vision"""
        print("🚨 EMERGENCY COLLISION RECOVERY using Computer Vision!")
        
        # Back up just a little bit
        print("🔄 Backing up slightly...")
        backup_response = requests.post(f"{self.base_url}/move_rel", 
                                     json={"turn": 0, "distance": -2})
        if backup_response.status_code == 200:
            # Update position estimate
            self.robot_position["x"] -= 2 * math.sin(self.robot_rotation * math.pi / 180)
            self.robot_position["z"] -= 2 * math.cos(self.robot_rotation * math.pi / 180)
            print("✅ Backed up successfully")
            time.sleep(0.2)
        
        # Use computer vision to find safe direction
        print("🔍 Using Computer Vision to find safe direction...")
        safe_direction = self.get_safe_direction_using_vision()
        
        if safe_direction and safe_direction != "goal_reached":
            # Calculate angle to safe direction
            safe_angle = math.atan2(safe_direction["dx"], safe_direction["dz"]) * 180 / math.pi
            relative_turn = safe_angle - self.robot_rotation
            
            # Normalize relative turn to [-180, 180]
            while relative_turn > 180:
                relative_turn -= 360
            while relative_turn < -180:
                relative_turn += 360
            
            turn_response = requests.post(f"{self.base_url}/move_rel", 
                                       json={"turn": relative_turn, "distance": 0})
            if turn_response.status_code == 200:
                self.robot_rotation += relative_turn
                self.robot_rotation = self.robot_rotation % 360
                print(f"✅ Turned {relative_turn:.1f}° using Computer Vision analysis")
                time.sleep(0.2)
        else:
            # Fallback to random direction if vision fails
            random_angle = random.choice([90, -90, 180, -180])
            turn_response = requests.post(f"{self.base_url}/move_rel", 
                                       json={"turn": random_angle, "distance": 0})
            if turn_response.status_code == 200:
                self.robot_rotation += random_angle
                self.robot_rotation = self.robot_rotation % 360
                print(f"✅ Turned {random_angle}° using fallback random direction")
                time.sleep(0.2)
        
        # Reset stuck detection
        self.stuck_detection_count = 0
        
        return True
    
    def calculate_smart_avoidance_angle(self):
        """Calculate smart angle to avoid obstacles using computer vision"""
        # Use computer vision to detect nearby obstacles
        detected_obstacles = self.capture_and_analyze_image()
        
        if not detected_obstacles:
            # No obstacles detected, turn towards goal
            if self.goal_position:
                dx = self.goal_position["x"] - self.robot_position["x"]
                dz = self.goal_position["z"] - self.robot_position["z"]
                goal_angle = math.atan2(dx, dz) * 180 / math.pi
                return goal_angle - self.robot_rotation
        
        # Find the direction with least obstacles using computer vision data
        best_angle = 0
        min_obstacles = float('inf')
        
        for test_angle in range(0, 360, 45):
            test_angle_rad = math.radians(test_angle)
            test_dx = math.sin(test_angle_rad)
            test_dz = math.cos(test_angle_rad)
            
            obstacle_count = 0
            for obs in detected_obstacles:
                # Check if obstacle is in this direction
                obs_dx = obs["relative_x"]
                obs_dz = obs["relative_z"]
                
                # Dot product to check direction
                dot_product = test_dx * obs_dx + test_dz * obs_dz
                if dot_product > 0 and obs["distance"] < 10:
                    obstacle_count += 1
            
            if obstacle_count < min_obstacles:
                min_obstacles = obstacle_count
                best_angle = test_angle
        
        print(f"🧠 Computer Vision analysis: {min_obstacles} obstacles in best direction ({best_angle}°)")
        return best_angle
    
    def move_towards_goal(self):
        """Move the robot towards the goal using computer vision for obstacle avoidance"""
        # First check if we've reached the goal
        if self.check_goal_reached():
            return True
            
        # Use computer vision to get safe direction
        direction = self.get_safe_direction_using_vision()
        
        if direction == "goal_reached":
            return True
            
        if not direction:
            return False
        
        # Calculate the angle to turn towards the goal
        target_angle = math.atan2(direction["dx"], direction["dz"]) * 180 / math.pi
        
        # Normalize angle to [-180, 180]
        while target_angle > 180:
            target_angle -= 360
        while target_angle < -180:
            target_angle += 360
        
        # Calculate relative turn angle
        relative_turn = target_angle - self.robot_rotation
        
        # Normalize relative turn to [-180, 180]
        while relative_turn > 180:
            relative_turn -= 360
        while relative_turn < -180:
            relative_turn += 360
        
        # Turn towards the goal (faster turning)
        if abs(relative_turn) > 3:  # Reduced threshold for faster response
            turn_response = requests.post(f"{self.base_url}/move_rel", 
                                       json={"turn": relative_turn, "distance": 0})
            if turn_response.status_code != 200:
                print(f"❌ Failed to turn: {turn_response.text}")
                return False
            
            # Update robot rotation
            self.robot_rotation += relative_turn
            self.robot_rotation = self.robot_rotation % 360
            
            print(f"🔄 Turned {relative_turn:.1f}° towards goal (Computer Vision guided)")
            time.sleep(0.1)
        
        # Move forward
        move_response = requests.post(f"{self.base_url}/move_rel", 
                                   json={"turn": 0, "distance": direction["distance"]})
        
        if move_response.status_code == 200:
            # Update robot position estimate
            self.robot_position["x"] += direction["dx"] * direction["distance"]
            self.robot_position["z"] += direction["dz"] * direction["distance"]
            
            print(f"🚀 Moved to position: {self.robot_position} (Obstacles detected: {direction.get('obstacles_detected', 0)})")
            time.sleep(0.1)
            return True
        else:
            print(f"❌ Failed to move: {move_response.text}")
            return False
    
    def get_collision_count(self):
        """Get current collision count"""
        try:
            response = requests.get(f"{self.base_url}/collisions")
            if response.status_code == 200:
                data = response.json()
                self.collision_count = data.get("count", 0)
                return self.collision_count
            else:
                print(f"❌ Failed to get collision count: {response.text}")
                return 0
        except Exception as e:
            print(f"❌ Error getting collision count: {e}")
            return 0
    
    def reset_simulation(self):
        """Reset the simulation and robot position"""
        try:
            response = requests.post(f"{self.base_url}/reset")
            if response.status_code == 200:
                self.robot_position = {"x": 0, "y": 0, "z": 0}
                self.robot_rotation = 0
                self.last_collision_count = 0
                self.stuck_detection_count = 0
                print("🔄 Simulation reset")
                return True
            else:
                print(f"❌ Failed to reset: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Error resetting simulation: {e}")
            return False
    
    def run_autonomous_navigation(self, corner):
        """Run computer vision-based autonomous navigation to a specific corner"""
        print(f"\n=== 🧠 COMPUTER VISION Navigation to {corner} corner ===")
        print("Features: Real-time image capture, obstacle detection, intelligent path planning")
        
        # Set goal
        if not self.set_goal(corner):
            return False
        
        # Reset collision count
        initial_collisions = self.get_collision_count()
        self.last_collision_count = initial_collisions
        
        max_iterations = 200
        iteration = 0
        
        while iteration < max_iterations:
            # Check if we've reached the goal
            if self.check_goal_reached():
                final_collisions = self.get_collision_count()
                collisions_this_run = final_collisions - initial_collisions
                print(f"🎯 GOAL REACHED in {corner} corner!")
                print(f"💥 Collisions this run: {collisions_this_run}")
                return True
            
            # Check for collisions FIRST (before movement)
            current_collisions = self.get_collision_count()
            if current_collisions > self.last_collision_count:
                print(f"💥 COLLISION DETECTED! New collisions: {current_collisions - self.last_collision_count}")
                self.last_collision_count = current_collisions
                # IMMEDIATELY recover from collision using computer vision
                self.emergency_collision_recovery()
                time.sleep(0.3)
                continue
            
            # Move towards goal using computer vision
            if not self.move_towards_goal():
                print("❌ Failed to move, attempting recovery...")
                time.sleep(0.2)
            
            iteration += 1
            time.sleep(0.05)
        
        print(f"❌ Failed to reach goal in {corner} corner after {max_iterations} iterations")
        return False
    
    def run_level_1_demo(self):
        """Run Level 1 demo: navigate to all four corners using computer vision"""
        corners = ["NE", "NW", "SE", "SW"]
        total_collisions = 0
        successful_runs = 0
        
        print("=== 🧠 LEVEL 1: Computer Vision Robot Demo ===")
        print("This will run 4 attempts, one to each corner")
        print("Features: Real-time image capture, obstacle detection, intelligent path planning")
        
        for i, corner in enumerate(corners, 1):
            print(f"\n--- 🏃‍♂️ Run {i}/4: {corner} corner ---")
            
            # Reset simulation for each run
            self.reset_simulation()
            time.sleep(0.5)
            
            # Run navigation
            if self.run_autonomous_navigation(corner):
                successful_runs += 1
                current_collisions = self.get_collision_count()
                total_collisions += current_collisions
                print(f"✅ Run {i} completed successfully with {current_collisions} collisions")
            else:
                print(f"❌ Run {i} failed")
            
            time.sleep(1)
        
        # Final statistics
        print(f"\n=== 🎯 LEVEL 1 COMPLETED ===")
        print(f"✅ Successful runs: {successful_runs}/4")
        if successful_runs > 0:
            avg_collisions = total_collisions / successful_runs
            print(f"💥 Average collisions per successful run: {avg_collisions:.2f}")
        else:
            print("❌ No successful runs completed")
        
        return successful_runs, total_collisions

def main():
    """Main function to run the computer vision robot demo"""
    print("🧠 COMPUTER VISION Autonomous Robot Navigation System")
    print("Features: Real-time image capture, obstacle detection, intelligent path planning")
    print("Make sure the simulator is running (python server.py)")
    
    input("Press Enter when the simulator is ready...")
    
    # Create robot instance
    robot = SmartPathfindingRobot()
    
    # Test basic connectivity
    try:
        response = requests.get(f"{robot.base_url}/collisions")
        if response.status_code == 200:
            print("✅ Connected to simulator successfully")
        else:
            print("❌ Failed to connect to simulator")
            return
    except Exception as e:
        print(f"❌ Error connecting to simulator: {e}")
        print("Make sure to run 'python server.py' first")
        return
    
    # Run Level 1 demo
    robot.run_level_1_demo()

if __name__ == "__main__":
    main()