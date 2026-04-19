import time
import json
import logging
from pathlib import Path
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def record(robot, duration=10, fps=30):
    print(f"Torque disabled. You can move the arm now.")
    print(f"Recording for {duration} seconds at {fps} FPS...")
    
    robot.bus.disable_torque()
    
    trajectory = []
    start_time = time.time()
    interval = 1.0 / fps
    
    try:
        while (time.time() - start_time) < duration:
            loop_start = time.time()
            
            # Read joint positions
            obs = robot.get_observation()
            # Extract only joint positions (excluding camera data if any)
            joints = {k: v for k, v in obs.items() if k.endswith('.pos')}
            trajectory.append(joints)
            
            # Maintain FPS
            elapsed = time.time() - loop_start
            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    
    return trajectory

def replay(robot, trajectory, fps=30):
    print("Replaying trajectory...")
    
    # Enable torque and move to starting position slowly
    robot.bus.enable_torque()
    
    if not trajectory:
        print("No trajectory to replay.")
        return

    # Move to start position first
    print("Moving to start position...")
    robot.send_action(trajectory[0])
    time.sleep(2) # Wait for it to reach start position
    
    interval = 1.0 / fps
    for action in trajectory:
        loop_start = time.time()
        
        robot.send_action(action)
        
        # Maintain FPS
        elapsed = time.time() - loop_start
        sleep_time = max(0, interval - elapsed)
        time.sleep(sleep_time)
        
    print("Replay finished.")

def main():
    # --- Configuration ---
    # Change 'COM3' to your actual port found by lerobot-find-port
    port = input("Please enter the COM port (e.g., COM3): ").strip()
    if not port:
        print("Port is required.")
        return

    config = SO101FollowerConfig(port=port, id="so101_slave")
    robot = SO101Follower(config)
    
    try:
        print(f"Connecting to SO-101 on {port} with ID 'so101_slave'...")
        robot.connect(calibrate=True) # Now we enable calibration check
        
        while True:
            choice = input("\nEnter 'r' to record, 'p' to play, 's' to save, 'l' to load, or 'q' to quit: ").lower()
            
            if choice == 'r':
                dur = float(input("Enter duration (seconds): ") or 10)
                traj = record(robot, duration=dur)
                print(f"Recorded {len(traj)} frames.")
            elif choice == 'p':
                if 'traj' in locals() and traj:
                    replay(robot, traj)
                else:
                    print("No recording in memory. Record or load first.")
            elif choice == 's':
                if 'traj' in locals() and traj:
                    filename = input("Enter filename to save (e.g., move.json): ") or "move.json"
                    with open(filename, 'w') as f:
                        json.dump(traj, f)
                    print(f"Saved to {filename}")
                else:
                    print("Nothing to save.")
            elif choice == 'l':
                filename = input("Enter filename to load: ") or "move.json"
                if Path(filename).exists():
                    with open(filename, 'r') as f:
                        traj = json.load(f)
                    print(f"Loaded {len(traj)} frames.")
                else:
                    print(f"File {filename} not found.")
            elif choice == 'q':
                break
                
    finally:
        if robot.is_connected:
            robot.bus.disable_torque() # Leave it in passive mode
            print("Disconnected.")

if __name__ == "__main__":
    main()
