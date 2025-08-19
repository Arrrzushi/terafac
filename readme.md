# Autonomous Robot Navigation System - Level 1

This project implements an autonomous robot navigation system that can navigate to goals in different corners of a simulated environment while avoiding obstacles using computer vision techniques.

## Features

- **Level 1**: Autonomous navigation to four corners (NE, NW, SE, SW) with obstacle avoidance
- **Computer Vision**: Uses image capture and analysis for obstacle detection
- **Collision Minimization**: Intelligent path planning to minimize collisions
- **Fully Autonomous**: No manual input required after launch
- **Real-time Monitoring**: Tracks collision counts and navigation progress
- **Smart Recovery**: Intelligent collision recovery using computer vision

## Prerequisites

- Python 3.7+
- Modern web browser (Chrome, Firefox, etc.)
- Git

## Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd sim-1
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

## Running the System

### Step 1: Start the Simulator

1. **Start the Python server:**
```bash
python server.py
```
This will start:
- WebSocket server on `ws://localhost:8080`
- Flask API on `http://localhost:5000`

2. **Open the simulator in your browser:**
- Open `index.html` in your web browser
- Or serve it using: `python -m http.server 8000`
- The simulator will automatically connect to the WebSocket server

### Step 2: Run the Autonomous Robot

**Run Level 1 demo:**
```bash
python smart_robot.py
```

This will:
- Run 4 navigation attempts (one to each corner)
- Use computer vision for obstacle detection
- Track collision counts for each run
- Calculate average collisions per successful run

## How It Works

### Computer Vision Approach

The system uses a simulated computer vision approach where:

1. **Image Capture**: Continuously captures images from the robot's perspective using `/capture` endpoint
2. **Obstacle Detection**: Analyzes images to detect obstacles in the path
3. **Path Planning**: Calculates safe directions avoiding detected obstacles
4. **Navigation**: Moves the robot towards the goal while avoiding collisions
5. **Collision Recovery**: Uses computer vision to find safe directions after collisions

### Navigation Strategy

1. **Goal Setting**: Sets goals in the four corners of the environment
2. **Path Calculation**: Determines the optimal path to the goal
3. **Obstacle Avoidance**: Detects obstacles and calculates avoidance maneuvers
4. **Continuous Monitoring**: Monitors progress and adjusts path as needed
5. **Smart Recovery**: Intelligent collision recovery without getting stuck in loops

### Level 1 Requirements

- ✅ Navigate to all four corners autonomously
- ✅ Use computer vision for obstacle detection (no hardcoded obstacle positions)
- ✅ Minimize collisions during navigation
- ✅ Fully autonomous operation
- ✅ Track collision statistics
- ✅ Intelligent collision recovery

## API Endpoints Used

- `POST /goal` - Set navigation goals
- `POST /move_rel` - Move robot relative to current position
- `POST /capture` - Capture images for computer vision
- `GET /collisions` - Get collision count
- `POST /reset` - Reset simulation

## Files

- `smart_robot.py` - Main autonomous robot navigation system with computer vision
- `server.py` - Main simulator server with WebSocket and Flask API
- `index.html` - 3D simulator interface using Three.js
- `requirements.txt` - Python dependencies
- `readme.md` - This documentation file

## Troubleshooting

### Common Issues

1. **"Connection refused" errors:**
   - Make sure `python server.py` is running
   - Check that ports 5000 and 8080 are available

2. **Simulator not loading:**
   - Ensure `index.html` is opened in a modern browser
   - Check browser console for JavaScript errors

3. **Robot not moving:**
   - Verify WebSocket connection in browser console
   - Check that the simulator is connected

4. **Import errors:**
   - Install dependencies: `pip install -r requirements.txt`
   - Ensure you're using Python 3.7+

### Performance Tips

- Close unnecessary browser tabs to reduce memory usage
- Use a modern browser for best 3D performance
- Ensure stable internet connection for Three.js CDN resources

## Level 1 Demo

The Level 1 demo will:

1. **Run 4 navigation attempts** - one to each corner
2. **Track collision counts** for each run
3. **Calculate average collisions** per successful run
4. **Provide detailed statistics** of the navigation performance

### Expected Output

```
=== 🧠 LEVEL 1: Computer Vision Robot Demo ===
This will run 4 attempts, one to each corner
Features: Real-time image capture, obstacle detection, intelligent path planning

--- 🏃‍♂️ Run 1/4: NE corner ---
🔄 Simulation reset
=== 🧠 COMPUTER VISION Navigation to NE corner ===
🎯 Goal set to NE corner: {'x': 45, 'z': -45}
📸 Image captured and analyzed for obstacles
🔍 Computer Vision detected 5 potential obstacles
🔄 Computer Vision chose safe direction: 135.0° (avoiding 4 obstacles)
🚀 Moved to position: {'x': 3.54, 'y': 0, 'z': -3.54}
[Additional navigation progress...]
🎯 Goal reached in NE corner!
Collisions this run: 2

[Additional runs...]

=== LEVEL 1 COMPLETED ===
Successful runs: 4/4
Average collisions per successful run: 1.75
```

## Key Features

### Computer Vision Integration
- **Real-time image capture** using `/capture` endpoint
- **Obstacle detection** without hardcoded positions
- **Safe path calculation** based on visual data
- **Intelligent avoidance** using multiple angle testing

### Collision Recovery
- **Smart backup** when collisions occur
- **Computer vision analysis** for safe direction finding
- **Loop prevention** to avoid getting stuck
- **Aggressive recovery** when standard methods fail

### Performance Optimization
- **Faster movement** when no obstacles detected
- **Efficient path planning** to minimize travel time
- **Reduced collision rates** through intelligent navigation

## Next Steps

After completing Level 1:

- **Level 2**: Implement moving obstacles and navigation
- **Level 3**: Analyze collision rates vs obstacle speeds
- **Enhancement**: Improve computer vision algorithms
- **Optimization**: Fine-tune navigation parameters

## Contributing

This is a private repository for the TeraFac selection process. Please do not share or distribute the code.

## License

Private project - All rights reserved.
