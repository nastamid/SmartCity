# Smart City - Autonomous Vehicle Simulation

![Unity](https://img.shields.io/badge/Unity-2021.3.16f1-black?logo=unity)
![ML-Agents](https://img.shields.io/badge/ML--Agents-2.3.0-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1-red?logo=pytorch)
![C#](https://img.shields.io/badge/C%23-10-239120?logo=c-sharp)
![License](https://img.shields.io/badge/License-Academic-green)

A Unity-based reinforcement learning simulation exploring autonomous vehicle navigation in urban environments without traffic signals. This project demonstrates how collective behavior patterns observed in nature (flocking of birds and fish) can be applied to coordinated autonomous vehicle movement.

## 📚 Academic Context

This project was developed as a Master's thesis at **Georgian Technical University** (February 2023) for the Computer Science program. The research validates the theory that autonomous vehicles can safely navigate complex intersections without traditional traffic control systems through machine learning and collective behavior patterns.

**Thesis Title:** Autonomous Vehicle Simulation in Urban Environment  
**Author:** Nikoloz Astamidze  
**Program:** Informatics (Code: 0613)  
**Supervisor:** Professor Merab Akhobadze  
**Reviewer:** Professor Mariam Chkhaidze

### 📊 Defense Presentation

**[View Master's Defense Presentation (English)](https://docs.google.com/presentation/d/1NOcKFnHB3iI_q2XVUcLx_OZ09nRyxu88X45hr9WY3zg)**

The defense presentation provides a visual overview of the research methodology, experimental results, and key findings.

---

## 🎯 Project Overview

### Core Concept

The simulation tests whether autonomous vehicles can adopt decentralized, nature-inspired navigation strategies similar to how flocks of birds or schools of fish move cohesively without central coordination. Instead of relying on traffic lights and road signs, vehicles learn to:

- Perceive their environment through simplified visual sensors
- Make real-time navigation decisions
- Avoid collisions with other vehicles and obstacles
- Follow road paths efficiently
- Coordinate movements collectively

### Key Innovation

**Grid Sensor Vision System:** Agents perceive the environment as colored pixels rather than complex 3D geometry:
- 🟢 **Green:** Reward ball (path guidance)
- 🔴 **Red:** Sidewalks and boundaries
- 🔵 **Blue:** Other vehicles

This simplified perception enables faster training while maintaining effective navigation capabilities.

---

## 🏗️ Architecture

### System Components

```
SmartCity/
├── ML Training System
│   ├── TeamDriverAgent.cs        # Main RL agent implementation
│   ├── MLTrainingScene.cs        # Multi-agent training orchestration
│   └── Reward system             # Behavior shaping logic
│
├── Car Control System
│   ├── CarController.cs          # Vehicle physics and movement
│   ├── CarPercepts.cs            # Collision and trigger detection
│   └── ICarAgent.cs              # Agent interface contract
│
├── Perception System
│   ├── Grid Sensors              # Simplified visual perception
│   └── Detectable Objects        # Environment pixelization
│
├── Navigation System
│   ├── PathCrawler.cs            # Path following logic
│   ├── NodePath.cs               # Waypoint management
│   └── Connected path network    # Road infrastructure
│
└── Environment
    ├── Road pieces               # Modular road segments
    ├── Intersections             # 3-way and 4-way junctions
    └── Training scenarios        # Progressive difficulty levels
```

### Design Principles

Following **Clean Architecture** and **SOLID** principles:
- **Separation of Concerns:** Domain logic isolated from Unity engine dependencies
- **Interface Segregation:** `ICarAgent` defines clear agent contracts
- **Dependency Injection:** Components referenced through serialized fields
- **Single Responsibility:** Each class handles one specific aspect
- **Law of Demeter:** Minimal coupling between components

---

## 🧠 Machine Learning Details

### Reinforcement Learning Setup

**Algorithm:** Proximal Policy Optimization (PPO) with Deep Q-Networks (DQN)

**Observation Space (3 parameters):**
```csharp
sensor.AddObservation(transform.rotation.y);      // Vehicle orientation
sensor.AddObservation(_carController.velocity);   // Current speed
sensor.AddObservation(PathCrawler.currentSideDist); // Distance from path center
```

**Action Space:**
- **Discrete Actions:** Forward (1), Idle (0), Reverse (2)
- **Continuous Actions:** Steering angle [-1, 1] → [-40°, 40°]

### Reward System

| Event | Reward | Purpose |
|-------|--------|---------|
| Reach path node (🟢) | +0.2 | Encourage forward progress |
| Collision with car (🔵) | -10.0 | Strong penalty for accidents |
| Hit sidewalk (🔴) | -1.0 | Discourage boundary violations |
| Cross lane line | -0.1 | Keep within proper lane |
| Flip upside down | -1.0 | Penalize unstable driving |
| Existential penalty | -1/MaxStep | Motivate efficient completion |

**Episode Termination Conditions:**
- Sidewalk collision
- Vehicle-to-vehicle collision
- Upside-down orientation (>45° tilt)
- Cumulative reward < -100
- Maximum steps reached (5000)

### Hyperparameters

```yaml
trainer_type: ppo
time_horizon: 128
max_steps: 10.0e6
batch_size: 128
buffer_size: 2048
learning_rate: 3.0e-4
learning_rate_schedule: linear
epsilon: 0.2
beta: 1e-3
lambda: 0.99

network_settings:
  hidden_units: 128
  num_layers: 2
  num_epoch: 3
  vis_encode_type: simple
```

---

## 🚀 Training Process

### Parallel Training Strategy

**Acceleration Multipliers:**
- 4 training environments running simultaneously
- 5 agents per environment = **20 concurrent agents**
- Time scale: 20x speed → **400x total training speedup**

### Progressive Curriculum

**Phase 1: Open Environment**
- Single agent in unbounded space
- Goal: Learn basic movement and reward collection
- Challenge: Initial over-optimization (agents exploiting reward system)

**Phase 2: Simple Closed Loop**
- Circular track with sidewalk boundaries
- Goal: Learn turning and boundary avoidance
- Outcome: Improved path following

**Phase 3: Complex Intersection**
- Cross-shaped intersection with multiple paths
- Goal: Handle branching decisions
- Challenge: 90° field-of-view limitation

**Phase 4: Final Configuration** ✅
- Complex intersection environment
- Enhanced 120° field-of-view (vs 90°)
- Optimized grid sensor geometry
- Multiple interacting agents
- **Result:** Significant accident reduction and improved coordination

### Training Statistics

- **Total Simulations:** 28 experiments conducted
- **Successful Training Runs:** 4 major experimental phases
- **Training Duration:** 10 million steps (configurable)
- **Key Metrics Tracked:**
  - Cumulative episode rewards
  - Episode length (survival time)
  - Sidewalk collisions
  - Vehicle accidents
  - Reward ball collections
  - Lane violations

---

## 📊 Results

### Performance Improvements

**Accident Reduction:**
- Significant decrease in vehicle-to-vehicle collisions over training period
- Sidewalk collision rate substantially reduced
- Episode length increased (agents survive longer)

**Learning Progress:**
- Cumulative rewards showed continuous improvement
- **No asymptote observed** → potential for further optimization
- Agents learned complex behaviors:
  - Intersection navigation
  - Multi-vehicle coordination
  - Path following with lane discipline
  - Collision avoidance

### Key Findings

1. **Simplified Perception Works:** Grid sensor's pixelated vision sufficient for navigation
2. **Collective Behavior Emerges:** Agents coordinate without explicit communication
3. **Gradual Curriculum Essential:** Progressive difficulty prevents confusion
4. **Sensor Configuration Critical:** 120° field-of-view vs 90° made significant difference
5. **Reward Balance Matters:** Fine-tuning reward values crucial for desired behavior

---

## 🛠️ Technical Stack

### Development Environment

| Component | Version |
|-----------|---------|
| Unity Editor | 2021.3.16f1 LTS |
| ML-Agents Toolkit | 2.3.0-exp.3 |
| Python | 3.8.8 |
| ml-agents (Python) | 0.29.0 |
| ml-agents-envs | 0.29.0 |
| PyTorch | 1.13.1+cu117 |
| Communicator API | 1.5.0 |
| TensorBoard | (for training visualization) |

### System Requirements

**Hardware Used:**
- CPU: Intel Core i7-7700k @ 4.20 GHz
- RAM: 16GB
- GPU: NVIDIA GeForce GTX 1050
- OS: Windows 10

**Note:** Training is GPU-accelerated. CUDA-compatible GPU recommended for optimal performance.

### Dependencies

**Unity Packages:**
- ML-Agents Unity Package
- Bézier Path Creator (road system)
- Grid Sensor Package (custom implementation)
- TextMesh Pro (UI)

**Python Libraries:**
```bash
pip install mlagents==0.29.0
pip install torch==1.13.1+cu117
pip install tensorboard
```

---

## 📁 Project Structure

```
Assets/
├── _Scripts/
│   ├── Car/                      # Vehicle agent implementations
│   │   ├── TeamDriverAgent.cs    # Main RL agent
│   │   ├── MLDriverAgent.cs      # Alternative agent implementation
│   │   ├── CarController.cs      # Physics-based vehicle control
│   │   ├── CarPercepts.cs        # Collision detection system
│   │   └── ICarAgent.cs          # Agent interface
│   │
│   ├── MLTraining/               # Training orchestration
│   │   └── MLTrainingScene.cs    # Multi-agent training manager
│   │
│   ├── Pathing/                  # Navigation system
│   │   ├── NodePath.cs           # Waypoint path definitions
│   │   ├── PathCrawler.cs        # Path following behavior
│   │   └── PathDebugDrawer.cs    # Visualization tools
│   │
│   ├── Sensors/                  # Perception system
│   │   └── Grid sensor implementations
│   │
│   ├── Roads/                    # Road infrastructure
│   │   ├── RoadPiece.cs
│   │   ├── FourWayIntersection.cs
│   │   └── ThreeWayIntersection.cs
│   │
│   └── TrafficSignals/           # Traditional traffic control (unused in final)
│
├── Resources/Prefabs/            # Reusable game objects
│   ├── GridSensorTrain/          # Training environment prefabs
│   └── RoadPieces/               # Modular road components
│
├── config/                       # ML-Agents training configurations
│   └── Trained models (.onnx, .pt)
│
├── Demonstrations/               # Recorded agent behaviors
│   └── *.demo files
│
├── Scenes/                       # Unity scenes
│   └── Training environments
│
└── Models/                       # 3D assets
    └── Low-poly vehicle models
```

---

## 🎮 Usage

### Setting Up Training

1. **Clone the repository:**
```bash
git clone <repository-url>
cd SmartCity
```

2. **Open in Unity:**
   - Launch Unity Hub
   - Add project (Unity 2021.3.16f1)
   - Open main training scene: `Assets/_TrainRoads/TrainScene_4.unity`

3. **Configure Python environment:**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install ML-Agents
pip install mlagents==0.29.0
```

4. **Start training:**
```bash
mlagents-learn Assets/config/MLDriver.yaml --run-id=smart-city-run-1
```

5. **Monitor training:**
```bash
tensorboard --logdir results/
```
Navigate to `http://localhost:6006` to view real-time training metrics.

### Testing Trained Models

1. Load trained model (`.onnx` file) into agent's Behavior Parameters component
2. Set Behavior Type to "Inference Only"
3. Press Play in Unity Editor
4. Observe autonomous navigation behavior

### Manual Control (Debugging)

1. Set Behavior Type to "Heuristic Only"
2. Use keyboard controls:
   - **W/S:** Forward/Reverse
   - **A/D:** Steering
3. Helpful for validating environment setup

---

## 📖 Key Learnings & Best Practices

### From 28 Experiments

1. **Start Simple:** Begin with minimal parameters. Complex environments overwhelm untrained agents.

2. **Tune Vehicle Physics First:** Ensure car controller is responsive before ML training. Acceleration delays caused thousands of wasted training iterations.

3. **Visualize Agent Perception:** Understanding what the agent "sees" is crucial for debugging unexpected behaviors.

4. **Minimize Neural Network Inputs:** Fewer observation parameters = faster convergence. Started with many, reduced to 3 critical ones.

5. **Balance Discrete vs Continuous Actions:** Movement (forward/back) as discrete, steering as continuous worked best.

6. **Test Sensor Configuration Manually:** Validate grid sensor settings before long training runs. Small configuration errors can invalidate entire simulations.

7. **Beware of Reward Exploitation:** Agents will find creative ways to maximize rewards that don't align with intended behavior. Example: racing to first reward then jumping off platform to end episode quickly.

8. **Curriculum Learning is Essential:** Progressive difficulty from open space → simple loop → intersection → multi-agent intersection.

---

## 🔬 Future Research Directions

- [ ] Increase to 100+ simultaneous agents
- [ ] More complex urban scenarios (multi-lane highways, roundabouts)
- [ ] Vehicle-to-vehicle communication protocols
- [ ] Pedestrian integration
- [ ] Dynamic obstacle avoidance (moving objects)
- [ ] Weather and visibility variations
- [ ] Transfer learning to different city layouts
- [ ] Real-world deployment considerations
- [ ] Compare with traditional traffic signal systems
- [ ] Energy efficiency optimization

---

## 📄 Publications & References

### Thesis Documentation

- **[Master's Defense Presentation (English)](https://docs.google.com/presentation/d/1NOcKFnHB3iI_q2XVUcLx_OZ09nRyxu88X45hr9WY3zg)** - Visual overview of research and results
- **Full Thesis (Georgian):** Available upon request for academic purposes

### Key References

1. **Michael Lanham** - "Learn Unity ML-Agents Fundamentals of Unity Machine Learning" (2018), pp. 73-78
2. **Miguel Morales** - "Grokking Deep Reinforcement Learning" (2020)
3. **Chip Huyen** - "Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications" (2022), pp. 1-21
4. **Adam Streck** - "Reinforcement Learning a Self-driving Car AI in Unity" ([Article](https://towardsdatascience.com/reinforcement-learning-a-self-driving-car-ai-in-unity-60b0e7a10d9e))
5. [Unity ML-Agents GitHub](https://github.com/Unity-Technologies/ml-agents)
6. [TensorFlow Documentation](https://www.tensorflow.org/)
7. [Unity ML-Agents: Hummingbirds Course](https://learn.unity.com/course/ml-agents-hummingbirds)
8. **Ryan McLarty** - "Creating a Road System" ([Blog Post](https://ryanmclarty.me/capstone-project-update-1/))

### Assets Used
- [Bézier Path Creator](https://assetstore.unity.com/packages/tools/utilities/b-zier-path-creator-136082)
- [Favoured Cars Pack - Low Poly (FREE)](https://assetstore.unity.com/packages/3d/vehicles/land/favoured-cars-pack-low-poly-free-226458)
- [Grid Sensor](https://github.com/mbaske/grid-sensor)

---

## 🤝 Contributing

This is an academic research project. While direct contributions may be limited, feedback and discussions are welcome:

- Report issues or bugs
- Suggest improvements to training methodology
- Share results from adapting this work
- Propose new experimental scenarios

---

## 📧 Contact

**Author:** Nikoloz Astamidze  
**Institution:** Georgian Technical University  
**Program:** Master's in Computer Science  
**Year:** 2023

For academic inquiries or collaboration opportunities, please reach out through the university.

---

## 📜 License

This project is released under an **Academic License**. 

- ✅ Non-commercial use permitted
- ✅ Academic research and study
- ✅ Educational purposes
- ⚠️ Commercial use requires permission
- ⚠️ Proper citation required for derivative works

When referencing this work, please cite:
```
Astamidze, N. (2023). Autonomous Vehicle Simulation in Urban Environment. 
Master's Thesis, Georgian Technical University, Tbilisi, Georgia.
```

---

## 🙏 Acknowledgments

Special thanks to:
- **Professor Merab Akhobadze** - Thesis supervisor
- **Professor Mariam Chkhaidze** - Thesis reviewer
- **Diana Astamidze** - Proofreading and motivation support
- **Georgian Technical University** - Providing the academic framework
- **Unity ML-Agents Team** - For the powerful toolkit
- **Open-source community** - For the various assets and tools used

This research represents the potential of combining nature-inspired algorithms with modern machine learning to solve real-world transportation challenges. The journey from theory to working simulation validated that autonomous vehicles can indeed coordinate without centralized traffic control.

---

**⭐ If this project helps your research or learning, please consider starring the repository!**

---

*Built with Unity, powered by ML-Agents, inspired by nature.* 🚗🤖🌿
