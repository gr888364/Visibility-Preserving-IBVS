# Visibility-Preserving-IBVS

MATLAB simulation and C++ real-robot code for **visibility-preserving image-based visual servoing (P-IBVS)** with reference–increment coordination under random sampling intervals. The method combines an FOV-aware virtual reference generation strategy (**G<sub>s</sub>**) in the image plane with a neurodynamic displacement-increment shaping mechanism (**G<sub>t</sub>**) in the task space, while keeping the nominal IBVS control law. Experiments are carried out in MATLAB simulation and on a JAKA Zu5 manipulator with an eye-in-hand Intel RealSense D435 camera.

<p align="center">
  <img src="docs/videos/real_robot_pibvs.gif" width="760" alt="P-IBVS on the JAKA Zu5 manipulator">
  <br>
  <em>P-IBVS on the JAKA Zu5 manipulator (real time). Left: camera view with the feature trajectories (green) and the safety region (red). Right: robot motion.
  Full video: <a href="docs/videos/real_robot_pibvs.mp4">real_robot_pibvs.mp4</a></em>
</p>

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Key Features](#key-features)
- [Experimental Tests](#experimental-tests)
- [License](#license)

## Overview

<p align="center"><img src="docs/images/fig01_architecture.png" width="900" alt="Architecture of the proposed method"></p>

- **G<sub>s</sub> (FOV-aware virtual reference).** A virtual image-space reference is driven toward the desired features by an attractive field and pushed away from the image border by a repulsive field triggered by the observed features. The reference is integrated every step and projected into the image domain, so boundary corrections are introduced and released gradually.
- **Control law.** The nominal IBVS law tracks the virtual reference, with a sigmoid gain schedule from λ<sub>min</sub> to λ<sub>max</sub>.
- **G<sub>t</sub> (neurodynamic increment shaping).** The raw displacement increment v<sub>c,k</sub>Δt<sub>k</sub> is passed through a shunting model with state memory and soft saturation before execution, which suppresses jitter caused by random sampling intervals.

The repository also contains the three comparison methods used in the experiments: OG-VFVR-IBVS, MPC-IBVS and APF-IBVS.

## Dependencies

**MATLAB simulation**

- MATLAB R2020b or later
- Optimization Toolbox (`quadprog`, MPC-IBVS)
- Fuzzy Logic Toolbox (`mamfis`, used in the `VisualServo` constructor)
- [Machine Vision Toolbox for MATLAB](https://github.com/petercorke/machinevision-toolbox-matlab) and [Spatial Math Toolbox for MATLAB](https://github.com/petercorke/spatialmath-matlab) by P. Corke

**Real robot (C++)**

- Ubuntu with a C++14 compiler
- [ViSP](https://visp.inria.fr/) (visual features, AprilTag detection, display and plotting)
- [librealsense2](https://github.com/IntelRealSense/librealsense) (Intel RealSense D435)
- OpenCV
- JAKA robot SDK (`JAKAZuRobot.h`, `jktypes.h`)
- Eigen and [OsqpEigen](https://github.com/robotology/osqp-eigen) (MPC-IBVS only)

## Repository Structure

```plaintext
Visibility-Preserving-IBVS/
├── README.md
├── LICENSE                              # GNU LGPL v3.0
├── COPYING                              # GNU GPL v3.0 (referenced by the LGPL)
├── matlab/                              # MATLAB simulation
│   ├── C_IBVS_PF_NDRG.m                 # Comparison of OG-VFVR-IBVS, MPC-IBVS, APF-IBVS and P-IBVS, each without/with G_t
│   ├── IBVS.m                           # IBVS class: G_s, G_t, gain schedule and the comparison front-ends
│   ├── VisualServo.m                    # Visual-servo base class with random sampling intervals
│   ├── APF_Log_Direct_with_jitter.m     # APF-IBVS simulation without G_t
│   └── APF_Log_Direct_with_jitter_ND.m  # APF-IBVS simulation with G_t
├── real_robot/                          # C++ programs for the JAKA Zu5 + RealSense D435 platform
│   ├── p_ibvs/
│   │   ├── p_ibvs.cpp                   # P-IBVS, performance indices
│   │   ├── p_ibvs_nd.cpp                # P-IBVS + G_t, performance indices
│   │   ├── p_ibvs_lite.cpp              # P-IBVS, minimal version
│   │   └── p_ibvs_lite_nd.cpp           # P-IBVS + G_t, minimal version
│   ├── og_vfvr_ibvs/
│   │   ├── og_vfvr_ibvs.cpp             # OG-VFVR-IBVS, performance indices
│   │   └── og_vfvr_ibvs_nd.cpp          # OG-VFVR-IBVS + G_t, increment jitter
│   └── mpc_ibvs/
│       ├── mpc_ibvs.cpp                 # MPC-IBVS (OSQP), performance indices
│       └── mpc_ibvs_nd.cpp              # MPC-IBVS + G_t, increment jitter
└── docs/
    ├── images/                          # Experimental figures
    └── videos/                          # Real-robot recording (GIF preview and MP4)
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/gr888364/Visibility-Preserving-IBVS.git
cd Visibility-Preserving-IBVS
```

### 2. MATLAB simulation

Add the Machine Vision Toolbox and Spatial Math Toolbox to the MATLAB path, or copy their `Camera/`, `common/`, `vision/` and `spatial-math/` folders into `matlab/` (the scripts add these folders automatically when present). The `IBVS.m` and `VisualServo.m` in `matlab/` must take precedence over the toolbox versions.

```matlab
cd matlab
C_IBVS_PF_NDRG                    % four methods, each without and with G_t
APF_Log_Direct_with_jitter        % APF-IBVS without G_t
APF_Log_Direct_with_jitter_ND     % APF-IBVS with G_t
```

`C_IBVS_PF_NDRG` writes figures, a metrics table (`unified_four_method_metrics.csv`) and the run histories to `matlab/compare_outputs/unified_four_method_compare/`. All runs use the same camera model, initial pose, desired features and random sampling sequence (25–35 ms, seed 42).

### 3. Real robot

Each `.cpp` file in `real_robot/` is a self-contained program (`main()`). Build it against ViSP, librealsense2, OpenCV and the JAKA SDK (MPC-IBVS additionally needs Eigen and OsqpEigen). Before running:

1. Set `JAKA_IP` to the IP address of the robot controller.
2. Set the camera intrinsics (`vpCameraParameters cam(...)`), the tag size (`TAG_SIZE`) and the hand–eye matrix (`raw_eMc`) for your setup.
3. Place the AprilTag 36h11 target in view. The program shows the camera image, the target box and the feature trajectories; a mouse click in the image window toggles servoing between START and PAUSE.

> **Safety:** these programs send motion commands to a real manipulator. Test with low speed limits and keep the emergency stop within reach.

## Key Features

- **FOV-aware virtual reference (G<sub>s</sub>)**: attractive and repulsive fields act on an image-space reference instead of the control command, keeping the features inside the image under large initial displacements.
- **Neurodynamic increment shaping (G<sub>t</sub>)**: a shunting model with bounded state regularizes the executed displacement increments under random sampling intervals; it can be added to other IBVS controllers without changing their control laws.
- **Gain schedule**: the IBVS gain rises from λ<sub>min</sub> to λ<sub>max</sub> along a sigmoid, avoiding an aggressive start.
- **Comparison methods**: OG-VFVR-IBVS, MPC-IBVS and APF-IBVS, each runnable without and with G<sub>t</sub> under the same random sampling sequence.
- **Real-robot implementation**: AprilTag 36h11 detection with a RealSense D435, separate vision and robot-control threads, and joint-space servo commands to the JAKA Zu5 through the JAKA SDK.

## Experimental Tests

### 1. Simulation

Four coplanar points are observed by a virtual 1024 × 1024 pinhole camera; the target starts close to the image border and the sampling interval is drawn from 25–35 ms.

<p align="center">
  <img src="docs/images/fig04_ibvs_fov_failure.png" width="400"><br>
  <em>IBVS without FOV handling: the features leave the image.</em>
</p>

<p align="center">
  <img src="docs/images/fig05_feature_trajectories.png" width="640"><br>
  <em>Feature trajectories: (a) OG-VFVR-IBVS, (b) MPC-IBVS, (c) APF-IBVS, (d) P-IBVS.</em>
</p>

| Method | N<sub>conv</sub> | P<sup>tr</sup><sub>II</sub> | P<sup>rot</sup><sub>II</sub> | σ<sub>Δv</sub> | σ<sub>Δ²s</sub> |
| :-- | --: | --: | --: | --: | --: |
| OG-VFVR-IBVS | 489 | 0.696 | 0.283 | 0.016 | 9.755 |
| MPC-IBVS | 694 | 0.250 | 0.250 | 0.022 | 1.188 |
| APF-IBVS | 785 | 1.538 | 0.579 | 0.186 | 4.698 |
| **P-IBVS** | **329** | **0.008** | **0.003** | **0.009** | **1.020** |

<details>
<summary><b>Jacobian condition number, feature errors and Cartesian velocity</b></summary>
<p align="center">
  <img src="docs/images/fig06_jacobian_condition.png" width="640"><br><br>
  <img src="docs/images/fig07_feature_error.png" width="640"><br><br>
  <img src="docs/images/fig08_cartesian_velocity.png" width="640">
</p>
</details>

<details>
<summary><b>Executed displacement increments without (blue) and with (red) G<sub>t</sub></b></summary>
<p align="center">
  <img src="docs/images/fig09_displacement_increments.png" width="760">
</p>
</details>

### 2. Real-robot experiments

The platform consists of a JAKA Zu5 6-DoF manipulator, an eye-in-hand Intel RealSense D435 camera (640 × 480), a planar AprilTag 36h11 target and a host PC running detection and control.

<p align="center">
  <img src="docs/images/fig10_platform.png" width="420" alt="Experimental platform">
</p>

Starting near the image border, conventional IBVS drives the target out of view, while P-IBVS keeps it visible and converges.

<p align="center">
  <img src="docs/images/fig11_real_ibvs.png" width="380">
  <img src="docs/images/fig12_real_pibvs.png" width="380"><br>
  <em>Left: conventional IBVS at t = 0 s and 2 s. Right: P-IBVS at t = 0 s, 5 s and 10.5 s.</em>
</p>

<p align="center">
  <img src="docs/images/fig13_real_final_views.png" width="760"><br>
  <em>Final camera views and feature trajectories: (a) P-IBVS with G<sub>t</sub>, (b) OG-VFVR-IBVS, (c) MPC-IBVS.</em>
</p>

<p align="center">
  <img src="docs/images/fig14_real_velocity.png" width="760"><br>
  <em>Camera velocity commands: (a) P-IBVS, (b) OG-VFVR-IBVS, (c) MPC-IBVS.</em>
</p>

<p align="center">
  <img src="docs/images/fig15_real_increments.png" width="760"><br>
  <em>Raw (left) and shaped (right) displacement increments with G<sub>t</sub>: (a) P-IBVS, (b) OG-VFVR-IBVS, (c) MPC-IBVS.</em>
</p>

| Method | T<sub>conv</sub> (s) | N<sub>conv</sub> | P<sup>tr</sup><sub>II</sub> | P<sup>rot</sup><sub>II</sub> | σ<sub>Δv</sub> | σ<sub>Δ²s</sub> | σ<sub>Δξ</sub> raw / shaped |
| :-- | --: | --: | --: | --: | --: | --: | :-- |
| OG-VFVR-IBVS | 11.205 | 322 | 0.378 | 1.521 | 0.041 | 2.160 | 0.022452 / 0.002775 |
| MPC-IBVS | 74.982 | 2029 | 0.250 | 0.250 | 0.012 | 1.074 | 0.000959 / 0.000463 |
| **P-IBVS** | **10.554** | **307** | **0.101** | **0.137** | **0.011** | **0.903** | 0.002268 / 0.000505 |

## License

This project is licensed under the [GNU Lesser General Public License v3.0](LICENSE). `IBVS.m` and `VisualServo.m` extend classes of the Machine Vision Toolbox for MATLAB by Peter Corke, which is distributed under the GNU LGPL.
