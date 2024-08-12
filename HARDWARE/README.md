# Jetson Nano based Microphone Array + Depth Camera

Install instructions for the Jetson Nano:
1. Connect the X-TAG 
2. Follow the steps on the [Get Started With Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit).
3. Install the RealSense SDK the easy way [Install RealSense Camera in 5 minutes – Jetson Nano](https://jetsonhacks.com/2019/12/22/install-realsense-camera-in-5-minutes-jetson-nano/).
4. Git Clone recorder file. TODO.
5. Compile the recorder file:
```
/usr/local/cuda/bin/nvcc realsense_pc.cpp -lrealsense2 -o realsense_pc
```
6. Run the recorder file:
```
./realsense_pc
```
7. An audio file `a.wav` and a point cloud file `Pointcloud.ply` will be in the home directory.
