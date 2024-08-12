# Jetson Nano based Microphone Array + Depth Camera

This install instructions includes commands run from a Ubuntu 22.04.3 LTS computer.

1. Download [xTimeComposer 14.4.1](https://www.xmos.com/file/xtimecomposer-community_14-linux64-installer?version=all) to your home `~/` directory.
2. From your home `~/` directory, untar the file:
```
tar xvf xTIMEcomposer-Community_14-Linux64-Installer_Community_14_4_1.tgz
```
3. Navigate to the parent folder:
```
cd XMOS/xTIMEcomposer/Community_14.4.1
```
4. Connect the X-TAG to the XMOS DSP board and to an available USB port. Flash de XMOS executable:
```
Source SetEnv
xflash ~/file.xe
```
3. Follow the steps on the [Get Started With Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit) to write the SD card and boot the first time.
4. Conect 

4. Install the RealSense SDK the easy way [Install RealSense Camera in 5 minutes – Jetson Nano](https://jetsonhacks.com/2019/12/22/install-realsense-camera-in-5-minutes-jetson-nano/).
5. Git Clone recorder file. TODO.
6. Compile the recorder file:
```
/usr/local/cuda/bin/nvcc realsense_pc.cpp -lrealsense2 -o realsense_pc
```
7. Run the recorder file:
```
./realsense_pc
```
8. An audio file `a.wav` and a point cloud file `Pointcloud.ply` will be in the home directory.
