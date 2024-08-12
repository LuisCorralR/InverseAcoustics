# Jetson Nano based Microphone Array + Depth Camera

This install instructions requieres a Ubuntu 22.04.3 LTS 64 bit computer. Although, all software used is fully compatible with Windows and macOS. Start by flashing the firmware on the xCORE-200 explorerKIT board.

1. Download the `HARDWARE` folder to your home `~/` directory.
2. Open a terminal and install OpenJDK 8:
```
sudo apt install openjdk-8-jre
sudo apt install openjdk-8-jdk
```
3. Download [xTimeComposer 14.4.1](https://www.xmos.com/file/xtimecomposer-community_14-linux64-installer?version=all) to your home `~/` directory (you will need a user account).
4. From your home `~/` directory, extract the files from the terminal (check the file name):
```
tar xvf xTIMEcomposer-Community_14-Linux64-Installer_Community_14_4_1.tgz
```
5. Navigate to the parent folder (check the folder names):
```
cd XMOS/xTIMEcomposer/Community_14.4.1
```
6. Connect the xTAG v3.0 to the xCORE-200 explorerKIT board xSYS connector and the micro USB to an available USB port on your computer and run the following scripts to activate the USB driver:
```
sudo ./scripts/setup_xmos_devices.sh
sudo ./scripts/check_xmos_devices.sh
```
7. Check the connection of the xTAG v3.0:
```
xflash -l
```
8. Flash de XMOS executable file `USB_MIC_ARRAY_16_SI5351.xe`:
```
source SetEnv
xflash ~/HARDWARE/USB_MIC_ARRAY_16_SI5351.xe --boot-partition-size 0x80000
```

For the Jetson nano installation:

1. Follow the steps on the [Get Started With Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit) to write the SD card and boot the first time. Make sure the device has a IP adress in the same range as your computer (a simple home router will work).
2. Copy the recorder file to the jetson nano home `~/` folder:
```
scp ~/HARDWARE/recorder.cpp nanouser@nanoipaddress:~/
```
3. Connect to the jetson nano via SSH (a nice tutorial is provided [here](https://www.digikey.com/en/maker/projects/getting-started-with-the-nvidia-jetson-nano-part-1-setup/2f497bb88c6f4688b9774a81b80b8ec2)).
```
ssh nanouser@nanoipadress
```
4. Once connected, install the RealSense SDK the easy way [Install RealSense Camera in 5 minutes – Jetson Nano](https://jetsonhacks.com/2019/12/22/install-realsense-camera-in-5-minutes-jetson-nano/).
5. Compile the recorder file with the nvcc CUDA compiler (g++ will work too):
```
/usr/local/cuda/bin/nvcc realsense_pc.cpp -lrealsense2 -o recorder
```
6. Run the recorder file:
```
./recorder
```
7. An audio file `mic_array_audio.wav` and a point cloud file `realsense_pointcloud.ply` will be in the home directory.
8. A MATLAB post-procesing audio and point cloud files `post-processing.m` is included to obtain results.
