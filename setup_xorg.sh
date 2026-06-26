#!/bin/bash
# Use this script to set up Xorg server powered by NVIDIA, and connect it directly from browser.
# 1. Install Velda CLI on your laptop: 
#      curl https://velda.cloud/install.sh | bash
# 2. Setup Xorg:
#      velda run --instance [instance-name] sh -c 'curl https://raw.githubusercontent.com/velda-io/examples/refs/heads/main/setup_xorg.sh | bash'
# 3. Start Xorg & VNC server: 
#      velda run --instance [instance-name] -P l40s-1-16d -s xorg ./start.sh
# 4. Connect: https://6080-xorg-$instance_id.ne.velda.cloud/vnc.html The instance ID will be printed in launch step above.

# Full serverless mode coming soon: Access from the URL will directly start the server, no more step 3 required.

sudo apt-get update && sudo apt-get install -y \
    xserver-xorg \
    x11vnc \
    xfce4 \
    xfce4-goodies \
    mesa-utils \
    xterm \
    dbus-x11 \
    novnc \
    websockify

echo 'Section "ServerLayout"
    Identifier     "Layout0"
    Screen      0  "Screen0" 0 0
EndSection

Section "ServerFlags"
    Option "AutoAddDevices" "False"
EndSection

Section "InputDevice"
    Identifier     "Mouse0"
    Driver         "void"
    Option         "CorePointer" "True"
EndSection

Section "Device"
    Identifier     "Device0"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
    Option         "AllowEmptyInitialConfiguration" "True"
    Option         "ConnectedMonitor" "DFP-0"
EndSection

Section "Screen"
    Identifier     "Screen0"
    Device         "Device0"
    DefaultDepth    24
    Option         "MetaModes" "DFP-0: 1920x1080 +0+0"
    SubSection     "Display"
        Depth       24
        Modes      "1920x1080"
    EndSubSection
EndSection' | sudo tee /etc/X11/xorg.conf >/dev/null

cat << EOF | sudo tee /usr/share/X11/xorg.conf.d/10-nvidia.conf > /dev/null
Section "OutputClass"
    Identifier "nvidia"
    MatchDriver "nvidia-drm"
    Driver "nvidia"
    Option "AllowEmptyInitialConfiguration"
    ModulePath "/usr/lib/x86_64-linux-gnu/nvidia/xorg"
EndSection
EOF
sudo mkdir -p /usr/lib/x86_64-linux-gnu/nvidia/xorg
sudo ln -s /var/nvidia/lib/libglxserver_nvidia.so1 /usr/lib/x86_64-linux-gnu/nvidia/xorg/libglxserver_nvidia.so
sudo ln -s /var/nvidia/lib/nvidia_drv.so /usr/lib/x86_64-linux-gnu/nvidia/xorg/nvidia_drv.so

cat << EOF > ~/start.sh
#!/bin/bash
# Usage: vrun -P l40s-1-16d -s xorg ./start.sh
# Replace l40s-1-16d with desired GPU pool. Note A100/H100/H200 do not support rendering.
sudo LD_LIBRARY_PATH=/var/nvidia/lib Xorg :0 -noreset -novtswitch -sharevts +extension GLX +extension RANDR +extension RENDER &
sleep 2

echo "Starting XFCE, logs at /tmp/xfce.log"
DISPLAY=:0 startxfce4 > /tmp/xfce.log 2>&1 &
sleep 2

echo "Starting x11vnc, logs at /tmp/x11vnc.log"
x11vnc -display :0 -forever -shared -nopw -rfbport 5901 -listen localhost > /tmp/x11vnc.log 2>&1 &

sleep 2

echo "Starting websockify, logs at /tmp/websockify.log"
websockify --web /usr/share/novnc localhost:6080 localhost:5901 >/tmp/websockify.log 2>&1 &

instance_id=\$(grep 'instance:' /run/velda/velda.yaml  | grep -o '[0-9]*')
echo "URL: https://6080-xorg-\$instance_id.ne.velda.cloud/vnc.html"

wait
EOF

chmod +x ~/start.sh
