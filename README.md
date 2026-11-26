<!--- Note: right-click->Open With->Markdown Preview for ideal viewing -->
# Kāhu Desktop Environment

    git clone --recurse-submodules git@github.com:Resonint/kahu-desktop.git

Run all cells in *install.ipynb* to install *matipo* python library and create the default configuration.
Restart the Jupyterlab session after installing.

When connecting to Kāhu by direct ethernet cable, assign a manual IP address of 192.168.137.1 to the ethernet device (netmask: 255.255.255.0).

Kāhu may also be connected to a router via ethernet cable. In this case check the router admin page for the automatically assigned IP address.

This IP address must be set in matipo-config.yaml as the `device_address`. Restart the Jupyterlab session after modifying the `matipo-config.yaml` file.
