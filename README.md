# Description 

This package provides a collection of installation, start and model scripts for using the NOVA Server package in combination with the NOVA annotation tool.


# Installation

## Windows
Run the `install_windows.cmd` to create a virtual environment and install all necessary dependencies. 

## Linux
Not yet officially supported

## Mac OS
Not yet officially supported

# Usage
Active the created virtual environment and run  .\nova_server\nova_backend.py using the following arguments:

  `--host`: The ip address to bin the server to. Usually the ip of your server or 0.0.0.0
  `--port`:  The port to listen for incoming commands
  `--cml_dir`: The coorporative machine learning directory as specified in NOVA
  `--data_dir`: The directory where your training data resides as specified in NOVA

You can als edit the `run_server.cmd` to refelect those settings for convenience.
