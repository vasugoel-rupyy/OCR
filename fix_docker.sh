#!/bin/bash
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root"
  exit
fi
groupadd docker
usermod -aG docker $USER
newgrp docker
