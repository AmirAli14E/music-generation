#!/bin/bash
pip install -r requirements.txt
apt-get install fluidsynth -qq
mkdir -p /usr/share/sounds/sf2/
wget https://musical-artifacts.com/artifacts/2744/FluidR3_GM.sf2 -O /usr/share/sounds/sf2/FluidR3_GM.sf2