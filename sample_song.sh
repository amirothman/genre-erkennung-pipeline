#!/bin/bash

mkdir dataset/test/pop
ls raw/pop | shuf | head -40 | sed "s/^/mv\ raw\/pop\//" | sed "s/$/\ dataset\/test\/pop/" > move.sh
sh move.sh
mv raw/pop dataset/train

mkdir dataset/test/system
ls raw/system | shuf | head -20 | sed "s/^/mv\ raw\/system\//" | sed "s/$/\ dataset\/test\/system/" > move.sh
sh move.sh
mv raw/system dataset/train

mkdir dataset/test/hiphop
ls raw/hiphop | shuf | head -40 | sed "s/^/mv\ raw\/hiphop\//" | sed "s/$/\ dataset\/test\/hiphop/" > move.sh
sh move.sh
mv raw/hiphop dataset/train
