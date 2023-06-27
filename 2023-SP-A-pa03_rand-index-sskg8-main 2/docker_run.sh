#!/bin/bash
# Execute this whenever you want to work on the assignment,
# and run grade.sh

# sudo docker run --interactive --tty --mount type=bind,source="$(pwd)"/,target=/your_code "$(basename "$(pwd)")" fish
sudo docker run --interactive --tty --mount type=bind,source="$(pwd)"/,target=/your_code pa02_rand-index fish
# TODO maybe run fish inside of tmux, include various dotfiles like nice vim config too? 
