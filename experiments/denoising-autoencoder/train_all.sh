#!/bin/bash

for version in 0 1 2
do
    python3 train.py ${version}
done
