#!/bin/bash
python3 estimate_volume.py ../denoising-autoencoder/out/v0/no-corruption/train/* --plot-path ../denoising-autoencoder/results/v0-no-corruption-train-volumes.png
python3 estimate_volume.py ../denoising-autoencoder/out/v0/no-corruption/validation/* --plot-path ../denoising-autoencoder/results/v0-no-corruption-validation-volumes.png
python3 estimate_volume.py ../denoising-autoencoder/out/v0/no-corruption/test/* --plot-path ../denoising-autoencoder/results/v0-no-corruption-test-volumes.png
python3 estimate_volume.py ../denoising-autoencoder/out/v0/with-corruption/train/* --plot-path ../denoising-autoencoder/results/v0-with-corruption-train-volumes.png
python3 estimate_volume.py ../denoising-autoencoder/out/v0/with-corruption/validation/* --plot-path ../denoising-autoencoder/results/v0-with-corruption-validation-volumes.png
python3 estimate_volume.py ../denoising-autoencoder/out/v0/with-corruption/test/* --plot-path ../denoising-autoencoder/results/v0-with-corruption-test-volumes.png
