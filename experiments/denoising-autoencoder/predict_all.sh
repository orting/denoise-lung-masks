#!/bin/bash
v0_checkpoint=lightning_logs/version_0/checkpoints/
v1_checkpoint=lightning_logs/version_1/checkpoints/
v2_checkpoint=lightning_logs/version_2/checkpoints/

python3 predict.py ${v0_checkpoint} out/v0/no-corruption/ 0
python3 predict.py ${v0_checkpoint} out/v0/with-corruption/ 0 --with-corruption

python3 predict.py ${v1_checkpoint} out/v1/no-corruption/ 1
python3 predict.py ${v1_checkpoint} out/v1/with-corruption/ 1 --with-corruption

python3 predict.py ${v2_checkpoint} out/v2/no-corruption/ 2
python3 predict.py ${v2_checkpoint} out/v2/with-corruption/ 2 --with-corruption
