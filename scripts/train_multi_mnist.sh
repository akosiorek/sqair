#!/usr/bin/env bash
cd sqair

python scripts/experiment.py\
    --results_dir results\
    --run_name multi_mnist\
    --data_config configs/seq_mnist_data.py\
    --model_config configs/mlp_mnist_model.py\
    --seq_len 3\
    --stage_itr 100000

cd -
