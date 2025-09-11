#!/usr/bin/env bash
set -euo pipefail
python -m bbml.train.train_rank +config=configs/train/ranking.yaml
