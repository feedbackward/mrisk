#!/bin/bash

ALGO_ANCILLARY="SGD"
ALGO_MAIN="Ave"
BATCH="8"
DATA="mnist"
ENTROPY="168590470585383178607324564460699024728"
EPOCHS="30"
LOSS="logistic"
MODEL="linreg_multi"
SIGMA="0.1"
STEP="0.01"
TASK="low"
TRIALS="10"


python "learn_driver.py" --algo-ancillary="$ALGO_ANCILLARY" --algo-main="$ALGO_MAIN" --batch-size="$BATCH" --data="$DATA" --entropy="$ENTROPY" --loss="$LOSS" --model="$MODEL" --num-epochs="$EPOCHS" --num-trials="$TRIALS" --sigma="$SIGMA" --step-size="$STEP" --task-name="$TASK"
