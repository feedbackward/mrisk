#!/bin/bash

ALGO_ANCILLARY="$2"
ALGO_MAIN="$3"
BATCH="8"
DATA="$1"
ENTROPY="168590470585383178607324564460699024728"
EPOCHS="30"
LOSS="logistic"
MODEL="linreg_multi"
SIGMA="$5"
STEP="0.01"
TASK="$4"
TRIALS="10"


python "learn_driver.py" --algo-ancillary="$ALGO_ANCILLARY" --algo-main="$ALGO_MAIN" --batch-size="$BATCH" --data="$DATA" --entropy="$ENTROPY" --loss="$LOSS" --model="$MODEL" --num-epochs="$EPOCHS" --num-trials="$TRIALS" --sigma="$SIGMA" --step-size="$STEP" --task-name="$TASK"
