#!/bin/bash

## A simple loop over all the datasets specified.
for arg
do
    bash remote_learn_off_run.sh "${arg}" "SGD" "Ave"
    bash remote_learn_run.sh "${arg}" "SGD" "Ave" "zero" "0.0"
    bash remote_learn_run.sh "${arg}" "SGD" "Ave" "low" "0.1"
    bash remote_learn_run.sh "${arg}" "SGD" "Ave" "med" "1.0"
    bash remote_learn_run.sh "${arg}" "SGD" "Ave" "high" "10.0"
    bash remote_learn_run.sh "${arg}" "SGD" "Ave" "inf" "1000000000.0"
done
