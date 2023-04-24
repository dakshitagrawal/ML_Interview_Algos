#!/bin/bash

mkdir output

python neuralnet.py data/small_train_data.csv data/small_validation_data.csv output/small_train_out.labels output/small_validation_out.labels output/small_metrics_out.txt 2 4 2 0.1

cat output/small_metrics_out.txt

# After running, the output should be:
# epoch=0 crossentropy(train): 1.994695028028554
# epoch=0 crossentropy(validation): 2.010686378337308
# epoch=1 crossentropy(train): 1.912184059993547
# epoch=1 crossentropy(validation): 1.9443269427900587
# error(train): 0.782
# error(test): 0.83
