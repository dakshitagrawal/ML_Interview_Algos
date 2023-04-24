#!/bin/bash

train_data=data/train_data.tsv
valid_data=data/valid_data.tsv
test_data=data/test_data.tsv
form_train_data=output/formatted_train.tsv
form_valid_data=output/formatted_valid.tsv
form_test_data=output/formatted_test.tsv
out_train_labels=output/train_out.labels
out_test_labels=output/test_out.labels
metrics=output/metrics_out.txt
epochs=500
lr=0.00001

mkdir output

python feature.py $train_data $valid_data $test_data data/dict.txt data/word2vec.txt $form_train_data $form_valid_data $form_test_data 1

python lr.py --train_data $form_train_data --val_data $form_valid_data --test_data $form_test_data --train_out $out_train_labels --test_out $out_test_labels --metrics_out $metrics --num_epoch $epochs --learning_rate $lr

cat output/metrics_out.txt

# After running above, the metrics out file should have:
# error(train): 0.042500
# error(test): 0.150000
# loss(train): 0.342576
# loss(test): 0.462652
