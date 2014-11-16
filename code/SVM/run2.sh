#!/bin/bash
. DATA.TXT
START=$(date +%s.%N)
echo 'python Implementation/svm_proc2.py $train train $test test'
python Implementation/svm_proc2.py $train train $test test
echo './Implementation/svm_learn -v 0 -b 0 -c $c train model'
./Implementation/svm_learn -v 0 -b 0 -c $c train model
echo './Implementation/svm_classify test model $c.prediction'
./Implementation/svm_classify test model $c.prediction
END=$(date +%s.%N)
DIFF=$(echo $END - $START | bc -l)
echo $DIFF
echo 'score is saved to `$c.prediction`'