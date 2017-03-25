#!/bin/bash


source /media/buckler/DataSSD/Phd/fall_detection/virtualenv/envkeras_py35/bin/activate


 
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python /media/buckler/DataSSD/Phd/fall_detection/framework/autoencoder_fall_detection/main_experiment.py --root-path "/media/buckler/DataSSD/Phd/fall_detection/framework/autoencoder_fall_detection" --exp-index 2 --trainset-list "['trainset.lst']" --case case6 --test-list-names "['testset_1.lst','testset_2.lst','testset_3.lst','testset_4.lst']" --dev-list-name "['devset_1.lst','devset_2.lst','devset_3.lst','devset_4.lst']" --input-type spectrograms --cnn-input-shape [1,129,197] --conv-layers-numb 3 --kernels-number [8,4,4] --pool-type all --kernel-shape [[4,4],[4,4],[4,4]] --strides [[3,3],[3,3],[3,3]] --max-pool-shape [[2,2],[2,2],[2,2]] --cnn-init glorot_uniform --cnn-conv-activation tanh --cnn-dense-activation tanh --border-mode same --cnn-w-reg None --cnn-b-reg None --cnn-act-reg None --cnn-w-constr None --cnn-b-constr None --d-w-reg None --d-b-reg None --d-act-reg None --d-w-constr None --d-b-constr None --dense-layers-numb 0 --dense-shape [] --epoch 2 --batch-size 0.113312594265 --optimizer adam --loss msle --dropout --drop-rate 0.468743227282 --learning-rate 0.000254471647737 --patiance 40 --aucMinImp 0.0001 --logging --fit-net