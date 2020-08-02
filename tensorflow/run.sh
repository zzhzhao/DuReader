

#注意三个文件都要，dev_file可以是测试集，最后计算分数None
python ./tensorflow/run.py --prepare \
                    --train \
                    --algo BIDAF \
                    --epoch 10 \
                    --max_q_len 80 \
                    --max_a_len 80 \
                    --max_p_num 1 \
                    --max_p_len 1000 \
                    --batch_size 32 \
                  	--train_files ./data/my_preprocess/train.json \
                  	--test_files ./data/my_preprocess/test.json \
                  	--dev_files ./data/my_preprocess/test.json


#python ./tensorflow/run.py --prepare --train --predict --algo BIDAF --epoch 10 --max_q_len 80 --max_a_len 80 --max_p_num 1 --max_p_len 100 --train_files ./data/my_preprocess/train.json --dev_files ./data/my_preprocess/test.json --test_files ./data/my_preprocess/test.json