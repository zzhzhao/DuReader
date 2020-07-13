


python ./tensorflow/run.py --prepare \
                    --train \
                    --algo BIDAF \
                    --epoch 10 \
                    --max_q_len 80 \
                    --max_a_len 80 \
                    --batch_size 16 \
                  	--train_files /match_lstm_bidaf_data/train_para_extra.json \
                  	--test_files /match_lstm_bidaf_data/test_para_extra.json \
                  	--dev_files /match_lstm_bidaf_data/test_para_extra.json


#python ./tensorflow/run.py --prepare --train --predict --algo BIDAF --epoch 10 --max_q_len 80 --max_a_len 80 --train_files ./data/du_format/train_para_extra_small.json --test_files ./data/du_format/test_small.json --dev_files ./data/du_format/test_small.json