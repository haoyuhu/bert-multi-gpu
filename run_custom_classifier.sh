#!/usr/bin/env bash

# default setting
max_seq_length=128
num_gpu_cores=4
train_batch_size=32
learning_rate=2e-5
num_train_epochs=3.0
CUDA_VISIBLE_DEVICES=0,1,2,3

# tranfer params through configuration
while getopts ":s:g:b:l:e:c:h" opt
do
    case $opt in
        s)
        echo "max_seq_length is: $OPTARG, default val is: ${max_seq_length}"
        max_seq_length=$OPTARG
        ;;
        g)
        echo "num_gpu_cores is: $OPTARG, default val is: ${num_gpu_cores}"
        num_gpu_cores=$OPTARG
        ;;
        b)
        echo "train_batch_size is: $OPTARG, default val is: ${train_batch_size}"
        train_batch_size=$OPTARG
        ;;
        l)
        echo "learning_rate is: $OPTARG, default val is: ${learning_rate}"
        learning_rate=$OPTARG
        ;;
        e)
        echo "num_train_epochs is: $OPTARG, default val is: ${num_train_epochs}"
        num_train_epochs=$OPTARG
        ;;
        c)
        echo "CUDA_VISIBLE_DEVICES is: $OPTARG, default val is: ${CUDA_VISIBLE_DEVICES}"
        CUDA_VISIBLE_DEVICES=$OPTARG
        ;;
        h)
        echo "current params setting:"
        echo "-s max_seq_length,        default val is: ${max_seq_length}"
        echo "-g num_gpu_cores,         default val is: ${num_gpu_cores}"
        echo "-b train_batch_size,      default val is: ${train_batch_size}"
        echo "-l learning_rate,         default val is: ${learning_rate}"
        echo "-e num_train_epochs,      default val is: ${num_train_epochs}"
        echo "-c CUDA_VISIBLE_DEVICES,  default val is: ${CUDA_VISIBLE_DEVICES}"
        exit 1
        ;;
        ?)
        echo "error: unknown params..."
        exit 1
        ;;
    esac
done

# assert gpu setting
cuda_devices=(${CUDA_VISIBLE_DEVICES//,/ })
if [ ${num_gpu_cores} == ${#cuda_devices[@]} ]
then
    echo "cuda devices and gpu cores matched"
else
    echo "error: please reset cuda devices or num_gpu_cores..."
    exit 1
fi

# set environmental variables
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

# execute the task
echo "task starts..."
python run_custom_classifier.py \
    --task_name=QQP \
    --do_lower_case=true \
    --do_train=true \
    --do_eval=true \
    --do_predict=true \
    --save_for_serving=true \
    --data_dir=/cfs/data/glue/QQP \
    --vocab_file=/cfs/models/bert-large-uncased/vocab.txt \
    --bert_config_file=/cfs/models/bert-large-uncased/bert_config.json \
    --init_checkpoint=/cfs/models/bert-large-uncased/bert_model.ckpt \
    --max_seq_length=${max_seq_length} \
    --train_batch_size=${train_batch_size} \
    --learning_rate${learning_rate} \
    --num_train_epochs=${num_train_epochs} \
    --use_gpu=true \
    --num_gpu_cores=${num_gpu_cores} \
    --use_fp16=true \
    --output_dir=/cfs/outputs/bert-large-uncased-qqp
echo "task is done..."