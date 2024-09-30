# Step 1: get speculative score
s_base_model='YOUR_DIR/wmt_gemma2-factory-sft'
tokenizer='YOUR_DIR/models/gemma-2b'
dataset='YOUR_DIR/STAFF/finetune/data/wmt_dataset-gemma-2b'
task='wmt'
finetune='True'
baseline="['Effort']"

CUDA_VISIBLE_DEVICES=0 python effort_score.py --base_model $s_base_model --tokenizer $tokenizer --dataset $dataset --finetune $finetune --task $task --baseline $baseline

# Step 2: get velidation score. Step 3 also implement an efficient version to get score in runtime. TODO: Update
l_base_model='YOUR_DIR/models/gemma-7b'
tokenizer='YOUR_DIR/models/gemma-7b'
dataset='YOUR_DIR/STAFF/finetune/data/wmt_dataset-gemma-7b'
task='wmt'
finetune='False'
baseline="['Effort']"

CUDA_VISIBLE_DEVICES=0 python effort_score.py --base_model $l_base_model --tokenizer $tokenizer --dataset $dataset --finetune $finetune --task $task --baseline $baseline


# Step 3: selection. We have already provided the selected coreset in the save_dir
save_dir='./selected_data/gemma/wmt_selected'
small_model_dir=$s_base_model
large_model_dir=$l_base_model
python selection-gemma.py --save_dir $save_dir --small_model_dir $small_model_dir --large_model_dir $large_model_dir