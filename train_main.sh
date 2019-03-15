model_dir="xxxx"
pipeline_config_path="xxx"
num_train_steps=300000
python model_main.py \
    --model_dir=$model_dir \
    --pipeline_config_path=$pipeline_config_path \
    --num_train_steps=$num_train_steps \
    --eval_training_data=False
