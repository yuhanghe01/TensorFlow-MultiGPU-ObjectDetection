eval_dir="/home/yuhang/mscoco/tf_record_2017/val_result/"
eval_config_path="/home/yuhang/mscoco/tf_record_2017/val_result/eval.config"
input_config_path="/home/yuhang/mscoco/tf_record_2017/val_result/input.config"

python offline_eval_map_corloc.py \
    --eval_dir=$eval_dir \
    --eval_config_path=$eval_config_path \
    --input_config_path=$input_config_path