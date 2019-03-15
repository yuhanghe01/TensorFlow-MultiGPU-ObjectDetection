checkpoint_dir="/home/yuhang/mscoco/model/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/"
eval_dir="/home/yuhang/mscoco/model/trained_model_fpn_300/eval_dir_iter_52111"
pipeline_config_path="/home/yuhang/mscoco/model/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/pipeline_640.config"

CUDA_VISIBLE_DEVICES="6" python eval.py \
    --logtostderr \
    --checkpoint_dir=$checkpoint_dir \
    --eval_dir=$eval_dir \
    --pipeline_config_path=$pipeline_config_path