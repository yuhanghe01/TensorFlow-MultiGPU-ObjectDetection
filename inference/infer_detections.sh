input_tfrecord_paths="/home/yuhang/mscoco/tf_record_2017/val/*"
output_tfrecord_path="/home/yuhang/mscoco/tf_record_2017/val_result/fpn_640.tfrecord"
inference_graph="/home/yuhang/mscoco/model/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb"

python infer_detections.py \
    --input_tfrecord_paths=$input_tfrecord_paths \
    --output_tfrecord_path=$output_tfrecord_path \
    --inference_graph=$inference_graph