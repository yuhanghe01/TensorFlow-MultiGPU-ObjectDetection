train_image_dir="/opt/FTE/users/opendata/mscoco/train2017"
val_image_dir="/opt/FTE/users/opendata/mscoco/val2017"
output_dir="/home/yuhang/mscoco/tf_record_2017_109cate_0311/"
val_annotations_file="/home/yuhang/project_huawei_smartphone/new_data_huawei_0225/new_data_process/data_process_0311_109cate/mscoco_huawei_mscoco_val_109cate_with_openimage.json"
#val_annotations_file="/home/yuhang/project_huawei_smartphone/new_data_huawei_0225/new_data_process/mscoco_val_with_12cate_0303_new.json"
train_annotations_file="/home/yuhang/project_huawei_smartphone/new_data_huawei_0225/new_data_process/data_process_0311_109cate/mscoco_huawei_mscoco_train_109cate_with_openimage.json"
#train_annotations_file="/home/yuhang/project_huawei_smartphone/new_data_huawei_0225/new_data_process/mscoco_train_with_12cate_0303_new.json"

CUDA_VISIBLE_DEVICES="" python create_coco_tf_record.py \
    --train_image_dir=$train_image_dir \
    --val_image_dir=$val_image_dir \
    --output_dir=$output_dir \
    --val_annotations_file=$val_annotations_file \
    --train_annotations_file=$train_annotations_file