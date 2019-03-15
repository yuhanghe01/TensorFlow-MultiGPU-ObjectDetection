import tensorflow as tf
import cv2
import os
os.putenv("CUDA_VISIBLE_DEVICES","")
import sys
sys.path.append('/home/yuhang/pycharm_project/tf-1.12.0-py2.7-mg8/tf-models-1.11/research/')
import glob
import tf_example_decoder

def create_file_name_list( record_save_dir, tf_records_file_suffix ):
    reg_pattern = record_save_dir.rstrip('/') + '/*' + tf_records_file_suffix

    file_name_list = glob.glob(reg_pattern)

    return file_name_list

record_save_dir = '/home/yuhang/mscoco/tf_record_2017_102cate/train'
tf_records_file_suffix = 'tfrecord'

file_name_list = create_file_name_list( record_save_dir, tf_records_file_suffix )

print( file_name_list )

file_name_queue = tf.train.string_input_producer( file_name_list, shuffle = True, num_epochs = 1 )

reader = tf.TFRecordReader()
_, serialized_example = reader.read( file_name_queue )


label_map_proto_file = '/home/yuhang/pycharm_project/tf-1.12.0-py2.7-mg8/tf-models-1.11/research/object_detection/data/mscoco_label_map_102cate.pbtxt'

decoder = tf_example_decoder.TfExampleDecoder( use_display_name = True, label_map_proto_file = label_map_proto_file )

tensor_dict = decoder.decode( serialized_example )

with tf.Session( graph=tf.get_default_graph()) as sess:
    init_op = tf.group( tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer() )
    sess.run( init_op )

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners( coord = coord )
    try:
        while not coord.should_stop():
            while True:
                tensor_dict_val = sess.run( tensor_dict )
                print( tensor_dict_val['filename'] )
    except tf.errors.OutOfRangeError:
        print('Epoch limit reached. Traning Done!')
    finally:
        coord.request_stop()
    coord.join( threads )




# def decode_tf_example( tf_example ):
#     example = tf.train.Example(
#         features=tf.train.Features(
#             feature={
#                 'image/encoded': dataset_util.bytes_feature(encoded_jpeg),
#                 'image/format': dataset_util.bytes_feature('jpeg'),
#                 'image/source_id': dataset_util.bytes_feature('image_id'),
#             })).SerializeToString()
#
#     example_decoder = tf_example_decoder.TfExampleDecoder()
#     tensor_dict = example_decoder.decode(tf.convert_to_tensor(example))

