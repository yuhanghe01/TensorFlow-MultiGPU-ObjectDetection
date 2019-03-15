# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw COCO dataset to TFRecord for object_detection.

Please note that this tool creates sharded output files.

Example usage:
    python create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import json
import os
import contextlib2
import numpy as np
import PIL.Image

import sys
sys.path.append('/home/yuhang/pycharm_project/tf-1.12.0-py2.7-mg8/tf-models-1.11/research/object_detection')
sys.path.append('/home/yuhang/pycharm_project/tf-1.12.0-py2.7-mg8/tf-models-1.11/research/')
from pycocotools import mask
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import cv2
import numpy as np


tf.logging.set_verbosity(tf.logging.INFO)


def create_tf_example(image,
                      annotations_list,
                      include_masks=False):
  """Converts image and annotations to a tf.Example proto.

  Args:
    image: dict with keys:
      [u'license', u'file_name', u'coco_url', u'height', u'width',
      u'date_captured', u'flickr_url', u'id']
    annotations_list:
      list of dicts with keys:
      [u'segmentation', u'area', u'iscrowd', u'image_id',
      u'bbox', u'category_id', u'id']
      Notice that bounding box coordinates in the official COCO dataset are
      given as [x, y, width, height] tuples using absolute coordinates where
      x, y represent the top-left (0-indexed) corner.  This function converts
      to the format expected by the Tensorflow Object Detection API (which is
      which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
      to image size).
    image_dir: directory containing the image files.
    category_index: a dict containing COCO category information keyed
      by the 'id' field of each category.  See the
      label_map_util.create_category_index function.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
  Returns:
    example: The converted tf.Example
    num_annotations_skipped: Number of (invalid) annotations that were ignored.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  image_height = image['height']
  image_width = image['width']
  filename = image['file_name']
  # image_id = image['id']
  image_id = filename

  #full_path = os.path.join(image_dir, filename)
  full_path = filename
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  key = hashlib.sha256(encoded_jpg).hexdigest()

  xmin = []
  xmax = []
  ymin = []
  ymax = []
  is_crowd = []
  category_names = []
  category_ids = []
  area = []
  encoded_mask_png = []
  num_annotations_skipped = 0
  for object_annotations in annotations_list:
    # (x, y, width, height) = tuple(object_annotations['bbox'])
    # if width <= 0 or height <= 0:
    #   num_annotations_skipped += 1
    #   continue
    # if x + width > image_width or y + height > image_height:
    #   num_annotations_skipped += 1
    #   continue
    xmin_tmp, ymin_tmp, xmax_tmp, ymax_tmp = object_annotations['xmin'], object_annotations['ymin'], object_annotations['xmax'], object_annotations['ymax']
    # xmin.append(float(x) / image_width)
    # xmax.append(float(x + width) / image_width)
    # ymin.append(float(y) / image_height)
    # ymax.append(float(y + height) / image_height)
    xmin.append( xmin_tmp )
    ymin.append( ymin_tmp )
    xmax.append( xmax_tmp )
    ymax.append( ymax_tmp )

    is_crowd.append( False )
    category_id = object_annotations['label_idx']
    #category_id = int(object_annotations['category_id'])
    category_ids.append(category_id)
    category_names.append(object_annotations['label_name'].encode('utf8'))
    area.append(0)

    if include_masks:
      run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                          image_height, image_width)
      binary_mask = mask.decode(run_len_encoding)
      if not object_annotations['iscrowd']:
        binary_mask = np.amax(binary_mask, axis=2)
      pil_image = PIL.Image.fromarray(binary_mask)
      output_io = io.BytesIO()
      pil_image.save(output_io, format='PNG')
      encoded_mask_png.append(output_io.getvalue())
  feature_dict = {
      'image/height':
          dataset_util.int64_feature(image_height),
      'image/width':
          dataset_util.int64_feature(image_width),
      'image/filename':
          dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id':
          dataset_util.bytes_feature(str(image_id).encode('utf8')),
      'image/key/sha256':
          dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded':
          dataset_util.bytes_feature(encoded_jpg),
      'image/format':
          dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin':
          dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax':
          dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin':
          dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax':
          dataset_util.float_list_feature(ymax),
      'image/object/class/text':
          dataset_util.bytes_list_feature(category_names),
      'image/object/is_crowd':
          dataset_util.int64_list_feature(is_crowd),
      'image/object/area':
          dataset_util.float_list_feature(area),
  }
  if include_masks:
    feature_dict['image/object/mask'] = (
        dataset_util.bytes_list_feature(encoded_mask_png))
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return key, example, num_annotations_skipped


def _create_tf_record_from_coco_annotations(num_shards, cate_idx_file,\
        train_img_list, \
        train_img_dir, \
        train_label_dir, \
        tf_record_ouptut_train ):
  """Loads COCO annotation json files and converts to tf.Record format.

  Args:
    annotations_file: JSON file containing bounding box annotations.
    image_dir: Directory containing the image files.
    output_path: Path to output tf.Record file.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
    num_shards: number of output file shards.
  """
  with contextlib2.ExitStack() as tf_record_close_stack:
    output_path = tf_record_ouptut_train
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_path, num_shards)
    # groundtruth_data = json.load(fid)
    # images = groundtruth_data['images']
    cate_idx_dict = dict()

    cate_idx_list = [line_tmp.rstrip('\n') for line_tmp in open(cate_idx_file, 'r').readlines()]
    for line_tmp in cate_idx_list:
        idx, cate_name = line_tmp.split(' ')
        idx = int( idx )
        cate_idx_dict[ idx ] = cate_name

    # category_index = label_map_util.create_category_index(
    #     groundtruth_data['categories'])

    # annotations_index = {}
    # if 'annotations' in groundtruth_data:
    #   tf.logging.info(
    #       'Found groundtruth annotations. Building annotations index.')
    #   for annotation in groundtruth_data['annotations']:
    #     image_id = annotation['image_id']
    #     if image_id not in annotations_index:
    #       annotations_index[image_id] = []
    #     annotations_index[image_id].append(annotation)
    # missing_annotation_count = 0
    # for image in images:
    #   image_id = image['id']
    #   if image_id not in annotations_index:
    #     missing_annotation_count += 1
    #     annotations_index[image_id] = []
    # tf.logging.info('%d images are missing annotations.',
    #                 missing_annotation_count)
    #
    total_num_annotations_skipped = 0
    img_list = [line_tmp.rstrip('\n') for line_tmp in open(train_img_list,'r').readlines()]
    for idx, image_name in enumerate(img_list):
      if idx % 100 == 0:
        tf.logging.info('On image %d of %d', idx, len(img_list))
      image_full_name = os.path.join( train_img_dir, image_name )
      label_name = image_name + '.csv'
      label_full_name = os.path.join( train_label_dir, label_name )

      if os.path.getsize( label_full_name ) == 0:
          continue

      label_matrix = np.loadtxt( label_full_name, dtype = np.float32 )
      label_matrix = np.reshape( label_matrix, ( -1, 5 ) )

      label_list = [line_tmp.rstrip('\n') for line_tmp in open( label_full_name, 'r' ).readlines() ]

      annotation_list = []
      for row_tmp in range( 0, label_matrix.shape[0] ):
          try:
            ymin, xmin, ymax, xmax, label_tmp = label_matrix[ row_tmp, : ]
          except:
            import pdb
            pdb.set_trace()
          ymin = float( ymin )
          xmin = float( xmin )
          ymax = float( ymax )
          xmax = float( xmax )
          # import pdb
          # pdb.set_trace()

          label_tmp = int( label_tmp )
          label_name_tmp = cate_idx_dict[ label_tmp ]

          anno_dict_tmp = dict()
          anno_dict_tmp['ymin'] = ymin
          anno_dict_tmp['xmin'] = xmin
          anno_dict_tmp['ymax'] = ymax
          anno_dict_tmp['xmax'] = xmax
          anno_dict_tmp['label_idx'] = label_tmp
          anno_dict_tmp['label_name'] = label_name_tmp

          annotation_list.append( anno_dict_tmp )

      img_tmp = cv2.imread( image_full_name, 1 )
      if img_tmp is None:
          print('cannot read the image {}'.format( image_full_name ) )
          continue

      img_h, img_w, _ = img_tmp.shape

      image_dict = dict()
      image_dict['height'] = img_h
      image_dict['width'] = img_w
      image_dict['file_name'] = image_full_name
      _, tf_example, num_annotations_skipped = create_tf_example(
          image_dict, annotation_list, include_masks = False )
      #total_num_annotations_skipped += num_annotations_skipped
      shard_idx = idx % num_shards
      output_tfrecords[shard_idx].write(tf_example.SerializeToString())
    tf.logging.info('Finished writing, skipped %d annotations.',
                    total_num_annotations_skipped)


def main(_):
  # assert FLAGS.train_image_dir, '`train_image_dir` missing.'
  # assert FLAGS.val_image_dir, '`val_image_dir` missing.'
  # assert FLAGS.test_image_dir, '`test_image_dir` missing.'
  # assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
  # assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'
  # assert FLAGS.testdev_annotations_file, '`testdev_annotations_file` missing.'

  tf_record_output_train = '/home/yuhang/caffe_fashion_tagging/FT_det_dataset/tf_records_dataset/train/train'
  train_img_list = '/home/yuhang/caffe_fashion_tagging/FT_det_dataset/img_list_train'
  train_img_dir = '/home/yuhang/caffe_fashion_tagging/FT_det_dataset/imgs'
  train_label_dir = '/home/yuhang/caffe_fashion_tagging/FT_det_dataset/labels'

  tf_record_output_val = '/home/yuhang/caffe_fashion_tagging/FT_det_dataset/tf_records_dataset/val/val'
  val_img_list = '/home/yuhang/caffe_fashion_tagging/FT_det_dataset/img_list_val'
  val_img_dir = train_img_dir
  val_label_dir = train_label_dir

  cate_idx_file = '/home/yuhang/pycharm_project/tf-1.12.0-py2.7-mg8/tf-models-1.11/research/object_detection/data/FT_13cate_cate_list'


  # if not tf.gfile.IsDirectory(FLAGS.output_dir):
  #   tf.gfile.MakeDirs(FLAGS.output_dir)
  # train_output_path = os.path.join(FLAGS.output_dir, 'coco_train.record')
  # val_output_path = os.path.join(FLAGS.output_dir, 'coco_val.record')
  # testdev_output_path = os.path.join(FLAGS.output_dir, 'coco_testdev.record')


  _create_tf_record_from_coco_annotations(
      num_shards=100, \
      cate_idx_file = cate_idx_file, \
      train_img_list=train_img_list, \
      train_img_dir = train_img_dir, \
      train_label_dir=train_label_dir,\
      tf_record_ouptut_train=tf_record_output_train
      )
  # _create_tf_record_from_coco_annotations(
  #     FLAGS.val_annotations_file,
  #     FLAGS.val_image_dir,
  #     val_output_path,
  #     FLAGS.include_masks,
  #     num_shards=10)
  # _create_tf_record_from_coco_annotations(
  #     FLAGS.testdev_annotations_file,
  #     FLAGS.test_image_dir,
  #     testdev_output_path,
  #     FLAGS.include_masks,
  #     num_shards=100)


if __name__ == '__main__':
  tf.app.run()
