# TensorFlow Multi-GPU based Object Detection code

This repository contains the implementation of [TensorFlow](https://www.tensorflow.org/) based Multi-GPU Object Detection, which is originally based on [TF-OD-API](https://github.com/tensorflow/models/tree/master/research/object_detection), but with important modification to support fancy features such as multi-GPU training (local machine mode, not distribute mode), flexible GPU assignment and GPU memory soft growth exploitation.

## Implementation Highlight

* Implementation is basically referred to [cifar multi-GPU train](https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py).
* Successfully tested with TF-1.12.0. Other TF versions are not warranted to work.
* Subtle difference with [TF-OD-API](https://github.com/tensorflow/models/tree/master/research/object_detection). You can use it nearly the same as [TF-OD-API](https://github.com/tensorflow/models/tree/master/research/object_detection)
* Modification mainly lies in `protos/train.proto` and `model_lib_multiGPU.py`. I sugguest to read the source code if you are interested in complete implementation.

* More fancy train configurations are implemented in `protos/train.proto`


## Usage

* Step1: build this repository the same way as [TF-OD-API](https://github.com/tensorflow/models/tree/master/research/object_detection). Don't forget to compile the ProtoBuf with the following command:
 
 ```python
 # From tensorflow/models/research/
 protoc object_detection/protos/*.proto --python_out=.
 ```

* Step2: Configure proto file, one typical configuration file might look as:

  ```python
  train_config {
    batch_size: 30
    GPU_num: 4
    visible_device_list: "0,1,2,3"
    keep_checkpoint_max: 10
  }
  ```

  Note that the `batch_size` here means the batch size in a single GPU. In `train` mode, the data input pipeline feeds `batch_size * GPU_num` data in each iteration.

* Step3: Then you can train as the usual way illustrated in [TF-OD-API](https://github.com/tensorflow/models/tree/master/research/object_detection). Don't forget to take a cup of coffee right now!

## Contact

Yuhang He yuhanghe01[at]gmail.com
