from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import logging
os.putenv('CUDA_VISIBLE_DEVICES','')
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
logging.getLogger("tensorflow").setLevel(logging.INFO)

g = tf.Graph()

with g.as_default():
    a = tf.Variable( initial_value=1.0, dtype=tf.float32 )
    b = tf.Variable( initial_value=2.0, dtype=tf.float32 )
    tf.logging.info( a )
    with tf.control_dependencies( [ tf.print( a ), tf.print( b ) ] ):
        c = tf.add( a, b )

    for i in tf.get_default_graph().get_operations():
        tf.logging.info(i.name)

with tf.Session( graph = g ) as sess:
    tf.logging.set_verbosity( tf.logging.INFO )
    init_op = tf.group( tf.local_variables_initializer(), tf.global_variables_initializer() )
    sess.run( init_op )

    print( sess.run( c ) )