# -*- coding:utf-8 -*-
import os
from tensorflow.python import pywrap_tensorflow
model_dir = "save/ll20b_1"
checkpoint_path = os.path.join(model_dir, "tf_placement.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key, end=' ')
    print(reader.get_tensor(key))