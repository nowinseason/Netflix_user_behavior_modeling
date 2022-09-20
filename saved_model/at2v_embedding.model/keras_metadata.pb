
�root"_tf_keras_network*�{"name": "model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 299]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAEAAABTAAAAcwQAAAB8AFMAKQFOqQApAdoBeHIBAAAAcgEAAAD6Ri91c3Iv\nbG9jYWwvbGliL3B5dGhvbjMuNy9kaXN0LXBhY2thZ2VzL3N0ZWxsYXJncmFwaC9sYXllci9hdHRy\naTJ2ZWMucHnaCDxsYW1iZGE+aAAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "stellargraph.layer.attri2vec", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["lambda_1", 0, 0]]}, "shared_object_id": 5, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 299]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 299]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 299]}, "float32", "input_3"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 299]}, "float32", "input_3"]}, "keras_version": "2.8.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 299]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["input_3", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAEAAABTAAAAcwQAAAB8AFMAKQFOqQApAdoBeHIBAAAAcgEAAAD6Ri91c3Iv\nbG9jYWwvbGliL3B5dGhvbjMuNy9kaXN0LXBhY2thZ2VzL3N0ZWxsYXJncmFwaC9sYXllci9hdHRy\naTJ2ZWMucHnaCDxsYW1iZGE+aAAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "stellargraph.layer.attri2vec", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 4}], "input_layers": [["input_3", 0, 0]], "output_layers": [["lambda_1", 0, 0]]}}}2
�root.layer-0"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 299]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 299]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}2
�root.layer_with_weights-0"_tf_keras_layer*�{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_3", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 299}}, "shared_object_id": 7}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 299]}}2
�root.layer-2"_tf_keras_layer*�{"name": "lambda_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAEAAABTAAAAcwQAAAB8AFMAKQFOqQApAdoBeHIBAAAAcgEAAAD6Ri91c3Iv\nbG9jYWwvbGliL3B5dGhvbjMuNy9kaXN0LXBhY2thZ2VzL3N0ZWxsYXJncmFwaC9sYXllci9hdHRy\naTJ2ZWMucHnaCDxsYW1iZGE+aAAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "stellargraph.layer.attri2vec", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 4}2