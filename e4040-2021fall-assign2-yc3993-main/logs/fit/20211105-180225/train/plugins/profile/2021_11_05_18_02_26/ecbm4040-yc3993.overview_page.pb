?	C9ѮA@C9ѮA@!C9ѮA@	m?Tȍ???m?Tȍ???!m?Tȍ???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6C9ѮA@???,A?=@1??c${@A`??i???Ih<?y???Ye9	?/???*	1?Z?]@2U
Iterator::Model::ParallelMapV2?"ڎ????!????f?>@)?"ڎ????1????f?>@:Preprocessing2F
Iterator::Model?q?	?O??!<Rܒ?I@)????b(??1??X3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???<?|??!FD??6@)???K???1?Yޟ??1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???????!?Gҩl5@)?@?C???11?{h)?-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???P1??!??$xT@@)???P1??1??$xT@@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???fw?!?eP@)???fw?1?eP@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??g?RD??!í#m=?H@)??ڦx\t?1???
9?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapz??????!uF?7@)??Z
H?_?1p???e??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 87.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9m?Tȍ???I\?S?.W@Q??3
?t@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???,A?=@???,A?=@!???,A?=@      ??!       "	??c${@??c${@!??c${@*      ??!       2	`??i???`??i???!`??i???:	h<?y???h<?y???!h<?y???B      ??!       J	e9	?/???e9	?/???!e9	?/???R      ??!       Z	e9	?/???e9	?/???!e9	?/???b      ??!       JGPUYm?Tȍ???b q\?S?.W@y??3
?t@?"l
@gradient_tape/sequential_5/conv2d_15/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?r??μ?!?r??μ?0"l
@gradient_tape/sequential_5/conv2d_16/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter!_??T9??!?h?'??0";
sequential_5/conv2d_15/Conv2DConv2D???p??!?0	^??0";
sequential_5/conv2d_16/Conv2DConv2D{UQ?V??!{CŶ?3??0"j
?gradient_tape/sequential_5/conv2d_17/Conv2D/Conv2DBackpropInputConv2DBackpropInput??ԭ=??!4??+????0"j
?gradient_tape/sequential_5/conv2d_16/Conv2D/Conv2DBackpropInputConv2DBackpropInput??K)??!syL?>???0"l
@gradient_tape/sequential_5/conv2d_17/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?ɋ͞?!鄄Z??0";
sequential_5/conv2d_17/Conv2DConv2D?`????!?o??J??0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits?Ea?ӝ?!??V9??"d
Cgradient_tape/sequential_5/average_pooling2d_15/AvgPool/AvgPoolGradAvgPoolGrad9?g[?Ü?!AԹq9??Q      Y@Y?/??0@a4??}?T@q9??j'?R@y^t????"?
both?Your program is POTENTIALLY input-bound because 87.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?75.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 