?	???A@???A@!???A@	&r9?~??&r9?~??!&r9?~??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6???A@?yVҊg?@1????}???A???&???I?????B @Y?[Z?{??*	??? ?B]@2U
Iterator::Model::ParallelMapV2?x??M???!k??c?E<@)?x??M???1k??c?E<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat$}ZE??!????k:@)&?<Y???1 ?Ge5@:Preprocessing2F
Iterator::Model??^???!?)??G@)C?Գ ???1ni?\?3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateR?r????!??Y?	2@)̚X?+???1e?? "@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?E????!????!@)?E????1????!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipo?ŏ1??!??F?J@)?cZ???z?1?l??v@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?'???v?!????R?@)?'???v?1????R?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?Qf`??!V???V4@)˟of?1Aˀ??d@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 88.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?5.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9&r9?~??I???? ?W@QV??@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?yVҊg?@?yVҊg?@!?yVҊg?@      ??!       "	????}???????}???!????}???*      ??!       2	???&??????&???!???&???:	?????B @?????B @!?????B @B      ??!       J	?[Z?{???[Z?{??!?[Z?{??R      ??!       Z	?[Z?{???[Z?{??!?[Z?{??b      ??!       JGPUY&r9?~??b q???? ?W@yV??@?"l
@gradient_tape/sequential_8/conv2d_24/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterxˣ??0??!xˣ??0??0"l
@gradient_tape/sequential_8/conv2d_25/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?????!?f-?;???0";
sequential_8/conv2d_24/Conv2DConv2DW?3???!??^?????0"j
?gradient_tape/sequential_8/conv2d_25/Conv2D/Conv2DBackpropInputConv2DBackpropInput̖{ּ_??!????????0";
sequential_8/conv2d_25/Conv2DConv2D???}?v??!??o?:???0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits?O?????!?c??24??"j
?gradient_tape/sequential_8/conv2d_26/Conv2D/Conv2DBackpropInputConv2DBackpropInput???5Mb??!Â_?|@??0";
sequential_8/conv2d_26/Conv2DConv2D???????!????}??0"l
@gradient_tape/sequential_8/conv2d_26/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterG???zӚ?!᭬?ZT??0"]
<gradient_tape/sequential_8/max_pooling2d/MaxPool/MaxPoolGradMaxPoolGrad8??z????!㲁??)??Q      Y@Y??{a0@a?{a??T@qw??pXE@y???;9??"?
both?Your program is POTENTIALLY input-bound because 88.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?5.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?42.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 