	?v?{@@?v?{@@!?v?{@@	+???ů??+???ů??!+???ů??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?v?{@@&?"?d?<@1???1?@A??};??I?$??C??Y?L?J???*	?l???y^@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??W???!?}????9@)?U?3???1?
???4@:Preprocessing2F
Iterator::Modelc('?UH??!U?\AD@)???????1E?Xk$?4@:Preprocessing2U
Iterator::Model::ParallelMapV2'K?????!???M??3@)'K?????1???M??3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?2?&c??!?ۨwA:@)!#?????1???{??-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice˃?9D??!??!??&@)˃?9D??1??!??&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip|?E{????!??U???M@)?N?Z?7z?1'a??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor-@?j?y?!??a.??@)-@?j?y?1??a.??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???????!???Us<@)j?????e?1?~iu?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 86.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9+???ů??I 8])q?V@Q?m??r@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	&?"?d?<@&?"?d?<@!&?"?d?<@      ??!       "	???1?@???1?@!???1?@*      ??!       2	??};????};??!??};??:	?$??C???$??C??!?$??C??B      ??!       J	?L?J????L?J???!?L?J???R      ??!       Z	?L?J????L?J???!?L?J???b      ??!       JGPUY+???ů??b q 8])q?V@y?m??r@