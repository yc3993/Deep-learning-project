	;??XB@;??XB@!;??XB@	s?cu????s?cu????!s?cu????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6;??XB@?׹i3R?@1;?2@A#K?Xޭ?Ish??|???Y?(??/??*	m?????`@2F
Iterator::Model?0|DL???!?OJ???G@)G9?M?a??1c?T@	9@:Preprocessing2U
Iterator::Model::ParallelMapV2?P?v0b??!?{?d?6@)?P?v0b??1?{?d?6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat3???/??!??ǎ'Q7@)?	?????1.??!3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateu?8F?G??!?j??Ss7@)?4f???1G|?[?N+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?{L?4??!-Y0c?#@)?{L?4??1-Y0c?#@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????w?!??uč?@)?????w?1??uč?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipr6?,??!	??W-.J@)^K?=?u?1?W?n@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapD?1uWv??!B?PE'9@)???R?b?1q?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 85.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?5.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9s?cu????I?η??V@Q*?4ʽ| @Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?׹i3R?@?׹i3R?@!?׹i3R?@      ??!       "	;?2@;?2@!;?2@*      ??!       2	#K?Xޭ?#K?Xޭ?!#K?Xޭ?:	sh??|???sh??|???!sh??|???B      ??!       J	?(??/???(??/??!?(??/??R      ??!       Z	?(??/???(??/??!?(??/??b      ??!       JGPUYs?cu????b q?η??V@y*?4ʽ| @