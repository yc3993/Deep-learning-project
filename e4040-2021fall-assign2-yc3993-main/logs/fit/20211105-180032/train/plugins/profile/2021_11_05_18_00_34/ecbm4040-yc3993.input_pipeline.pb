	?Բ??B@?Բ??B@!?Բ??B@	0??}W??0??}W??!0??}W??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?Բ??B@? Ϡ??@1.??:??@A?f???¯?Ip???$T??Y'?E'K???*	?n??"]@2U
Iterator::Model::ParallelMapV2??imۛ?!??3sW7@)??imۛ?1??3sW7@:Preprocessing2F
Iterator::Model?ȭI?%??!1?$???E@)?%Tp??1?/tpz4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatVb??????!??a??]7@)?Z`?????1?????2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?a?4??!??
??O9@)	?/?????1pz{?c?*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicex?=\r??!u??1?'@)x?=\r??1u??1?'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip~?p?°?!???L@)?%?<y?1~???n @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???8u?!??b[C?@)???8u?1??b[C?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??mr??!??n?.?;@)]?E?~e?1vQ?!@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 84.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no91??}W??Iy?6}6V@Q(D[6L?$@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	? Ϡ??@? Ϡ??@!? Ϡ??@      ??!       "	.??:??@.??:??@!.??:??@*      ??!       2	?f???¯??f???¯?!?f???¯?:	p???$T??p???$T??!p???$T??B      ??!       J	'?E'K???'?E'K???!'?E'K???R      ??!       Z	'?E'K???'?E'K???!'?E'K???b      ??!       JGPUY1??}W??b qy?6}6V@y(D[6L?$@