	#?J %?A@#?J %?A@!#?J %?A@	[3?7???[3?7???![3?7???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6#?J %?A@@?P?%;>@1?H?5C@A???ّ???IWZF?=???Y5?ׂ???*	????K]@2F
Iterator::Modelt34???!?v?&?E@)Y?d:t??1??)]-6@:Preprocessing2U
Iterator::Model::ParallelMapV2?>????!?$??x5@)?>????1?$??x5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??b? ̝?!?i???8@)?y?'L??1?ʊ^4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??#???!?????7@)??^??1?(???+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??;????!?3??$@)??;????1?3??$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?x>?Ͱ?!E??j?,L@)-^,??w?1??^xs?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???u?!"?=?dq@)???u?1"?=?dq@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapw.???v??!????q`:@)0?AC?g?1XIGL@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 85.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?5.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Z3?7???I????V@Q?g?N?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	@?P?%;>@@?P?%;>@!@?P?%;>@      ??!       "	?H?5C@?H?5C@!?H?5C@*      ??!       2	???ّ??????ّ???!???ّ???:	WZF?=???WZF?=???!WZF?=???B      ??!       J	5?ׂ???5?ׂ???!5?ׂ???R      ??!       Z	5?ׂ???5?ׂ???!5?ׂ???b      ??!       JGPUYZ3?7???b q????V@y?g?N?@