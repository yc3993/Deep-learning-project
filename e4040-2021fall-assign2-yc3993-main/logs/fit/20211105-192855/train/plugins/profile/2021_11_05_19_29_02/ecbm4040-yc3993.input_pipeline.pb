	|~!<
@|~!<
@!|~!<
@	g????@g????@!g????@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6|~!<
@P5z5@? @1!W?Y?@A?
?7k??I??Za??@Yę_???*	~j?t??a@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?>??s(??!}iN?N?B@)?Pi??>??15?a???@:Preprocessing2U
Iterator::Model::ParallelMapV2???x!??!x?????4@)???x!??1x?????4@:Preprocessing2F
Iterator::Model?.??"j??!a.ѽZ	B@)???#???1??8?q/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??^fؘ?!?/	
?0@) p??s???1????)"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice,e?X??!?szԈ@),e?X??1?szԈ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip%????g??!??.B??O@)Q??Û??1;Ss?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorUm7?7M?!????_@)Um7?7M?1????_@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???A???!??Tv3@)W!?'?>m?1??F2?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?33.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t28.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9g????@I??<UO@Q????F?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	P5z5@? @P5z5@? @!P5z5@? @      ??!       "	!W?Y?@!W?Y?@!!W?Y?@*      ??!       2	?
?7k???
?7k??!?
?7k??:	??Za??@??Za??@!??Za??@B      ??!       J	ę_???ę_???!ę_???R      ??!       Z	ę_???ę_???!ę_???b      ??!       JGPUYg????@b q??<UO@y????F?@