		ȳ?W@@	ȳ?W@@!	ȳ?W@@	????????????!??????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6	ȳ?W@@?}??أ<@1A(??h???A:?6U???I?a?A
???Y??:M???*	V-?5`@2F
Iterator::Model???????!??ŶtJ@)>x?҆â?1????B<@:Preprocessing2U
Iterator::Model::ParallelMapV2;??]??!?6?ɫ?8@);??]??1?6?ɫ?8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat? ??ǟ?!ôڮ?7@)GV~???1?W?,??2@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???~31??!??v???%@)???~31??1??v???%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate_~?Ɍ???!g?4??Z0@)n?8)?{|?17???6s@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;??]?z?!-t??Z7@);??]?z?1-t??Z7@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???G?C??!2?_:I?G@)ض(?A&y?1E㪢p?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?@?v??!?pr?k2@)?N^?e?1L+???? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 87.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?5.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??????I???JSrW@Q???h?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?}??أ<@?}??أ<@!?}??أ<@      ??!       "	A(??h???A(??h???!A(??h???*      ??!       2	:?6U???:?6U???!:?6U???:	?a?A
????a?A
???!?a?A
???B      ??!       J	??:M?????:M???!??:M???R      ??!       Z	??:M?????:M???!??:M???b      ??!       JGPUY??????b q???JSrW@y???h?@