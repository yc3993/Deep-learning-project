	¤????B@¤????B@!¤????B@	j??@j??@!j??@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6¤????B@?U??6?@@1?\??@A?J?.????I????K??Y ???w??*	f;?O?W^@2U
Iterator::Model::ParallelMapV2]k?SUh??!?󔗣C@)]k?SUh??1?󔗣C@:Preprocessing2F
Iterator::Model?o??R???!P?q?0?L@)l?f?ܖ?1;7??1e2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?a??h???!i??4@)G?j?????1{??蝦/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?@?Ρ??!?|???0@)??b????1????$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice$????5??!ZU꫼?@)$????5??1ZU꫼?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?nf???t?!?X?LB?@)?nf???t?1?X?LB?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip8???LM??!?U?o?)E@)?aQ?s?1?I?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?C?.l͖?!??C?X2@)()? ?\?1l~???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9j??@I?m?wV@Q;$H=@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?U??6?@@?U??6?@@!?U??6?@@      ??!       "	?\??@?\??@!?\??@*      ??!       2	?J?.?????J?.????!?J?.????:	????K??????K??!????K??B      ??!       J	 ???w?? ???w??! ???w??R      ??!       Z	 ???w?? ???w??! ???w??b      ??!       JGPUYj??@b q?m?wV@y;$H=@