	^gC???A@^gC???A@!^gC???A@	ɫ?)????ɫ?)????!ɫ?)????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6^gC???A@?]???>@1 Tq??@Aj?TQ?ʢ?I?=?-??Y?~߿yq??*	أp=
?Z@2F
Iterator::ModelQf?L2r??!C?<E??K@)?/EHݞ?1?b??P,<@:Preprocessing2U
Iterator::Model::ParallelMapV2??S??!?%̵?h;@)??S??1?%̵?h;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?ʼUס??!<I??XO8@)???]g??1?Y?oP?3@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicex
?Rς??!?z??}$@)x
?Rς??1?z??}$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?Ēr?9??!????=?+@)6t??Pn{?1f??|?	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor հ??t?!%???!@) հ??t?1%???!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??ϛ?T??!?;ún5F@)???? ?s?1W??;?=@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap]1#?=??!Q?%?/@)=??- ?^?1???s???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 86.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9ȫ?)????I??z??V@Q?ό?;? @Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?]???>@?]???>@!?]???>@      ??!       "	 Tq??@ Tq??@! Tq??@*      ??!       2	j?TQ?ʢ?j?TQ?ʢ?!j?TQ?ʢ?:	?=?-???=?-??!?=?-??B      ??!       J	?~߿yq???~߿yq??!?~߿yq??R      ??!       Z	?~߿yq???~߿yq??!?~߿yq??b      ??!       JGPUYȫ?)????b q??z??V@y?ό?;? @