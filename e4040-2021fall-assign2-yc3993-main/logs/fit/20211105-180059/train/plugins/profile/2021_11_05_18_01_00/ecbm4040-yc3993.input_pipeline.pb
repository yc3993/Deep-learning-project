	->?x?A@->?x?A@!->?x?A@	??3*??????3*????!??3*????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6->?x?A@T?D?[??@1?]M?????ApA?,_??I?SV?????Y??T?:??*	?/?$?^@2U
Iterator::Model::ParallelMapV2M?-?Π?!??f???:@)M?-?Π?1??f???:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?DR????!?q7??@9@)(?r?w???1'F9oܱ3@:Preprocessing2F
Iterator::ModelWд??h??!#??I?F@),g~5??1r?z??|2@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice=?බ???!?y?k?'@)=?බ???1?y?k?'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?F???R??!?,ʋ??5@)??A?V???1Q?ث?y$@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??z2??{?!߮??|;@)??z2??{?1߮??|;@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipjkD0.??!???^K@)?<??S?z?1{???<?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???lY??!??m?,8@)_?vj.7h?1@~]J@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 88.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??3*????IF??ؠoW@QX?,k-@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	T?D?[??@T?D?[??@!T?D?[??@      ??!       "	?]M??????]M?????!?]M?????*      ??!       2	pA?,_??pA?,_??!pA?,_??:	?SV??????SV?????!?SV?????B      ??!       J	??T?:????T?:??!??T?:??R      ??!       Z	??T?:????T?:??!??T?:??b      ??!       JGPUY??3*????b qF??ؠoW@yX?,k-@