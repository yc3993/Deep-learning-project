	?7L4H?B@?7L4H?B@!?7L4H?B@	?ݎtmQ???ݎtmQ??!?ݎtmQ??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?7L4H?B@?ʿ?W?@@1U??N?9@Ah??`ob??I?mr?$??Yywd?6???*	??|?5V_@2U
Iterator::Model::ParallelMapV2?NGɫ??!\"=??>@)?NGɫ??1\"=??>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???<?;??!??x??=@)E)!XU/??1??@??K8@:Preprocessing2F
Iterator::Model ??Ud??!3???F@)j??Gq??1h:.g-+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??ʉ?!С??8$@)??ʉ?1С??8$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???????!???1?2@)2?w???1_???*8!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???l }?!."?p4?@)???l }?1."?p4?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??/?????!?T?U?K@)???mz?1KOȫ?S@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??% ????!@?_???4@)??H??e?1BQr??k @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?ݎtmQ??IN?8=?V@Q?#?"?G@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?ʿ?W?@@?ʿ?W?@@!?ʿ?W?@@      ??!       "	U??N?9@U??N?9@!U??N?9@*      ??!       2	h??`ob??h??`ob??!h??`ob??:	?mr?$???mr?$??!?mr?$??B      ??!       J	ywd?6???ywd?6???!ywd?6???R      ??!       Z	ywd?6???ywd?6???!ywd?6???b      ??!       JGPUY?ݎtmQ??b qN?8=?V@y?#?"?G@