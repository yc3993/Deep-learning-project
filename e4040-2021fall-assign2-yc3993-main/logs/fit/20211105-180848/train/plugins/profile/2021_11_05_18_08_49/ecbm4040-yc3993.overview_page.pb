?	?$??ɋ@@?$??ɋ@@!?$??ɋ@@	???=F?????=F??!???=F??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?$??ɋ@@.??T?<@1{?V??? @A\Y???"??I?@?C????YC?K???*	??"??6^@2U
Iterator::Model::ParallelMapV2y?ՏM??!Q?n	?=@)y?ՏM??1Q?n	?=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????Q٠?!?dn?y:;@)?Ͻ???18:??>6@:Preprocessing2F
Iterator::Model=???m??!'o_2eI@)Z,E??@??1???Z65@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice,???)W??!~9{S??#@),???)W??1~9{S??#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??I`s??!ِ??͚H@)?{L?4{?1Nj[??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?a???x?!B???@)?a???x?1B???@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenaten??t???!a]$???+@)a?unڌs?1?????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapq<??f??!??Q>|0@) 
fL?j?13??G[?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 86.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?5.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???=F??Iݣ??G6W@Q??r@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	.??T?<@.??T?<@!.??T?<@      ??!       "	{?V??? @{?V??? @!{?V??? @*      ??!       2	\Y???"??\Y???"??!\Y???"??:	?@?C?????@?C????!?@?C????B      ??!       J	C?K???C?K???!C?K???R      ??!       Z	C?K???C?K???!C?K???b      ??!       JGPUY???=F??b qݣ??G6W@y??r@?"k
@gradient_tape/sequential_11/conv2d_34/Conv2D/Conv2DBackpropInputConv2DBackpropInputZh??۽?!Zh??۽?0"m
Agradient_tape/sequential_11/conv2d_34/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????????!?I???f??0"<
sequential_11/conv2d_34/Conv2DConv2D?'??????!?.??Z??0"m
Agradient_tape/sequential_11/conv2d_33/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterm?6D???!??V8???0"<
sequential_11/conv2d_33/Conv2DConv2DQ"??s???!??p:???0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits???6L??!Q??Ql???"-
IteratorGetNext/_1_Send?LF?GG??!??֎????"k
@gradient_tape/sequential_11/conv2d_35/Conv2D/Conv2DBackpropInputConv2DBackpropInputn	"5*h??!??????0"<
sequential_11/conv2d_35/Conv2DConv2D?cC?	???!2-?J??0"m
Agradient_tape/sequential_11/conv2d_35/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterɨ???Ԗ?!c6?N??0Q      Y@Y??????0@aVUUUU?T@q????b*N@y.?C~????"?
both?Your program is POTENTIALLY input-bound because 86.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?5.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?60.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 