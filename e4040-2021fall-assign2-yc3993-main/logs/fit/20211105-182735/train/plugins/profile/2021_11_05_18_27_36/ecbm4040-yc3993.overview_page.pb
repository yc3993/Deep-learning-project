?	T;?ԖzB@T;?ԖzB@!T;?ԖzB@	??%=?????%=???!??%=???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6T;?ԖzB@?.oS>@1VҊo(|@Ajܛ?0???I?????Y??-W?6??*	?????`@2F
Iterator::Model??,
?(??!?Ǹ:RJ@)_?"??]??17@X?S?=@:Preprocessing2U
Iterator::Model::ParallelMapV2?qm????!?Oy?7@)?qm????1?Oy?7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatnm?y?ؠ?!C=??k8@)?????1R?P)qd3@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??
E???!???m?&@)??
E???1???m?&@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor ?t???{?!??qA@) ?t???{?1??qA@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???	.V??!8G???G@),???cz?12??< @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?<?$??!???u00@)?[X7?y?1???F?3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap+4?f??!??vư(2@)???eNg?1?|??? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 82.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??%=???I??\.?V@Q?`*}k@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?.oS>@?.oS>@!?.oS>@      ??!       "	VҊo(|@VҊo(|@!VҊo(|@*      ??!       2	jܛ?0???jܛ?0???!jܛ?0???:	??????????!?????B      ??!       J	??-W?6????-W?6??!??-W?6??R      ??!       Z	??-W?6????-W?6??!??-W?6??b      ??!       JGPUY??%=???b q??\.?V@y?`*}k@?"<
sequential_15/conv2d_46/Conv2DConv2D??#?)???!??#?)???0"k
@gradient_tape/sequential_15/conv2d_47/Conv2D/Conv2DBackpropInputConv2DBackpropInput?Z?iû?!? t??X??0"k
@gradient_tape/sequential_15/conv2d_46/Conv2D/Conv2DBackpropInputConv2DBackpropInput,?:`ka??!???U?D??0"m
Agradient_tape/sequential_15/conv2d_46/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter{??/4??!շ<????0"m
Agradient_tape/sequential_15/conv2d_47/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterK?S|???!?
31????0"m
Agradient_tape/sequential_15/conv2d_45/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?5?Gs??!?; ????0"<
sequential_15/conv2d_45/Conv2DConv2D7t?߀C??!??????0"<
sequential_15/conv2d_47/Conv2DConv2D?[?M>0??!???;??0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits?b?証??!?{!L????"`
?gradient_tape/sequential_15/max_pooling2d_7/MaxPool/MaxPoolGradMaxPoolGrad5?(????!|h??~??Q      Y@Y??????0@aVUUUU?T@q?N??`?L@y?v?{???"?
both?Your program is POTENTIALLY input-bound because 82.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?57.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 