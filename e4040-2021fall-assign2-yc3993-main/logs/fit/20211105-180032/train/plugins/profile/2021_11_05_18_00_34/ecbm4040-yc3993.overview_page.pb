?	?Բ??B@?Բ??B@!?Բ??B@	0??}W??0??}W??!0??}W??"w
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
	? Ϡ??@? Ϡ??@!? Ϡ??@      ??!       "	.??:??@.??:??@!.??:??@*      ??!       2	?f???¯??f???¯?!?f???¯?:	p???$T??p???$T??!p???$T??B      ??!       J	'?E'K???'?E'K???!'?E'K???R      ??!       Z	'?E'K???'?E'K???!'?E'K???b      ??!       JGPUY1??}W??b qy?6}6V@y(D[6L?$@?"k
?gradient_tape/sequential_1/conv2d_4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??f+`1??!??f+`1??0"i
>gradient_tape/sequential_1/conv2d_4/Conv2D/Conv2DBackpropInputConv2DBackpropInputS??`??!???I??0":
sequential_1/conv2d_4/Conv2DConv2Du6Ы??!=?N?	???0"k
?gradient_tape/sequential_1/conv2d_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??o?I???!?Lu????0":
sequential_1/conv2d_3/Conv2DConv2D?b,{??!;??? ??0"c
Bgradient_tape/sequential_1/average_pooling2d_3/AvgPool/AvgPoolGradAvgPoolGrad^j?Sy???!??NG???"J
,gradient_tape/sequential_1/conv2d_3/TanhGradTanhGrad????????!>?[????"k
?gradient_tape/sequential_1/conv2d_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??M(???!???????0":
sequential_1/conv2d_3/BiasAddBiasAdd??Y;(???!??x5?\??"i
>gradient_tape/sequential_1/conv2d_5/Conv2D/Conv2DBackpropInputConv2DBackpropInput??qo5???!A]?????0Q      Y@Y??????0@aVUUUU?T@q??}??I@y??%??"?
both?Your program is POTENTIALLY input-bound because 84.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?51.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 