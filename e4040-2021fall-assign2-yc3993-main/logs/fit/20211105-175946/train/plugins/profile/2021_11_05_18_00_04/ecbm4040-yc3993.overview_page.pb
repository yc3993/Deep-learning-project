?	od?? @od?? @!od?? @	ޡ"?z\@ޡ"?z\@!ޡ"?z\@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6od?? @?Nyt#?@1??b?@A8?GnM???IU?=ϟ???Y??Z&????*	X9??d@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????˪?!??:h?N@@)K?R??%??1D???Ž9@:Preprocessing2U
Iterator::Model::ParallelMapV2IH?m????!1???&0@)IH?m????11???&0@:Preprocessing2F
Iterator::Model?F?@??!z?&K ?>@)g??j+???1???p*-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicebJ$??(??!??ѹ??*@)bJ$??(??1??ѹ??*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateR?r????!?7?/?P:@)B???1??8???)@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??uR_???!~???<~@)??uR_???1~???<~@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?)??s??!!p6??PQ@)?B?Գ ??1aK?e?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap ??L??!J?
??>@)qN`:?{?1cQ?m?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 29.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?13.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9ߡ"?z\@I?'V?E@Q?˄SJ@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Nyt#?@?Nyt#?@!?Nyt#?@      ??!       "	??b?@??b?@!??b?@*      ??!       2	8?GnM???8?GnM???!8?GnM???:	U?=ϟ???U?=ϟ???!U?=ϟ???B      ??!       J	??Z&??????Z&????!??Z&????R      ??!       Z	??Z&??????Z&????!??Z&????b      ??!       JGPUYߡ"?z\@b q?'V?E@y?˄SJ@?"i
=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?W,_6??!?W,_6??0"8
sequential/conv2d_1/Conv2DConv2D?Q?s??!??????0"g
<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput\,??H??!%ÂL???0"g
;gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?m ???!:?q?>??0"6
sequential/conv2d/Conv2DConv2D???%??!v?(?C??0"g
<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput?u-?`???!ԗ?	????0"i
=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterW???)??!F???2 ??0"_
>gradient_tape/sequential/average_pooling2d/AvgPool/AvgPoolGradAvgPoolGradt"????!?М?????"F
(gradient_tape/sequential/conv2d/TanhGradTanhGraduV??F???!???,z???"-
IteratorGetNext/_1_Send?*?%??!R?.?N??Q      Y@Y?B???0@aNozӛ?T@qKAFa +#@y?????=??"?

both?Your program is POTENTIALLY input-bound because 29.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?13.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 