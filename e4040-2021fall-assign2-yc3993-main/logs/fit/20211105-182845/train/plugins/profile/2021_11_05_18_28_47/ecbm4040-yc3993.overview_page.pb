?	غ???A@غ???A@!غ???A@	d??+P???d??+P???!d??+P???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6غ???A@Q???ʩ>@1Iط??@A?fHū??IWzm6V???YV?)??%??*	?VU`@2U
Iterator::Model::ParallelMapV2?+d????!?歆?+6@)?+d????1?歆?+6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??{?E{??!?x[?ם>@)?ŏ1w??1̶w??6@:Preprocessing2F
Iterator::Model£?#????!.?j??D@)??????1??wMw?3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat؝?<????!w)ʚ#"6@) Й?????1F_?EQ?1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?Ye?????!T?ǭ?/!@)?Ye?????1T?ǭ?/!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?6S!?w?!?(KTI?@)?6S!?w?1?(KTI?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip#/kb???!?:??L(M@)`?????u?1?x?yP@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap? v??y??!wK?+@@)I?V?_?1;Q??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 86.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9d??+P???I8??3R?V@QH??`ɯ @Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Q???ʩ>@Q???ʩ>@!Q???ʩ>@      ??!       "	Iط??@Iط??@!Iط??@*      ??!       2	?fHū???fHū??!?fHū??:	Wzm6V???Wzm6V???!Wzm6V???B      ??!       J	V?)??%??V?)??%??!V?)??%??R      ??!       Z	V?)??%??V?)??%??!V?)??%??b      ??!       JGPUYd??+P???b q8??3R?V@yH??`ɯ @?"k
@gradient_tape/sequential_17/conv2d_53/Conv2D/Conv2DBackpropInputConv2DBackpropInputQ1??ƽ?!Q1??ƽ?0"k
@gradient_tape/sequential_17/conv2d_52/Conv2D/Conv2DBackpropInputConv2DBackpropInput?[?z???!SV??4m??0"m
Agradient_tape/sequential_17/conv2d_53/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter;??iڸ?!x˸4???0"<
sequential_17/conv2d_52/Conv2DConv2D?Rg??k??!??'??0"m
Agradient_tape/sequential_17/conv2d_52/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterAG?P????!x???E5??0"m
Agradient_tape/sequential_17/conv2d_51/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?ݴo>???!R避9??0"<
sequential_17/conv2d_53/Conv2DConv2D???6???!?	?*]???0"<
sequential_17/conv2d_51/Conv2DConv2DA#???"??!??4?n??0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits??mjm???!?h??t#??"a
@gradient_tape/sequential_17/max_pooling2d_10/MaxPool/MaxPoolGradMaxPoolGrad??G4F??!???$????Q      Y@Y??????0@aVUUUU?T@q?nQM?tQ@y/ ?????"?
both?Your program is POTENTIALLY input-bound because 86.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?69.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 