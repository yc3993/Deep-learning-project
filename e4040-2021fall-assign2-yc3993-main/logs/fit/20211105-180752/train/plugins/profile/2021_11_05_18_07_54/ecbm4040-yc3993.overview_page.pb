?	???Z?A@???Z?A@!???Z?A@	?6??c???6??c??!?6??c??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6???Z?A@????/>@1?v?l @AP??n???I????:??Yw??g??*	{?G?Z_@2U
Iterator::Model::ParallelMapV2m?i?*???!1?+.?<@)m?i?*???11?+.?<@:Preprocessing2F
Iterator::ModelJ??Gp#??!??Hwj?J@)&???J??1??e??^9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat=Զa??!@"?j&8@)???[???1?0??L?1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceV}??b??!???gq?#@)V}??b??1???gq?#@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??R?r/??!	?su4@)??R?r/??1	?su4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?Gp#e???!???I?/@)?#d ?.?1_?U?G@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip`?eM,???!?6???OG@)oG8-x?w?1?*??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???-????!:??{??1@)?شR?b?1?S?$?j??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?6??c??I8????OW@Q?e??t@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????/>@????/>@!????/>@      ??!       "	?v?l @?v?l @!?v?l @*      ??!       2	P??n???P??n???!P??n???:	????:??????:??!????:??B      ??!       J	w??g??w??g??!w??g??R      ??!       Z	w??g??w??g??!w??g??b      ??!       JGPUY?6??c??b q8????OW@y?e??t@?"k
@gradient_tape/sequential_10/conv2d_31/Conv2D/Conv2DBackpropInputConv2DBackpropInput"???/??!"???/??0"m
Agradient_tape/sequential_10/conv2d_31/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????t??!^Ӑ???0"<
sequential_10/conv2d_31/Conv2DConv2Du?q? ???!?d8	???0"m
Agradient_tape/sequential_10/conv2d_30/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??n?X??!???,$??0"<
sequential_10/conv2d_30/Conv2DConv2D?څv6???!s?!ʚ???0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsT?:????!??М???"k
@gradient_tape/sequential_10/conv2d_32/Conv2D/Conv2DBackpropInputConv2DBackpropInputk??E6??!j?? O???0"<
sequential_10/conv2d_32/Conv2DConv2D???)????!?d?y????0"`
?gradient_tape/sequential_10/max_pooling2d_2/MaxPool/MaxPoolGradMaxPoolGrad????b??!?#?L??"m
Agradient_tape/sequential_10/conv2d_32/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter#?_?^??!}1????0Q      Y@Y??????0@aVUUUU?T@q??#??Q@y2?(?'??"?
both?Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?70.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 