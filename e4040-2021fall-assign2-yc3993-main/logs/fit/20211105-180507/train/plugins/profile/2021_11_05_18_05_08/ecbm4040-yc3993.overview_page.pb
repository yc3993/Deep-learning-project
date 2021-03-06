?	??`?$@@??`?$@@!??`?$@@	??l=??????l=????!??l=????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??`?$@@+Kt?YT<@1??Y????A;?I/??Ig?8?/??Y??&S??*	????x?a@2F
Iterator::Model????9???!?VGcB?G@)8en?ݣ?1???V;@:Preprocessing2U
Iterator::Model::ParallelMapV2?????B??!)??j"4@)?????B??1)??j"4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateL8????!?E%?=?8@)?mē?̘?1?I?6?1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatb?cҟ?!??i??5@)???8a?1?+u	1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice)H4?"??!A??Fv@))H4?"??1A??Fv@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorPVW@|?!?ٮ;ap@)PVW@|?1?ٮ;ap@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Ziph??s???!$????CJ@)P?i4?x?19F?$@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[A?+??!?F?$a:@)?,??;?c?1 ?j.??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?5.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??l=????I?r?.mW@Q?:??Y@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	+Kt?YT<@+Kt?YT<@!+Kt?YT<@      ??!       "	??Y??????Y????!??Y????*      ??!       2	;?I/??;?I/??!;?I/??:	g?8?/??g?8?/??!g?8?/??B      ??!       J	??&S????&S??!??&S??R      ??!       Z	??&S????&S??!??&S??b      ??!       JGPUY??l=????b q?r?.mW@y?:??Y@?"l
@gradient_tape/sequential_7/conv2d_21/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??W????!??W????0"l
@gradient_tape/sequential_7/conv2d_22/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltereUl=??!nV:????0";
sequential_7/conv2d_21/Conv2DConv2D??!s?0??!????A??0"j
?gradient_tape/sequential_7/conv2d_22/Conv2D/Conv2DBackpropInputConv2DBackpropInput?`RИ???!<G.???0";
sequential_7/conv2d_22/Conv2DConv2D?F?t?G??!?X>????0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits???????!B??i??"j
?gradient_tape/sequential_7/conv2d_23/Conv2D/Conv2DBackpropInputConv2DBackpropInput?j????!?]⿜a??0"-
IteratorGetNext/_1_Send?o?`7???!s???&??";
sequential_7/conv2d_23/Conv2DConv2D??z?7??!ˏ?d ??0"l
@gradient_tape/sequential_7/conv2d_23/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??	;???!1?h?^???0Q      Y@Y??{a0@a?{a??T@q F??2oR@y?2?7^???"?
both?Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?5.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?73.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 