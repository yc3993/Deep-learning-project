?		ȳ?W@@	ȳ?W@@!	ȳ?W@@	????????????!??????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6	ȳ?W@@?}??أ<@1A(??h???A:?6U???I?a?A
???Y??:M???*	V-?5`@2F
Iterator::Model???????!??ŶtJ@)>x?҆â?1????B<@:Preprocessing2U
Iterator::Model::ParallelMapV2;??]??!?6?ɫ?8@);??]??1?6?ɫ?8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat? ??ǟ?!ôڮ?7@)GV~???1?W?,??2@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???~31??!??v???%@)???~31??1??v???%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate_~?Ɍ???!g?4??Z0@)n?8)?{|?17???6s@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;??]?z?!-t??Z7@);??]?z?1-t??Z7@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???G?C??!2?_:I?G@)ض(?A&y?1E㪢p?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?@?v??!?pr?k2@)?N^?e?1L+???? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 87.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?5.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??????I???JSrW@Q???h?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?}??أ<@?}??أ<@!?}??أ<@      ??!       "	A(??h???A(??h???!A(??h???*      ??!       2	:?6U???:?6U???!:?6U???:	?a?A
????a?A
???!?a?A
???B      ??!       J	??:M?????:M???!??:M???R      ??!       Z	??:M?????:M???!??:M???b      ??!       JGPUY??????b q???JSrW@y???h?@?"l
@gradient_tape/sequential_9/conv2d_27/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??n!??!??n!??0"l
@gradient_tape/sequential_9/conv2d_28/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???eB1??!??|?X)??0";
sequential_9/conv2d_27/Conv2DConv2D?<??.??!?x???0"j
?gradient_tape/sequential_9/conv2d_28/Conv2D/Conv2DBackpropInputConv2DBackpropInput?M?B???!??k??X??0";
sequential_9/conv2d_28/Conv2DConv2D?~{????!0P/???0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsOe\Aо??!??Z
????"j
?gradient_tape/sequential_9/conv2d_29/Conv2D/Conv2DBackpropInputConv2DBackpropInput????????!??QI??0";
sequential_9/conv2d_29/Conv2DConv2D???:Ā??!y@?_??0"_
>gradient_tape/sequential_9/max_pooling2d_1/MaxPool/MaxPoolGradMaxPoolGrad??Jv????!?1z6??"-
IteratorGetNext/_1_Send??Y?)Κ?!g?ƀ??Q      Y@Y??{a0@a?{a??T@q??>?V?O@yd?/?y???"?
both?Your program is POTENTIALLY input-bound because 87.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?5.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?63.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 