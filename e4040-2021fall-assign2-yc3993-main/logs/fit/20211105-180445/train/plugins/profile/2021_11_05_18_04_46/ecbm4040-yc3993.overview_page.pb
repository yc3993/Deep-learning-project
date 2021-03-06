?	??Os? A@??Os? A@!??Os? A@	?xfH????xfH???!?xfH???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??Os? A@???Ɋ?<@1???); @Av?[??`??I:??ȵ??Y???0(???*	?G?z?^@2F
Iterator::ModelP?,?cy??!??;??H@)/??????1?A?\?v<@:Preprocessing2U
Iterator::Model::ParallelMapV2B?L????!IB??5@)B?L????1IB??5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?B????!~??F0?3@)?e???~??1??????.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?A?p?-??!U?????7@)B???D??1?=?<I?,@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?jIG9??!??w??<#@)?jIG9??1??w??<#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipm????|??!?=??I@)SX????w?1@?E?E?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?@?Ρu?!?Jr?]?@)?@?Ρu?1?Jr?]?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapS?G??!;W׽?9@)?v??-u`?1P????#??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 84.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?xfH???I???6W@Q??I2?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???Ɋ?<@???Ɋ?<@!???Ɋ?<@      ??!       "	???); @???); @!???); @*      ??!       2	v?[??`??v?[??`??!v?[??`??:	:??ȵ??:??ȵ??!:??ȵ??B      ??!       J	???0(??????0(???!???0(???R      ??!       Z	???0(??????0(???!???0(???b      ??!       JGPUY?xfH???b q???6W@y??I2?@?"l
@gradient_tape/sequential_6/conv2d_19/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???+???!???+???0"l
@gradient_tape/sequential_6/conv2d_18/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??Z????!?`?o֯??0";
sequential_6/conv2d_18/Conv2DConv2D&{?? N??!Op[k+??0"j
?gradient_tape/sequential_6/conv2d_20/Conv2D/Conv2DBackpropInputConv2DBackpropInput)???Ѳ?! ?R?????0"j
?gradient_tape/sequential_6/conv2d_19/Conv2D/Conv2DBackpropInputConv2DBackpropInput??????!??4r?f??0";
sequential_6/conv2d_19/Conv2DConv2D?t?d'??!5??m???0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits??U>???!???A????"l
@gradient_tape/sequential_6/conv2d_20/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltere??m說?!u??W??0";
sequential_6/conv2d_20/Conv2DConv2D5?AH$??!?⸛???0"-
IteratorGetNext/_1_Sende?p2???!?jL?\??Q      Y@Y?C=?C=0@a??
??T@qu?p6??S@y??qt????"?
both?Your program is POTENTIALLY input-bound because 84.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?79.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 