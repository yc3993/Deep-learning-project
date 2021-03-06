?	ɯb??A@ɯb??A@!ɯb??A@	?Tӷ?E???Tӷ?E??!?Tӷ?E??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6ɯb??A@??? ?_?@1>?$@M@A)?QG?ը?I???????Y!?'?>??*	??~j??b@2F
Iterator::Model?V?f???!?[??rM@)7U??檩?1???p?@@:Preprocessing2U
Iterator::Model::ParallelMapV2?W?B?_??!d?`??8@)?W?B?_??1d?`??8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??0a4??!?1M?ӄ1@)????=??1ճՄ?^,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??????!?O? @3@)?yȔA??1??? ?_+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicep^??jG??!?S?A@)p^??jG??1?S?A@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??:M??!F?55??D@)???E&?w?1??._
?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorۅ?:??t?!?????
@)ۅ?:??t?1?????
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap8i???!<8%E?4@)?M???a?1c?>?D??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 87.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?Tӷ?E??I?_UФW@QМ?C??@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??? ?_?@??? ?_?@!??? ?_?@      ??!       "	>?$@M@>?$@M@!>?$@M@*      ??!       2	)?QG?ը?)?QG?ը?!)?QG?ը?:	??????????????!???????B      ??!       J	!?'?>??!?'?>??!!?'?>??R      ??!       Z	!?'?>??!?'?>??!!?'?>??b      ??!       JGPUY?Tӷ?E??b q?_UФW@yМ?C??@?"k
@gradient_tape/sequential_16/conv2d_50/Conv2D/Conv2DBackpropInputConv2DBackpropInput䀢?Ѽ?!䀢?Ѽ?0"k
@gradient_tape/sequential_16/conv2d_49/Conv2D/Conv2DBackpropInputConv2DBackpropInput:s􏥼?!??	????0"<
sequential_16/conv2d_49/Conv2DConv2D??lm??!ȿ?@9???0"m
Agradient_tape/sequential_16/conv2d_50/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?ā??!??????0"m
Agradient_tape/sequential_16/conv2d_49/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?meD???!?\I?????0"m
Agradient_tape/sequential_16/conv2d_48/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterh??=????!Oʀ?????0"<
sequential_16/conv2d_50/Conv2DConv2D]ʨ?d???!?V?{???0"<
sequential_16/conv2d_48/Conv2DConv2D??p9?2??!4dR&?P??0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsfg??O??!o???3??"`
?gradient_tape/sequential_16/max_pooling2d_8/MaxPool/MaxPoolGradMaxPoolGrad?dR6?ď?!??F???Q      Y@Y??????0@aVUUUU?T@q?????5Q@yq'%??"?
both?Your program is POTENTIALLY input-bound because 87.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?68.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 