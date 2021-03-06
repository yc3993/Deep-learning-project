?	v?Kp??A@v?Kp??A@!v?Kp??A@	??Dd??@??Dd??@!??Dd??@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6v?Kp??A@R?(?19?@1???{?@A?аu???I:=?Ƃ???YPō[???*	|?5^?I]@2F
Iterator::Modelk'JB"??!pEbIH@)????1P7ϫ?9@:Preprocessing2U
Iterator::Model::ParallelMapV2Ϡ????!?S)???6@)Ϡ????1?S)???6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate{????!??B?8@)?=&R?͓?15Fd???0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????b??!???
T4@)??X ??1 ??a??/@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?????m??!s?`?02 @)?????m??1s?`?02 @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip+?ٮ?!??????I@)u ???Ww?1?8?_u@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorp?'v?u?!??????@)p?'v?u?1??????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?J=By??!??N?j<:@)?VC?K_?1t??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 87.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??Dd??@I?ݫ?3?U@Q???`??@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	R?(?19?@R?(?19?@!R?(?19?@      ??!       "	???{?@???{?@!???{?@*      ??!       2	?аu????аu???!?аu???:	:=?Ƃ???:=?Ƃ???!:=?Ƃ???B      ??!       J	Pō[???Pō[???!Pō[???R      ??!       Z	Pō[???Pō[???!Pō[???b      ??!       JGPUY??Dd??@b q?ݫ?3?U@y???`??@?"<
sequential_13/conv2d_40/Conv2DConv2D?'s????!?'s????0"k
@gradient_tape/sequential_13/conv2d_41/Conv2D/Conv2DBackpropInputConv2DBackpropInput?<?f???!^2???|??0"k
@gradient_tape/sequential_13/conv2d_40/Conv2D/Conv2DBackpropInputConv2DBackpropInput6??̴??!|F???C??0"m
Agradient_tape/sequential_13/conv2d_40/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???ǃ???!^???????0"m
Agradient_tape/sequential_13/conv2d_41/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?:(??u??!?????0"m
Agradient_tape/sequential_13/conv2d_39/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?Gi?[??!?Т?x???0"<
sequential_13/conv2d_39/Conv2DConv2D??1?$???!@?U";???0"<
sequential_13/conv2d_41/Conv2DConv2D???t:??!?̥?N$??0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsw?9????!??hN???"-
IteratorGetNext/_1_Send?????P??!d?s???Q      Y@Y??????0@aVUUUU?T@q`?Np??9@y?x??????"?

both?Your program is POTENTIALLY input-bound because 87.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?25.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 