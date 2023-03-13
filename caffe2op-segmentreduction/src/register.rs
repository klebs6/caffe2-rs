crate::ix!();

/**
  | Helper macro when the main op is defined
  | elsewhere, and we only need to define the
  | schema, and the gradient op.
  |
  | TODO: enable input fillers
  */
#[macro_export] macro_rules! register_segment_def_schema_gradient_only {
    () => {
        /*
                (                            
            segment_name, gradient_name, ...)                                         
          static_assert(                                                              
              equal(#segment_name, __VA_ARGS__::basename, __VA_ARGS__::OpDef::name),  
              #segment_name);                                                         
          static_assert(                                                              
              equal(                                                                  
                  #gradient_name,                                                     
                  __VA_ARGS__::basename,                                              
                  __VA_ARGS__::OpDef::name,                                           
                  "Gradient"),                                                        
              #gradient_name);                                                        
          OPERATOR_SCHEMA(segment_name)                                               
              .NumInputs(__VA_ARGS__::ForwardOp::kNumInputs)                          
              .NumOutputs(1)                                                          
              .DisallowInputFillers()                                                 
              .SetDoc(FormatDoc<__VA_ARGS__>())                                       
              .Output(0, "OUTPUT", "Aggregated tensor")                               
              .FillUsing(__VA_ARGS__::PopulateSchema);                                
          REGISTER_CPU_OPERATOR_STR(string(#gradient_name), __VA_ARGS__::BackwardOp); 
          OPERATOR_SCHEMA(gradient_name)                                              
              .NumInputs(__VA_ARGS__::BackwardOp::kNumInputs)                         
              .NumOutputs(1)                                                          
              .DisallowInputFillers();                                                
          REGISTER_GRADIENT_STR(string(#segment_name), __VA_ARGS__::GetGradient)
        */
    }
}

#[macro_export] macro_rules! register_segment_def {
    ($segment_name:expr, $gradient_name:expr, $($arg:expr),*) => {
        /*
        
          static_assert(                                                             
              equal(#segment_name, __VA_ARGS__::basename, __VA_ARGS__::OpDef::name), 
              #segment_name);                                                        
          REGISTER_CPU_OPERATOR_STR(string(#segment_name), __VA_ARGS__::ForwardOp);  
          REGISTER_SEGMENT_DEF_SCHEMA_GRADIENT_ONLY(                                 
              segment_name, gradient_name, __VA_ARGS__)
        */
    }
}

register_segment_def!{
    SortedSegmentRangeSum,
    SortedSegmentRangeSumGradient,
    AbstractSortedSegmentRangeDef::<
        f32, 
        i32, 
        CPUContext, 
        SumRangeReducerDef>
}

register_segment_def!{
    SortedSegmentRangeLogSumExp,
    SortedSegmentRangeLogSumExpGradient,
    AbstractSortedSegmentRangeDef::<
        f32,
        i32,
        CPUContext,
        LogSumExpRangeReducerDef>
}

register_segment_def!{
    SortedSegmentRangeLogMeanExp,
    SortedSegmentRangeLogMeanExpGradient,
    AbstractSortedSegmentRangeDef::<
        f32,
        i32,
        CPUContext,
        LogMeanExpRangeReducerDef>
}

register_segment_def!{
    SortedSegmentRangeMean,
    SortedSegmentRangeMeanGradient,
    AbstractSortedSegmentRangeDef::<f32, i32, CPUContext, MeanRangeReducerDef> 
}

register_segment_def!{
    SortedSegmentRangeMax,
    SortedSegmentRangeMaxGradient,
    AbstractSortedSegmentRangeDef::<f32, i32, CPUContext, MaxRangeReducerDef> 
}

register_segment_def!{
    SortedSegmentSum,
    SortedSegmentSumGradient,
    AbstractSortedSegmentDef::<f32, i32, CPUContext, SumReducerDef> 
}

register_segment_def!{
    SparseSortedSegmentSum,
    SparseSortedSegmentSumGradient,
    AbstractSparseSortedSegmentDef::<f32, i32, CPUContext, SumReducerDef> 
}

register_segment_def!{
    UnsortedSegmentSum,
    UnsortedSegmentSumGradient,
    AbstractUnsortedSegmentDef::<f32, i32, CPUContext, SumReducerDef> 
}

register_segment_def!{
    SparseUnsortedSegmentSum,
    SparseUnsortedSegmentSumGradient,
    AbstractSparseUnsortedSegmentDef::<f32, i32, CPUContext, SumReducerDef>
}

register_segment_def!{
    SortedSegmentMean,
    SortedSegmentMeanGradient,
    AbstractSortedSegmentDef::<f32, i32, CPUContext, MeanReducerDef>
}

register_segment_def!{
    SparseSortedSegmentMean,
    SparseSortedSegmentMeanGradient,
    AbstractSparseSortedSegmentDef::<f32, i32, CPUContext, MeanReducerDef>
}

register_segment_def!{
    UnsortedSegmentMean,
    UnsortedSegmentMeanGradient,
    AbstractUnsortedSegmentDef::<f32, i32, CPUContext, MeanReducerDef>
}

register_segment_def!{
    SparseUnsortedSegmentMean,
    SparseUnsortedSegmentMeanGradient,
    AbstractSparseUnsortedSegmentDef::<f32, i32, CPUContext, MeanReducerDef>
}

register_segment_def!{
    ReduceFrontWeightedSum,
    ReduceFrontWeightedSumGradient,
    AbstractReduceFrontDef::<f32, CPUContext, WeightedSumReducerDef>
}

register_segment_def!{
    SortedSegmentWeightedSum,
    SortedSegmentWeightedSumGradient,
    AbstractSortedSegmentDef::<f32, i32, CPUContext, WeightedSumReducerDef>
}

register_segment_def!{
    SparseSortedSegmentWeightedSum,
    SparseSortedSegmentWeightedSumGradient,
    AbstractSparseSortedSegmentDef::<
        f32,
        i32,
        CPUContext,
        WeightedSumReducerDef>
}

register_segment_def!{
    UnsortedSegmentWeightedSum,
    UnsortedSegmentWeightedSumGradient,
    AbstractUnsortedSegmentDef::<f32, i32, CPUContext, WeightedSumReducerDef>
}

register_segment_def!{
    SparseUnsortedSegmentWeightedSum,
    SparseUnsortedSegmentWeightedSumGradient,
    AbstractSparseUnsortedSegmentDef::<
        f32,
        i32,
        CPUContext,
        WeightedSumReducerDef>
}

/**
  | The *LengthsWeightedSum* op takes
  | three inputs *DATA*, *LENGTHS*, and
  | *SCALARS*, and produces a single output
  | *OUTPUT*.
  | 
  | The op finds the weighted sum in each
  | of the segments of *DATA*, where segments
  | are defined by their lengths. Before
  | calculating the sums, the input *DATA*
  | is weighted by the contents of *SCALARS*.
  | 
  | For example, if $DATA = [2,4,3,1,2,10]$,
  | $SCALARS = [8, 2, 1, 4, 1, 0.6]$, and $LENGTHS
  | = [2,3,1]$, then $OUTPUT = [sum([8*2,2*4]),
  | sum([1*3,4*1,1*2]), sum([0.6*10])]
  | = [24,9,6]$.
  | 
  | Github Link:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc
  |
  */
register_segment_def!{
    LengthsWeightedSum,
    LengthsWeightedSumGradient,
    AbstractLengthsDef::<f32, i32, CPUContext, WeightedSumReducerDef, false>
}

register_gradient_with_main_input!{
    LengthsWeightedSumWithMainInputGradient,
    AbstractLengthsDef::<f32, i32, CPUContext, WeightedSumReducerDef>
}

register_gradient_with_main_input!{
    SparseLengthsWeightedSumWithMainInputGradient,
    AbstractSparseLengthsDef::<f32, i32, CPUContext, WeightedSumReducerDef>
}

/**
  | registering 5 input gradient with main
  | output gradient of SparseLengthsWeightedSum
  |
  */
num_inputs!{SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient, 5}

num_outputs!{SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient, 2}

register_cpu_operator!{
    SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient,
    AbstractLengthsWithMainInputGradientOp::<
        f32,
        f32,
        i32,
        CPUContext,
        WeightedSumReducerDef::ReducerGradient::<f32, CPUContext>,
        SparseFused,
        GradientNeedIndices>
}

/**
  | registering 4 input version
  |
  */
num_inputs!{SparseLengthsIndicesInGradientWeightedSumGradient, 4}

num_outputs!{SparseLengthsIndicesInGradientWeightedSumGradient, 1}

register_cpu_operator!{
    SparseLengthsIndicesInGradientWeightedSumGradient,
    AbstractLengthsGradientOp<
        f32,
        i32,
        CPUContext,
        WeightedSumReducerDef::ReducerGradient<f32, CPUContext>,
        GradientNeedIndices>
}

/**
  | registering 3 input version gradient
  | of SparseLengthsSum
  |
  */
num_inputs!{SparseLengthsIndicesInGradientSumGradient, 3}

num_outputs!{SparseLengthsIndicesInGradientSumGradient, 1}

register_cpu_operator!{
    SparseLengthsIndicesInGradientSumGradient,
    AbstractLengthsGradientOp::<
        f32,
        i32,
        CPUContext,
        SumReducerDef::ReducerGradient::<f32, CPUContext>,
        GradientNeedIndices>
}

/**
  | gradient of LengthsSum
  |
  */
num_inputs!{LengthsIndicesInGradientSumGradient, 3}

num_outputs!{LengthsIndicesInGradientSumGradient, 1}

register_cpu_operator!{
    LengthsIndicesInGradientSumGradient,
    AbstractLengthsGradientOp::<
        f32,
        i32,
        CPUContext,
        SumReducerDef::ReducerGradient::<f32, CPUContext>,
        GradientNeedIndices>
}

/**
  | registering 3 input version gradient
  | of SparseLengthsMean
  |
  */
num_inputs!{SparseLengthsIndicesInGradientMeanGradient, 3}

num_outputs!{SparseLengthsIndicesInGradientMeanGradient, 1}

register_cpu_operator!{
    SparseLengthsIndicesInGradientMeanGradient,
    AbstractLengthsGradientOp::<
        f32,
        i32,
        CPUContext,
        MeanReducerDef::ReducerGradient::<f32, CPUContext>,
        GradientNeedIndices>
}

/**
  | gradient of LengthsMean
  |
  */
num_inputs!{LengthsIndicesInGradientMeanGradient, 3}

num_outputs!{LengthsIndicesInGradientMeanGradient, 1}

register_cpu_operator!{
    LengthsIndicesInGradientMeanGradient,
    AbstractLengthsGradientOp::<
        f32,
        i32,
        CPUContext,
        MeanReducerDef::ReducerGradient::<f32, CPUContext>,
        GradientNeedIndices>
}

/**
  | The *LengthsMax* op takes two inputs
  | *DATA* and *LENGTHS*, and produces
  | a single output *OUTPUT*.
  | 
  | The op finds the maximum value in each
  | of the segments of *DATA*, where segments
  | are defined by their lengths. For example,
  | if $DATA = [2,4,3,1,2,10]$ and $LENGTHS
  | = [2,3,1]$ then $OUTPUT = [max([2,4]),
  | max([3,1,2]), max([10])] = [4,3,10]$.
  | 
  | Github Link:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc
  |
  */
register_lengths_ops_main_input_and_forward_output_gradient!{
    LengthsMax,
    LengthsMaxWithMainInputAndForwardOutputGradient,
    AbstractLengthsDef::<f32, i32, CPUContext, MaxReducerDef>
}


pub type LengthsMaxCPUOp = <AbstractLengthsDef<
    f32,
    i32,
    CPUContext,
    MaxReducerDef,
    true> as HasForwardOp>::ForwardOp;

export_caffe2_op_to_c10_cpu!{
    LengthsMax,
    "_caffe2::LengthsMax(Tensor data, Tensor lengths) -> Tensor",
    LengthsMaxCPUOp
}

/**
  | The *LengthsMean* op takes two inputs
  | *DATA* and *LENGTHS*, and produces
  | a single output *OUTPUT*.
  | 
  | The op finds the mean value in each of
  | the segments of *DATA*, where segments
  | are defined by their lengths. For example,
  | if $DATA = [2,4,3,1,2,10]$ and $LENGTHS
  | = [2,3,1]$ then $OUTPUT = [mean([2,4]),
  | mean([3,1,2]), mean([10])] = [3,2,10]$.
  | 
  | Github Link:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc
  |
  */
register_segment_def!{
    LengthsMean,
    LengthsMeanGradient,
    AbstractLengthsDef::<f32, i32, CPUContext, MeanReducerDef, true>
}

export_caffe2_op_to_c10_cpu!{
    LengthsMean,
    "_caffe2::LengthsMean(Tensor data, Tensor lengths) -> Tensor",
    LengthsMeanCPUOp}

/**
  | The *LengthsSum* op takes two inputs
  | *DATA* and *LENGTHS*, and produces
  | a single output *OUTPUT*.
  | 
  | The op finds the sum in each of the segments
  | of *DATA*, where segments are defined
  | by their lengths. For example, if $DATA
  | = [2,4,3,1,2,10]$ and $LENGTHS = [2,3,1]$
  | then $OUTPUT = [sum([2,4]), sum([3,1,2]),
  | sum([10])] = [6,6,10]$.
  | 
  | Github Link:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc
  |
  */
register_segment_def!{
    LengthsSum,
    LengthsSumGradient,
    AbstractLengthsDef::<f32, i32, CPUContext, SumReducerDef, true>
}

export_caffe2_op_to_c10_cpu!{
    LengthsSum,
    "_caffe2::LengthsSum(Tensor data, Tensor lengths) -> Tensor",
    LengthsSumCPUOp
}

declare_export_caffe2_op_to_c10!{LengthsSum}
declare_export_caffe2_op_to_c10!{LengthsMean}
declare_export_caffe2_op_to_c10!{LengthsMax}

// Range reducer ops: leverage that input segment is continuous and allow reducer functors to do
// something special
//
// Note: for now there are no real use cases for it yet :)
//
// Also, doesn't support additional arguments for now
