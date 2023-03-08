crate::ix!();

declare_export_caffe2_op_to_c10!{Percentile}

/**
  | Operator to calculate percentile values
  | for an input tensor of data, given samples
  | of data from the same distribution,
  | labeled with their percentile values.
  |
  | This operator is used to find percentile
  | representations for raw values, given
  | a sample set of raw values, labeled with
  | their corresponding percentiles from
  | the same distribution.
  | 
  | In particular, this operator takes
  | as input a tensor of floats to find the
  | percentile values for, a 2D tensor of
  | floats, where the first column of the
  | tensor represents sampled values,
  | and the second column represents the
  | percentile labels, and a tensor of integers
  | lengths.
  | 
  | This lengths tensor is used because
  | the operator works on multiple sets
  | of raw values at the same time. For example,
  | for an input:
  | 
  | original_values=[[3, 5, 3],[5, 1, 6]],
  | 
  | lengths = [2, 1, 1],
  | 
  | value_to_pct = [[3, 0.2], [5, 0.5], [1, 0.3], [3. 0.6]]
  | 
  | Our operator expects that each column
  | i of the input tensor is sampled from
  | distribution i. Lengths tells us that
  | the first two elements in value_to_pct
  | are sampled from distribution 1, the
  | next is from distribution two, and the
  | last is from distribution 3. We expect
  | the output of our operator to give us
  | [[0.2, 1.0, 0.6], [0.5, 0.3, 1.0]].
  | 
  | To calculate the percentile of an element,
  | we check to see if its value is already
  | mapped to a percentile in value_to_pct.
  | If so, we return that value. If not, we
  | linearly interpolate between the two
  | closest values in value_to_pct. If
  | the value is larger than all values in
  | value_to_pct, we return 1. If it's smaller
  | than all the values, we return 0.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct PercentileOp<Context> {
    storage:             OperatorStorage,
    context:             Context,
    values_tensor:       Tensor,
    percentiles_tensor:  Tensor,
}

register_cpu_operator!{
    Percentile, 
    PercentileOp<CPUContext>
}

num_inputs!{Percentile, 3}

num_outputs!{Percentile, 1}

inputs!{Percentile, 
    0 => ("original_values", "Input 2D tensor of floats, representing the original, raw data to calculate percentiles for."),
    1 => ("value_to_pct",    "Sorted 2D tensor, with 2 columns. Each element in the first column is a float representing the raw value of a sample. Its corresponding element in the next column represents the percentile it maps to."),
    2 => ("lengths",         "1D tensor, representing the length of each distribution. We expect that the sum of elements of this tensor is equal to the total length of value_to_pct.")
}

outputs!{Percentile, 
    0 => ("percentile_values", "1D tensor of floats, with the same dimensions as the flattened input tensor. Each element of this tensor, percentile_values[i], corresponds to the percentile calculated for original_values[i].")
}

identical_type_and_shape_of_input!{Percentile, 0}

no_gradient!{Percentile}

impl<Context> PercentileOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}

input_tags!{
    PercentileOp {
        X,
        ValPctPairs,
        Lens
    }
}

output_tags!{
    PercentileOp {
        Pct
    }
}

export_caffe2_op_to_c10_cpu!{
    Percentile,
    "_caffe2::Percentile(Tensor original_values, Tensor value_to_pct, Tensor lengths) -> Tensor percentile_values",
    caffe2::PercentileOp<caffe2::CPUContext>
}
