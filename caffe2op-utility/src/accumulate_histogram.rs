crate::ix!();

/**
  | This operator calculate thes histogram
  | of values in input tensor.
  | 
  | There're 2 outputs, one for histogram
  | of current input tensor, and another
  | for histogram of the all input tensors
  | accumulated through history.
  | 
  | The output would contain num_buckets
  | + 2 values. index[1 ... num_buckets]
  | for values in [lower_bound, upper_bound)
  | interval. And the rest 2 for values smaller
  | than lower_bound or greater than upper_bound
  | respectively.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AccumulateHistogramOp<T,Context> {
    storage:            OperatorStorage,
    context:            Context,
    lower_bound:        f32,
    upper_bound:        f32,
    num_buckets:        i32,
    num_output_buckets: i32,
    accumulate_hist:    Vec<i64>,
    phantom:            PhantomData<T>,
}

num_inputs!{AccumulateHistogram, 1}

num_outputs!{AccumulateHistogram, 2}

inputs!{AccumulateHistogram, 
    0 => ("X", "Input tensor.")
}

outputs!{AccumulateHistogram, 
    0 => ("CurHist", "Output histogram of the current tensor."),
    1 => ("AccHist", "Accumulated histogram of the history tensor.")
}

args!{AccumulateHistogram, 
    0 => ("lower_bound", "the lower bound value"),
    1 => ("upper_bound", "the upper bound value"),
    2 => ("num_buckets", "number of buckets to use in [lower_bound, upper_bound)")
}

input_tags!{
    AccumulateHistogramOp
    {
        XIn
    }
}

output_tags!{
    AccumulateHistogramOp
    {
        CurHist,
        AccHist
    }
}
