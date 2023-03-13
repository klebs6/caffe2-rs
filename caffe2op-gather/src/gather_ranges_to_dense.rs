crate::ix!();

pub type GatherRangesToDenseCPUOp = GatherRangesToDenseOp<CPUContext>;

/**
  | Given DATA tensor of rank 1, and RANGES
  | tensor of rank 3, gather values corresponding
  | to each range into a separate output
  | tensor.
  | 
  | If the optional input KEY tensor is also
  | given, the output will be sorted by KEY
  | for each example.
  | 
  | RANGES dimensions description:
  | 
  | 1: represents list of examples within
  | a batch
  | 
  | 2: represents list features
  | 
  | 3: two values which are start and length
  | or a range (to be applied on DATA)
  | 
  | Each feature has fixed lengths which
  | are passed as lengths argument and a
  | separate tensor will be produced for
  | each feature.
  | 
  | i.e. DATA.dim(1) = len(lengths) = NumOuptuts.
  | 
  | Missing features (represented by empty
  | ranges) filled with default_value.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct GatherRangesToDenseOp<Context> {
    storage:           OperatorStorage,
    context:           Context,
    lengths:           Vec<i32>,
    total_ranges:      i64,
    empty_ranges:      Vec<i64>,
    mismatched_ranges: Vec<i64>,

    /**
      | To avoid false alarm due to insufficient
      | sample (e.g., first batch being mismatched
      | and causing 100% to be mismatched), use
      | a threshold to ensure enough samples are
      | gathered before decideding whether there
      | is an alarm or not.
      */
    min_observation:      i64,
    max_mismatched_ratio: f32,
    max_empty_ratio:      f32,
}

num_inputs!{GatherRangesToDense, (2,3)}

num_outputs!{GatherRangesToDense, (1,INT_MAX)}

inputs!{GatherRangesToDense, 
    0 => ("DATA",   "Tensor of rank 1."),
    1 => ("RANGES", "Tensor of int32/int64 ranges, of dims (N, M, 2). Where N is number of examples and M is a size of each example. Last dimension represents a range in the format (start, lengths)"),
    2 => ("KEY",    "Tensor of rank 1 and type int64.")
}

outputs!{GatherRangesToDense, 
    0 => ("OUTPUT", "1-D tensor of size sum of range lengths")
}

args!{GatherRangesToDense, 
    0 => ("lengths",                "Expected lengths for ranges"),
    1 => ("min_observation",        "The number of observations needed before deciding that the ratio of mismatched ranges is alarming, also determines whether an info sumarizing the empty and mismatch ratio will be printed at the end."),
    2 => ("max_mismatched_ratio",   "An error is raised when ratio of mismatched ranges exceeds this."),
    3 => ("max_empty_ratio",        "An error is raised when ratio of empty ranges exceeds this (default is 1, which means by default no error will be triggered).")
}

tensor_inference_function!{GatherRangesToDense, /* [](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      auto lengths = helper.GetRepeatedArgument<int>("lengths");
      CAFFE_ENFORCE_EQ(in[0].dims_size(), 1, "DATA should be 1-D tensor.");
      CAFFE_ENFORCE_EQ(in[1].dims_size(), 3, "RANGES should be 3-D tensor.");
      if (in.size() > 2) {
        CAFFE_ENFORCE_EQ(in[2].dims_size(), 1, "KEY should be 1-D tensor.");
      }
      CAFFE_ENFORCE_GT(lengths.size(), 0, "lengths should be non-empty.");
      std::vector<TensorShape> out(lengths.size());
      for (int i = 0; i < lengths.size(); ++i) {
        out[i].set_data_type(in[0].data_type());
        out[i].add_dims(in[1].dims(0));
        out[i].add_dims(lengths[i]);
      }
      return out;
    } */}

register_cpu_operator!{GatherRangesToDense, GatherRangesToDenseOp<CPUContext>}

no_gradient!{GatherRangesToDense}

input_tags!{
    GatherRangesToDenseOp {
        Data,
        Ranges,
        Key
    }
}
