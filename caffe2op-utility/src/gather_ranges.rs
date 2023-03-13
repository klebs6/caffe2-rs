crate::ix!();

/**
  | Given DATA tensor of rank 1, and RANGES
  | tensor of rank 3, gather corresponding
  | ranges into a 1-D tensor OUTPUT.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct GatherRangesOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{GatherRanges, 2}

num_outputs!{GatherRanges, 2}

inputs!{GatherRanges, 
    0 => ("DATA",   "Tensor of rank 1."),
    1 => ("RANGES", "Tensor of int32/int64 ranges, of dims (N, M, 2). Where N is number of examples and M is a size of each example. Last dimension represents a range in the format (start, lengths)")
}

outputs!{GatherRanges, 
    0 => ("OUTPUT",  "1-D tensor of size sum of range lengths"),
    1 => ("LENGTHS", "1-D tensor of size N with lengths over gathered data for each row in a batch. sum(LENGTHS) == OUTPUT.size()")
}

tensor_inference_function!{
    GatherRanges, 
    OpSchema::NeedsAllInputShapes(
        |def: &OperatorDef, input: &Vec<TensorShape>| {
            todo!();
            /*
              std::vector<TensorShape> out(2);

              int total = 1;
              for (auto d : in[0].dims()) {
                total *= d;
              }
              out[0].add_dims(total);
              out[0].set_data_type(in[0].data_type());
              out[1].add_dims(in[1].dims(0));
              out[1].set_data_type(in[1].data_type());
              return out;
            */
        }
    )
}

input_tags!{
    GatherRangesOp
    {
        Data,
        Ranges,
        Lengths
    }
}
