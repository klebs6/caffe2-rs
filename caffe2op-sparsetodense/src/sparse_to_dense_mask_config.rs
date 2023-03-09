crate::ix!();

num_inputs!{SparseToDenseMask, (3,4)}

num_outputs!{SparseToDenseMask, (1,2)}

// TODO: enable the filler
disallow_input_fillers!{SparseToDenseMask}

inputs!{SparseToDenseMask, 
    0 => ("indices",              "1-D int32/int64 tensor of concatenated ids of data"),
    1 => ("values",               "Data tensor, first dimension has to match `indices`"),
    2 => ("default_value",        "Default value for the output if the id is not present in `indices`. Must have the same type as `values` and the same shape, but without the first dimension"),
    3 => ("lengths",              "Optional lengths to represent a batch of `indices` and `values`.")
}

outputs!{SparseToDenseMask, 
    0 => ("output",               "Output tensor of the same type as `values` of shape `[len(lengths), len(mask)] + shape(default_value)` (if `lengths` is not provided the first dimension is omitted)"),
    1 => ("presence_mask",        "Bool tensor of shape `[len(lengths), len(mask)]` (if `lengths` is not provided the first dimension is omitted). True when a value for given id was present, false otherwise.")
}

args!{SparseToDenseMask, 
    0 => ("mask",                 "list(int) argument with desired ids on the 'dense' output dimension"),
    1 => ("return_presence_mask", "bool whether to return presence mask, false by default"),
    2 => ("max_skipped_indices",  "int argument representing the maximum number of invalid row ids that can be skipped before returning an error. 50 by default")
}

tensor_inference_function!{SparseToDenseMask, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          ArgumentHelper helper(def);
          auto mask = helper.template GetRepeatedArgument<int64_t>("mask");
          bool return_presence_mask = helper.template GetSingleArgument<bool>(
              "return_presence_mask", false);
          vector<TensorShape> out(1);

          if (in.size() == 4) {
            out[0].add_dims(in[3].dims(0));
          }
          out[0].add_dims(mask.size());
          for (const auto dim : in[2].dims()) {
            out[0].add_dims(dim);
          }
          out[0].set_data_type(in[2].data_type());

          if (return_presence_mask) {
            out.emplace_back();
            if (in.size() == 4) {
              out[1].add_dims(in[3].dims(0));
            }
            out[1].add_dims(mask.size());
            out[1].set_data_type(TensorProto::BOOL);
          }

          return out;
        */
    }
}

input_tags!{
    SparseToDenseMaskOp
    {
        Indices,
        Values,
        Default,
        Lengths
    }
}

output_tags!{
    SparseToDenseMaskOp
    {
        Outputvalue,
        Presencemask
    }
}
