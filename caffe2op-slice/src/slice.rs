crate::ix!();

/**
  | Produces a slice of the input tensor.
  | 
  | - Currently, only slicing in a single
  | dimension is supported.
  | 
  | - Start and end indices are either passed
  | as two 1D input tensors or using the `starts`
  | and `ends` arguments.
  | 
  | - If a negative value is passed for any
  | of the start or end indices, it represents
  | |value| - 1 elements before the end of
  | that dimension. End indices are non-inclusive
  | unless negative (end index
  | 
  | -1 means up to and including the last
  | element).
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/slice_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SliceOp<Context> {
    storage:            OperatorStorage,
    context:            Context,
    starts:             Vec<i64>,
    ends:               Vec<i64>,
    statically_inited:  bool,
    starts_host:        Tensor,
    ends_host:          Tensor,
}

register_cpu_operator!{Slice, SliceOp<CPUContext>}

num_inputs!{Slice, (1,3)}

num_outputs!{Slice, 1}

inputs!{Slice, 
    0 => ("X", "(*Tensor*): tensor to extract slices from"),
    1 => ("starts", "(*Tensor`<int>`*): 1D tensor of start-indices for each dimension of data (dimensions following the sliced one might be omitted)"),
    2 => ("ends", "(*Tensor`<int>`*): 1D tensor of end-indices for each dimension of data (dimensions following the sliced one might be omitted)")
}

outputs!{Slice, 
    0 => ("Y", "(*Tensor*): sliced output tensor")
}

args!{Slice, 
    0 => ("starts", "(*Tuple(int)*): list of starting indices"),
    1 => ("ends", "(*Tuple(int)*): list of ending indices")
}

/// the filler cannot be enabled without output dims
disallow_input_fillers!{Slice}

tensor_inference_function!{Slice, /* ([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      if (in.size() > 1) {
        // Cannot compute shape inference when the splits are defined
        // in data.
        return vector<TensorShape>();
      }
      auto const& data = in[0];

      ArgumentHelper helper(def);
      auto starts = helper.GetRepeatedArgument<int>("starts", vector<int>());
      auto ends = helper.GetRepeatedArgument<int>("ends", vector<int>());
      vector<int> dst_sizes(data.dims_size());

      for (int i = 0; i < data.dims_size(); ++i) {
        if (i >= starts.size()) {
          dst_sizes[i] = data.dims(i);
          continue;
        }
        if (data.dims(i) > 0) {
          auto start = starts[i];
          auto end = ends[i];
          if (start < 0) {
            start = data.dims(i) + 1 + start;
          }
          if (end < 0) {
            end = data.dims(i) + 1 + end;
          }
          dst_sizes[i] = end - start;
        } else {
          dst_sizes[i] = 0;
        }
      }
      return vector<TensorShape>{
          CreateTensorShape(dst_sizes, data.data_type())};
    }) */}

inherit_onnx_schema!{Slice}

