crate::ix!();

/**
  | The *ExpandDims* op inserts single-dimensional
  | entries into the shape of the input tensor
  | *data,* and produces a single output
  | tensor *expanded*.
  | 
  | The op also takes an argument *dims*
  | with a list of dimensions for where to
  | add the single dimensional entries.
  | 
  | If the same blob is provided as input
  | and output, the operation is copy-free.
  | This is the exact inverse operation
  | of *Squeeze*.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/expand_squeeze_dims_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/expand_squeeze_dims_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ExpandDimsOp<Context> {
    storage: OperatorStorage,
    context: Context,
    dims:    Vec<i32>,
}

num_inputs!{ExpandDims, 1}

num_outputs!{ExpandDims, 1}

inputs!{ExpandDims, 
    0 => ("data", "Input tensor of data to be operated on.")
}

outputs!{ExpandDims, 
    0 => ("expanded", "Reshaped tensor with same data as input.")
}

args!{ExpandDims, 
    0 => ("dims", "*(type: [int])* List of dimensions of *data* to add single dimensional entry.")
}

inherit_onnx_schema!{ExpandDims}

tensor_inference_function!{ExpandDims, 
    /*[](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      auto dims = helper.template GetRepeatedArgument<int>("dims");
      auto originalSize = dims.size();
      CAFFE_ENFORCE(originalSize > 0, "Parameter `dims` must be provided.");

      std::sort(dims.begin(), dims.end());
      dims.erase(std::unique(dims.begin(), dims.end()), dims.end());
      if (dims.size() < originalSize) {
        LOG(WARNING) << "Parameter `dims` has repeated dimensions.";
      }

      CAFFE_ENFORCE(dims.front() >= 0, "Dimension ids must be non-negative.");
      CAFFE_ENFORCE_GE(
          in[0].dims_size() + dims.size(),
          dims.back() + 1,
          "Input needs at least ",
          (1 + dims.back() - dims.size()),
          " dimensions given `dims`.");

      vector<TensorShape> out(1);

      int cur_pos = 0;
      int idx = 0;
      for (const auto new_dim : dims) {
        for (int i = cur_pos; i < new_dim; i++) {
          out[0].add_dims(in[0].dims(idx++));
        }
        out[0].add_dims(1);
        cur_pos = new_dim + 1;
      }
      for (; idx < in[0].dims_size(); idx++) {
        out[0].add_dims(in[0].dims(idx));
      }
      out[0].set_data_type(in[0].data_type());
      return out;
    }*/
}

allow_inplace!{ExpandDims, vec![(0, 0)]}
