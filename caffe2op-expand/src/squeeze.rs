crate::ix!();

/**
  | The *Squeeze* op removes single-dimensional
  | entries from the shape of the input tensor
  | *data,* and produces a single output
  | tensor *squeezed*.
  | 
  | The op also takes an argument *dims*
  | with a list of dimensions to squeeze.
  | 
  | If the same blob is provided as input
  | and output, the operation is copy-free.
  | 
  | This is the exact inverse operation
  | of
  | 
  | ExpandDims* given the same *dims* argument.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/expand_squeeze_dims_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/expand_squeeze_dims_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SqueezeOp<Context> {
    storage: OperatorStorage,
    context: Context,
    dims:    Vec<i32>,
}

num_inputs!{Squeeze, 1}

num_outputs!{Squeeze, 1}

inputs!{Squeeze, 
    0 => ("data", "Input tensor of data to be operated on.")
}

outputs!{Squeeze, 
    0 => ("squeezed", "Reshaped tensor with same data as input.")
}

args!{Squeeze, 
    0 => ("dims", "*(type: [int])* List of dimensions of *data* to squeeze out.")
}

allow_inplace!{Squeeze, vec![(0, 0)]}

inherit_onnx_schema!{Squeeze}

tensor_inference_function!{Squeeze, /*[](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      auto dims = helper.template GetRepeatedArgument<int>("dims");
      auto originalSize = dims.size();
      std::sort(dims.begin(), dims.end());
      dims.erase(std::unique(dims.begin(), dims.end()), dims.end());
      if (dims.size() < originalSize) {
        LOG(WARNING) << "Parameter `dims` has repeated dimensions.";
      }
      CAFFE_ENFORCE(dims.front() >= 0, "Dimension ids must be non-negative.");

      vector<TensorShape> out(1);
      std::vector<int> newDims =
          SqueezeOp<CPUContext>::ComputeDims(GetDimsVector(in[0]), dims);
      out[0] = CreateTensorShape(newDims, in[0].data_type());
      return out;
    }*/
}
