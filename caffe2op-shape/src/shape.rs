crate::ix!();

/**
  | Produce a 1D int64 tensor with the shape
  | of the input tensor.
  | 
  | If called with an optional argument
  | `axes`, the result will only contain
  | the dimensions of specified axes.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/shape_op.cc
  | 
  | RecordShapeOp records the shape of
  | the input tensor to a vector of int. You
  | mostly don't need this operator explicitly,
  | and it is mostly used in the autodiff
  | process.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ShapeOp<Context> {
    storage: OperatorStorage,
    context: Context,
    axes:    Vec<i32>,
}

register_cpu_operator!{Shape, ShapeOp<CPUContext>}

num_inputs!{Shape, 1}

num_outputs!{Shape, 1}

inputs!{Shape, 
    0 => ("X", "*(type: Tensor)* Input tensor.")
}

outputs!{Shape, 
    0 => ("shape", "*(type: Tensor)* Output tensor containing shape of input tensor.")
}

args!{Shape, 
    0 => ("axes", "*(type: int[])* Array of interested axes. If given, this operator only returns the dimensions of the given axes. Otherwise, the operator returns the dimensions of all axes.")
}

tensor_inference_function!{Shape, /* ([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper args(def);
      const vector<int>& axes = args.GetRepeatedArgument<int>("axes");
      vector<TensorShape> out(1);
      if (axes.empty()) {
        out[0].add_dims(in[0].dims().size());
      } else {
        out[0].add_dims(axes.size());
      }
      out[0].set_data_type(TensorProto::INT64);
      return out;
    }) */}

should_not_do_gradient!{Shape}

register_cuda_operator!{Shape, ShapeOp<CUDAContext>}

input_tags!{
    ShapeOp {
        Data
    }
}
