crate::ix!();

/**
  | The *Gather* op accepts a *DATA* tensor
  | of rank $r >= 1$ and *INDICES* tensor
  | of rank $q$ as inputs.
  | 
  | It then gathers entries of the outer-most
  | dimension of *DATA*, indexed by *INDICES*,
  | and concatenate them in an output tensor
  | of rank $q + (r - 1)$.
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/gather_op.cc
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/gather_op.h
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct GatherOp<Context> {
    storage:      OperatorStorage,
    context:      Context,
    axis:         i32,
    wrap_indices: bool,
    match_outer:  bool,
}

register_cpu_operator!{Gather, GatherOp<CPUContext>}

num_inputs!{Gather, 2}

num_outputs!{Gather, 1}

inputs!{Gather, 
    0 => ("DATA", "Input data tensor of rank $r>=1$"),
    1 => ("INDICES", "Input indices tensor of rank $q$. This tensor must contain integers.")
}

outputs!{Gather, 
    0 => ("OUTPUT", "Output tensor of rank $q+(r-1)$")
}

tensor_inference_function!{Gather, 
    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          ArgumentHelper helper(def);
          const int axis = helper.GetSingleArgument<int>("axis", 0);
          const bool match_outer =
              helper.GetSingleArgument<bool>("match_outer", false);
          const auto& data_dims = GetDimsVector(in[0]);
          const auto& indices_dims = GetDimsVector(in[1]);

          vector<int> output_dims =
              caffe2::gather_helper::calc_output_shape_vector<int>(
                  data_dims, indices_dims, axis, match_outer);
          vector<TensorShape> out(1);
          out[0] = CreateTensorShape(output_dims, in[0].data_type());
          return out;
        */
    }
}

inherit_onnx_schema!{Gather}

input_tags!{
    GatherOp {
        Data,
        Indices
    }
}
