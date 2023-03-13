crate::ix!();

/**
  | Perform the same operation as Gather,
  | but operating on 8-bit rowwise quantized
  | matrices with fused storage (where
  | each row stores quantized values, and
  | then the scale and offset).
  | 
  | DATA needs to have rank 2 and INDICES
  | needs to have rank 1.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct GatherFused8BitRowwiseOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{GatherFused8BitRowwise, 2}

num_outputs!{GatherFused8BitRowwise, 1}

inputs!{GatherFused8BitRowwise, 
    0 => ("DATA", "uint8 tensor with rank 2 obtained with operator FloatToFused8BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the rows that are being gathered")
}

outputs!{GatherFused8BitRowwise, 
    0 => ("OUTPUT", "output")
}

tensor_inference_function!{GatherFused8BitRowwise, /* [](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      for (auto d : in[1].dims()) {
        out[0].add_dims(d);
      }
      for (int i = 1; i < in[0].dims_size(); ++i) {
        out[0].add_dims(in[0].dims(i));
      }
      out[0].set_data_type(in[0].data_type());
      return out;
    } */
}

register_cpu_operator!{
    GatherFused8BitRowwise,
    GatherFused8BitRowwiseOp<CPUContext>
}

input_tags!{
    GatherFused8BitRowwiseOp {
        Data,
        Indices
    }
}
