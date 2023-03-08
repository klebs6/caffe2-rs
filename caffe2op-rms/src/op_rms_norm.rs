crate::ix!();

/// https://openreview.net/pdf?id=SygkZ3MTJE
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct RMSNormOp<Context> {
    storage: OperatorStorage,
    context: Context,
    axis:    i32,
    eps:     f32,
}

num_inputs!{RMSNorm, 3}

num_outputs!{RMSNorm, 2}

inputs!{RMSNorm, 
    0 => ("input", "Input tensor which layer normalization will be applied to"),
    1 => ("gamma", "scale tensor for elementwise_affine, the shape should be the same as the dimensions of X begin from axis"),
    2 => ("beta",  "bias tensor for elementwise_affine, the shape should be the same as the dimensions of X begin from axis")
}

outputs!{RMSNorm, 
    0 => ("output","Normalized values"),
    1 => ("rrms",  "Reciprocal of root mean square for each feature vector")
}

args!{RMSNorm, 
    0 => ("axis",    "(int) default to 1; Describes axis of the inputs. Defaults to one because the 0th axis most likely describes the batch size"),
    1 => ("epsilon", "(float) default to 0.001. Small value to be added to the stdev when dividing out by that value. This prevents division by zero.")
}

tensor_inference_function!{RMSNorm, /* [](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      std::vector<TensorShape> out(2);
      const auto input_dims_long = GetDimsVector(in[0]);
      const std::vector<int> input_dims(
          input_dims_long.cbegin(), input_dims_long.cend());
      out[0] = CreateTensorShape(input_dims, in[0].data_type());
      ArgumentHelper helper(def);
      const int axis = helper.GetSingleArgument<int32_t>("axis", 1);
      const int canonical_axis =
          canonical_axis_index_(axis, in[0].dims().size());
      const std::vector<int> rms_dims(
          input_dims.cbegin(), input_dims.cbegin() + canonical_axis);
      out[1] = CreateTensorShape(rms_dims, in[0].data_type());
      return out;
    } */
}

impl<Context> RMSNormOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "axis", axis_, 1),
            OP_SINGLE_ARG(float, "eps", eps_, 0.0f)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
        */
    }
}
