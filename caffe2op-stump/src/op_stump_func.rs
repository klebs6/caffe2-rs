crate::ix!();

/**
  | Converts each input element into either
  | high_ or low_value based on the given
  | threshold.
  | 
  | out[i] = low_value if in[i] <= threshold
  | else high_value
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct StumpFuncOp<TIN,TOUT,Context> {

    storage:     OperatorStorage,
    context:     Context,

    threshold:   TIN,
    low_value:   TOUT,
    high_value:  TOUT,

    /*
      | Input: label,
      | 
      | output: weight
      |
      */
}

register_cpu_operator!{StumpFunc, StumpFuncOp<float, float, CPUContext>}

num_inputs!{StumpFunc, 1}

num_outputs!{StumpFunc, 1}

inputs!{StumpFunc, 
    0 => ("X", "tensor of float")
}

outputs!{StumpFunc, 
    0 => ("Y", "tensor of float")
}

tensor_inference_function!{StumpFunc, /* ([](const OperatorDef&,
                                const vector<TensorShape>& input_types) {
      vector<TensorShape> out(1);
      out.at(0) = input_types.at(0);
      out.at(0).set_data_type(TensorProto_DataType::TensorProto_DataType_FLOAT);
      return out;
    }) */
}

no_gradient!{StumpFunc}

impl<TIN,TOUT,Context> StumpFuncOp<TIN,TOUT,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            threshold_(this->template GetSingleArgument<TIN>("threshold", 0)),
            low_value_(this->template GetSingleArgument<TOUT>("low_value", 0)),
            high_value_(this->template GetSingleArgument<TOUT>("high_value", 0))
        */
    }
}

impl StumpFuncOp<f32, f32, CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& in = Input(0);
      const float* in_data = in.template data<float>();

      auto* out = Output(0, in.sizes(), at::dtype<float>());
      float* out_data = out->template mutable_data<float>();
      for (int i = 0; i < in.numel(); i++) {
        out_data[i] = (in_data[i] <= threshold_) ? low_value_ : high_value_;
      }
      return true;
        */
    }
}

/**
  | Split the elements and return the indices
  | based on the given threshold.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct StumpFuncIndexOp<TIN,TOUT,Context> {

    storage: OperatorStorage,
    context: Context,

    threshold:  TIN,

    /**
      | Input: label
      | 
      | output: indices
      |
      */
    phantom: PhantomData<TOUT>,
}

register_cpu_operator!{
    StumpFuncIndex,
    StumpFuncIndexOp<f32, i64, CPUContext>
}

num_inputs!{StumpFuncIndex, 1}

num_outputs!{StumpFuncIndex, 2}

inputs!{StumpFuncIndex, 
    0 => ("X", "tensor of float")
}

outputs!{StumpFuncIndex, 
    0 => ("Index_Low",  "tensor of int64 indices for elements below/equal threshold"),
    1 => ("Index_High", "tensor of int64 indices for elements above threshold")
}

no_gradient!{StumpFuncIndex}

impl<TIN,TOUT,Context> StumpFuncIndexOp<TIN,TOUT,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            threshold_(this->template GetSingleArgument<TIN>("threshold", 0))
        */
    }
}

impl StumpFuncIndexOp<f32, i64, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& in = Input(0);
      const float* in_data = in.template data<float>();

      int lo_cnt = 0;
      for (int i = 0; i < in.numel(); i++) {
        lo_cnt += (in_data[i] <= threshold_);
      }
      auto* out_lo = Output(0, {lo_cnt}, at::dtype<int64_t>());
      auto* out_hi = Output(1, {in.numel() - lo_cnt}, at::dtype<int64_t>());
      int64_t* lo_data = out_lo->template mutable_data<int64_t>();
      int64_t* hi_data = out_hi->template mutable_data<int64_t>();
      int lidx = 0;
      int hidx = 0;
      for (int i = 0; i < in.numel(); i++) {
        if (in_data[i] <= threshold_) {
          lo_data[lidx++] = i;
        } else {
          hi_data[hidx++] = i;
        }
      }
      return true;
        */
    }
}

