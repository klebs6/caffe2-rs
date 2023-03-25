crate::ix!();

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

    #[inline] pub fn run_on_device_f32_in_i64_out_cpu(&mut self) -> bool {
        
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
