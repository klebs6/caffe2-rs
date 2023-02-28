crate::ix!();

use crate::{
    OperatorStorage,
    GradientMakerBase,
    OperatorDef,
};

/**
  | NegagteGradient operator in forward
  | pass simply copies input to the output,
  | and in backward pass, flips the sign
  | of the output gradient
  |
  */
pub struct NegateGradientOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
}

impl<Context> NegateGradientOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& in = Input(0);
        auto* out = Output(0);
        if (out != &in) {
          out->CopyFrom(in, /* async */ true);
        }
        return true;
        */
    }
}

register_cpu_operator!{NegateGradient,  NegateGradientOp<CPUContext>}

register_cuda_operator!{NegateGradient, NegateGradientOp<CUDAContext>}

num_inputs!{NegateGradient, 1}

num_outputs!{NegateGradient, 1}

allow_inplace!{NegateGradient, vec![(0, 0)]}

register_gradient!{NegateGradient, GetNegateGradientGradient}

pub struct GetNegateGradientGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetNegateGradientGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(def_.input_size(), 1);
        return SingleGradientDef(
            "Negative", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        */
    }
}
