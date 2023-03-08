crate::ix!();

/**
  | ZeroGradient operators doesn't produce
  | any output blobs.
  | 
  | One can use this operator to produce
  | 0 gradient for the input blob.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ZeroGradientOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

impl<Context> ZeroGradientOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        true
    }
}

register_cpu_operator!{ZeroGradient, ZeroGradientOp<CPUContext>}

num_inputs!{ZeroGradient, 1}

num_outputs!{ZeroGradient, 0}

register_gradient!{ZeroGradient, GetZeroGradientOpGradient}

register_cuda_operator!{ZeroGradient, ZeroGradientOp<CUDAContext>}

pub struct GetZeroGradientOpGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetZeroGradientOpGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
        return SingleGradientDef(
            "ConstantFill",
            "",
            vector<string>{I(0)},
            vector<string>{GI(0)},
            vector<Argument>{MakeArgument<float>("value", 0.0)});
        */
    }
}
