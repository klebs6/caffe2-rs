crate::ix!();

// TODO(jiayq): Add example to the doc string.

/**
  | StopGradient is a helper operator that
  | does no actual numerical computation,
  | and in the gradient computation phase
  | stops the gradient from being computed
  | through it.
  |
  */
pub struct StopGradientOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
}

impl<Context> StopGradientOp<Context> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& in = Input(0);
        auto* out = Output(0);
        if (out != &in) {
          out->CopyFrom(in, true /*async*/);
        }
        return true;
        */
    }
}

register_cpu_operator!{StopGradient, StopGradientOp<CPUContext>}

num_inputs!{StopGradient, (1,1)}

num_outputs!{StopGradient, (1,1)}

identical_type_and_shape!{StopGradient}

allow_inplace!{StopGradient, vec![(0, 0)]}

no_gradient!{StopGradient}

register_cuda_operator!{StopGradient, StopGradientOp<CUDAContext>}
