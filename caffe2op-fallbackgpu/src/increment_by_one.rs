crate::ix!();

pub struct IncrementByOneOp {
    storage: OperatorStorage,
    context: CPUContext,
}

impl IncrementByOneOp {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& in = Input(0);

        auto* out = Output(0, in.sizes(), at::dtype<float>());
        const float* in_data = in.template data<float>();
        float* out_data = out->template mutable_data<float>();
        for (int i = 0; i < in.numel(); ++i) {
          out_data[i] = in_data[i] + 1.f;
        }
        return true;
        */
    }
}

num_inputs!{IncrementByOne, 1}

num_outputs!{IncrementByOne, 1}

allow_inplace!{IncrementByOne, vec![(0, 0)]}

register_cpu_operator!{IncrementByOne, IncrementByOneOp}

register_cuda_operator!{IncrementByOne, GPUFallbackOp}
