crate::ix!();

pub struct CopyRowsToTensorGradientOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

impl<Context> CopyRowsToTensorGradientOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<
            TensorTypes<at::Half, float, double, int32_t, int64_t>>::
            call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<T>() -> bool {
        todo!();
        /*
            auto* output = Output(0);
            output->ResizeLike(Input(0));
            auto* output_data = output->template mutable_data<T>();
            auto& input = Input(0);
            const auto* input_data = input.template data<T>();
            std::memcpy(output_data, input_data, input.size(0) * sizeof(T));

            return true;
        */
    }
}

register_cpu_operator!{
    CopyRowsToTensor, 
    CopyRowsToTensorOp<CPUContext>
}

register_cpu_gradient_operator!{
    CopyRowsToTensorGradient,
    CopyRowsToTensorGradientOp<CPUContext>
}

num_inputs!{CopyRowsToTensorGradient, 1}

num_outputs!{CopyRowsToTensorGradient, 1}

allow_inplace!{CopyRowsToTensorGradient, vec![(0, 0)]}
