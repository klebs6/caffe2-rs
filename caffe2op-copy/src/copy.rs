crate::ix!();

/**
  | Copy input tensor into output, potentially
  | across devices.
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/copy_op.cc
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/copy_op.h
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct CopyOp<Context, DstContext, SrcContext> {
    storage:  OperatorStorage,
    context:  Context,
    phantomA: PhantomData<SrcContext>,
    phantomB: PhantomData<DstContext>,
}

num_inputs!{Copy, 1}

num_outputs!{Copy, 1}

inputs!{Copy, 
    0 => ("input", "(*Tensor*): input tensor to copy")
}

outputs!{Copy, 
    0 => ("output", "(*Tensor*): copy of input tensor")
}

identical_type_and_shape!{Copy}

inputs_can_cross_devices!{Copy}

inherit_onnx_schema!{"Identity"}

register_cpu_operator!{
    Copy, 
    CopyOp<CPUContext, CPUContext, CPUContext>
}

impl<Context, DstContext, SrcContext> CopyOp<Context, DstContext, SrcContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = this->template Input<Tensor>(0, SrcContext::GetDeviceType());
        auto* output =
            this->template Output<Tensor>(0, DstContext::GetDeviceType());
        output->ResizeLike(input);
        this->context_.template CopyItems<SrcContext, DstContext>(
            input.dtype(),
            input.numel(),
            input.raw_data(),
            output->raw_mutable_data(input.dtype()));
        return true;
        */
    }
}
