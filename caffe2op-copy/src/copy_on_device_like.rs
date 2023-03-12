crate::ix!();

/**
  | Copy input tensor into output to the
  | specific device
  |
  */
pub struct CopyOnDeviceLikeOp<Context, DstContext, SrcContext> {
    base:     CopyOp<Context, DstContext, SrcContext>,
    phantomA: PhantomData<SrcContext>,
    phantomB: PhantomData<DstContext>,
}

num_inputs!{CopyOnDeviceLike, 2}

num_outputs!{CopyOnDeviceLike, 1}

inputs!{CopyOnDeviceLike, 
    0 => ("input", "The input tensor."),
    1 => ("dst", "Tensor, on which device the copy will be performed.")
}

outputs!{CopyOnDeviceLike, 
    0 => ("output", "Tensor that will contain a copy of the input.")
}

register_cpu_operator!{
    CopyOnDeviceLike,
    CopyOnDeviceLikeOp<CPUContext, CPUContext, CPUContext>
}

impl<Context,DstContext,SrcContext> CopyOnDeviceLikeOp<Context,DstContext,SrcContext> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : CopyOp<Context, DstContext, SrcContext>(std::forward<Args>(args)...)
        */
    }
}
