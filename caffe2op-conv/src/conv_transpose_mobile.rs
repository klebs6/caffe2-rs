crate::ix!();

#[USE_CONV_TRANSPOSE_UNPOOL_BASE_FUNCTIONS("Context")]
pub struct ConvTransposeMobileOp<T, Context> {
    phantom: PhantomData<T>,


    base: ConvTransposeUnpoolBase<Context>,

    /**
      | We store a numThreasds per-worker  tiles
      | of Y, and numThreads per-worker
      | threadBuffer for the gemm output, laid out
      | in that order.
      */
    thread_buffer: Tensor, // {CPU};

    // Input: X, W, b
    // Output: Y
}

input_tags!{
    ConvTransposeMobileOp {
        Input,
        Filter,
        Bias
    }
}

impl<T,Context> ConvTransposeMobileOp<T,Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : ConvTransposeUnpoolBase<Context>(operator_def, ws) 

        OPERATOR_NEEDS_FEATURE(
            order_ == StorageOrder::NCHW,
            "Only NCHW order is supported right now.");
        OPERATOR_NEEDS_FEATURE(
            this->pad_l() == 0, "operator does not handle row width padding");
        OPERATOR_NEEDS_FEATURE(
            this->pad_r() == 0, "operator does not handle row width padding");
        OPERATOR_NEEDS_FEATURE(this->stride_w() <= 4, "stride width must be <= 4");
        */
    }
}

/**
  | mobile-only implementation (tiled
  | + vectorized + multithreaded)
  |
  */
#[cfg(c10_mobile)]
register_cpu_operator_with_engine!{
    ConvTranspose,
    MOBILE,
    ConvTransposeMobileOp<f32, CPUContext>
}
