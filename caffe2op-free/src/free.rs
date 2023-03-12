crate::ix!();

/// FreeOp frees the content of the output
/// blob. We allow it to take in input blobs
/// purely for the reason that it can "wait"
/// on the input blobs to be produced by some
/// of the earlier operators before a free
/// is called.
/// 
/// Frees the content of the blobs. The input
/// and output blobs should be one-to-one
/// inplace.
/// 
pub struct FreeOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{Free, (1,INT_MAX)}

num_outputs!{Free, (1,INT_MAX)}

same_number_of_output!{Free}

enforce_one_to_one_inplace!{Free}

impl<Context> FreeOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            for (Blob* output : OperatorStorage::Outputs()) {
          output->Reset();
        }
        return true;
        */
    }
}

register_cpu_operator!{Free, FreeOp<CPUContext>}

should_not_do_gradient!{Free}

register_cuda_operator!{Free, FreeOp<CUDAContext>}
