crate::ix!();

/**
  | Put forward and backward in the same
  | template?
  |
  */
pub struct SumRangeReducer<T, CPUContext> {
    phantom:           PhantomData<T>,
    phantomCPUContext: PhantomData<CPUContext>,
}

impl<T, CPUContext> SumRangeReducer<T, CPUContext> {
    
    #[inline] pub fn invoke(&mut self, 
        block_size: i64,
        blocks:     i64,
        input:      *const T,
        out:        *mut T,
        context:    *mut CPUContext)  {

        todo!();
        /*
            // do we need to go through wrapper in math.h?
        EigenVectorMap<T> out_vec(out, block_size);
        out_vec = ConstEigenMatrixMap<T>(in, block_size, blocks).rowwise().sum();
        */
    }
}
