crate::ix!();

#[macro_export] macro_rules! cuda_1d_kernel_loop {
    ($i:ident, $n:ident) => {
        todo!();
        /*
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); 
            i += blockDim.x * gridDim.x)
        */
    }
}

#[macro_export] macro_rules! cuda_2d_kernel_loop {
    ($i:ident, $n:ident, $j:ident, $m:ident) => {
        todo!();
        /*
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
        for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); j += blockDim.y * gridDim.y)
        */
    }
}
