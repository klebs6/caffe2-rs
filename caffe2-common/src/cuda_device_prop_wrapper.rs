crate::ix!();

pub struct CudaDevicePropWrapper {
    props: Vec<CudaDeviceProp>,
}

impl Default for CudaDevicePropWrapper {
    
    fn default() -> Self {
        todo!();
        /*
            : props(NumCudaDevices()) 
                  for (int i = 0; i < NumCudaDevices(); ++i) {
                      CUDA_ENFORCE(cudaGetDeviceProperties(&props[i], i));
                  
        */
    }
}
