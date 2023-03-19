crate::ix!();

/**
  | CudnnWorkspace is a wrapper around
  | a raw cuda pointer that holds the cudnn
  | scratch space.
  | 
  | This struct is meant to be only used in
  | 
  | CudnnWrapper to provide a program-wide
  | scratch space for Cudnn.
  | 
  | The reason behind it is that cudnn function
  | calls are usually very efficient, hence
  | one probably does not want to run multiple
  | cudnn calls at the same time.
  | 
  | As a result, one should not need more
  | than one cudnn workspace per device.
  |
  */
pub struct CudnnWorkspace {

    /// {nullptr, nullptr, &NoDelete, at::Device(CUDA)};
    data: DataPtr,

    /// {0};
    nbytes: usize,
}

impl CudnnWorkspace {
    
    #[inline] pub fn get(&mut self, nbytes: usize)  {
        
        todo!();
        /*
            if (nbytes_ < nbytes) {
                reset();
                data_ = CUDAContext::New(nbytes);
                nbytes_ = nbytes;
            }
            CAFFE_ENFORCE_GE(nbytes_, nbytes);
            return data_.get();
        */
    }
    
    #[inline] pub fn reset(&mut self)  {
        
        todo!();
        /*
            data_.clear();
            nbytes_ = 0;
        */
    }
}
