crate::ix!();

/**
 | MIOpenWorkspace is a wrapper around a raw cuda
 | pointer that holds the miopen scratch
 | space. This struct is meant to be only used in
 | MIOPENWrapper to provide a program-wide scratch
 | space for MIOPEN. The reason behind it is that
 | miopen function calls are usually very
 | efficient, hence one probably does not want to
 | run multiple miopen calls at the same time. As
 | a result, one should not need more than one
 | miopen workspace per device.
 */
pub struct MIOpenWorkspace
{
    data:    DataPtr,
    nbytes:  usize, // default = 0
}

impl MIOpenWorkspace {
    
    #[inline] pub fn reset(&mut self)  {
        
        todo!();
        /*
            data_.clear();
          nbytes_ = 0;
        */
    }
    
    #[inline] pub fn get(&mut self, nbytes: usize)  {
        
        todo!();
        /*
            if(nbytes_ < nbytes)
            {
                reset();
                data_ = HIPContext::New(nbytes);
                nbytes_               = nbytes;
            }
            CAFFE_ENFORCE_GE(nbytes_, nbytes);
            return data_.get();
        */
    }
}
