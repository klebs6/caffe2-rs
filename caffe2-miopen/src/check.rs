crate::ix!();

#[macro_export] macro_rules! miopen_check {
    ($condition:ident) => {
        todo!();
        /*
        
            do                                                                                            
            {                                                                                             
                miopenStatus_t status = condition;                                                        
                CHECK(status == miopenStatusSuccess) << ::caffe2::internal::miopenGetErrorString(status); 
            } while(0)
        */
    }
}
