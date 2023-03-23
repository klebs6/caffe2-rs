crate::ix!();

#[macro_export] 
macro_rules! caffe2_specialized_compute_transposed_strides {
    ($TIndex:ty) => {
        /*
        template <>                                                                 
            C10_EXPORT void ComputeTransposedStrides<TIndex>(                           
                const int ndim, const TIndex* dims, const int* axes, TIndex* strides) { 
                std::vector<TIndex> buff(ndim);                                           
                TIndex cur_stride = 1;                                                    
                for (int i = ndim - 1; i >= 0; --i) {                                     
                    buff[i] = cur_stride;                                                   
                    cur_stride *= dims[i];                                                  
                }                                                                         
                for (int i = 0; i < ndim; ++i) {                                          
                    strides[i] = buff[axes[i]];                                             
                }                                                                         
        }
        */
    }
}

caffe2_specialized_compute_transposed_strides!{i32}
caffe2_specialized_compute_transposed_strides!{i64}

