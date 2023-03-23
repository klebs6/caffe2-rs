crate::ix!();

/// Computest the broadcast binary operation dims.
#[macro_export] macro_rules! caffe2_specialized_compute_broadcast_binary_op_dims {
    ($TIndex:ty) => {
        /*
        template <>                                                             
            C10_EXPORT void ComputeBroadcastBinaryOpDims(                           
                const int A_ndim,                                                   
                const TIndex* A_dims,                                               
                const int B_ndim,                                                   
                const TIndex* B_dims,                                               
                TIndex* A_broadcast_dims,                                           
                TIndex* B_broadcast_dims,                                           
                TIndex* C_broadcast_dims) {                                         
                const int ndim = std::max(A_ndim, B_ndim);                            
                std::fill(A_broadcast_dims, A_broadcast_dims + ndim - A_ndim, 1);     
                std::fill(B_broadcast_dims, B_broadcast_dims + ndim - B_ndim, 1);     
                std::copy(A_dims, A_dims + A_ndim, A_broadcast_dims + ndim - A_ndim); 
                std::copy(B_dims, B_dims + B_ndim, B_broadcast_dims + ndim - B_ndim); 
                for (int i = 0; i < ndim; ++i) {                                      
                    CAFFE_ENFORCE(                                                      
                        A_broadcast_dims[i] == B_broadcast_dims[i] ||                   
                        A_broadcast_dims[i] <= 1 || B_broadcast_dims[i] <= 1);          
                    if (A_broadcast_dims[i] == 0 || B_broadcast_dims[i] == 0) {         
                        C_broadcast_dims[i] = 0;                                          
                    } else {                                                            
                        C_broadcast_dims[i] =                                             
                            std::max(A_broadcast_dims[i], B_broadcast_dims[i]);           
                    }                                                                   
                }                                                                     
            }
        */
    }
}

caffe2_specialized_compute_broadcast_binary_op_dims!{i32}
caffe2_specialized_compute_broadcast_binary_op_dims!{i64}
