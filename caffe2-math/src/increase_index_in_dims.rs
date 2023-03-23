crate::ix!();

/**
  | Increase the index digits by one based
  | on dims.
  |
  */
#[macro_export] macro_rules! caffe2_specialized_increase_index_in_dims {
    ($TIndex:ty) => {
        /*
        template <>                                              
            C10_EXPORT void IncreaseIndexInDims<TIndex>(             
                const int ndim, const TIndex* dims, TIndex* index) { 
                for (int i = ndim - 1; i >= 0; --i) {                  
                    ++index[i];                                          
                    if (index[i] >= dims[i]) {                           
                        index[i] -= dims[i];                               
                    } else {                                             
                        break;                                             
                    }                                                    
                }                                                      
            }
        */
    }
}

caffe2_specialized_increase_index_in_dims!{i32}
caffe2_specialized_increase_index_in_dims!{i64}
