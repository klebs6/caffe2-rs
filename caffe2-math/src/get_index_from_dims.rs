crate::ix!();

/// Get index value from dims and index digits.
#[macro_export] macro_rules! caffe2_specialized_get_index_from_dims {
    ($TIndex:ty) => {
        /*
        template <>                                                 
            C10_EXPORT TIndex GetIndexFromDims(                         
                const int n, const TIndex* dims, const TIndex* index) { 
                TIndex sum = 0;                                           
                for (int i = 0; i < n; ++i) {                             
                    if (dims[i] > 1) {                                      
                        sum = sum * dims[i] + index[i];                       
                    }                                                       
                }                                                         
                return sum;                                               
            }
        */
    }
}

caffe2_specialized_get_index_from_dims!{i32}
caffe2_specialized_get_index_from_dims!{i64}
