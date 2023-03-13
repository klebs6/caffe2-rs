crate::ix!();

#[macro_export] 
macro_rules! reduction_op_shape_inference {
    ($is_front_reducer:ident) => {
        todo!();
        /*
        
          CAFFE_ENFORCE_LE(1, in.size());                                           
          CAFFE_ENFORCE_GE(2, in.size());                                           
          ArgumentHelper helper(def);                                               
          int num_reduce_dims = helper.GetSingleArgument<int>("num_reduce_dim", 1); 
          int start_index = is_front_reducer ? num_reduce_dims : 0;                 
          int end_index = is_front_reducer ? in[0].dims_size()                      
                                           : in[0].dims_size() - num_reduce_dims;   
          vector<int> output_shape;                                                 
          for (int i = start_index; i < end_index; ++i) {                           
            output_shape.push_back(in[0].dims(i));                                  
          }                                                                         
          return vector<TensorShape>{                                               
              CreateTensorShape(output_shape, in[0].data_type())};
        */
    }
}
