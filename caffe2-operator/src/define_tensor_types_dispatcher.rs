crate::ix!();

define_tensor_types_dispatcher!{
    /*
    TensorTypes,
    DoRunWithType,
    DoRunWithOtherType
    */
}

define_tensor_types_dispatcher!{
    /*
    TensorTypes2,
    DoRunWithType2,
    DoRunWithOtherType2
    */
}

#[macro_export] macro_rules! define_tensor_types_dispatcher {
    () => {
        /*
                (                                    
            TensorTypes, DoRunWithType, DoRunWithOtherType)                            
          template <typename FirstType, typename... Types, typename... ExtraArgs>      
          struct DispatchHelper<TensorTypes<FirstType, Types...>, ExtraArgs...> {      
            template <typename Op>                                                     
            static bool call(Op* op, const TypeMeta meta) {                           
              static_assert(                                                           
                  !std::is_same<GenericTensorImplementation, FirstType>::value,        
                  "GenericTensorImplementation must be the last in TensorTypes list"); 
              if (meta.Match<FirstType>()) {                                           
                return op->template DoRunWithType<ExtraArgs..., FirstType>();          
              }                                                                        
              return DispatchHelper<TensorTypes<Types...>, ExtraArgs...>::             
                  template call<Op>(op, meta);                                         
            }                                                                          
            template <typename Op>                                                     
            static bool call(Op* op, const Tensor& tensor) {                           
              return call<Op>(op, tensor.dtype());                                     
            }                                                                          
            template <typename Op>                                                     
            static bool call(Op* op, const Blob& blob) {                               
              return call<Op>(op, blob.meta());                                        
            }                                                                          
          };                                                                           
                                                                                       
          template <typename... ExtraArgs>                                             
          struct DispatchHelper<TensorTypes<>, ExtraArgs...> {                         
            template <typename Op>                                                     
            static bool call(Op* /* unused */, const TypeMeta meta) {                 
              CAFFE_THROW("Unsupported type of tensor: ", meta.name());                
            }                                                                          
            template <typename Op>                                                     
            static bool call(Op* op, const Tensor& tensor) {                           
              return call<Op>(op, tensor.dtype());                                     
            }                                                                          
            template <typename Op>                                                     
            static bool call(Op* op, const Blob& blob) {                               
              return call<Op>(op, blob.meta());                                        
            }                                                                          
          };                                                                           
                                                                                       
          template <typename... ExtraArgs>                                             
          struct DispatchHelper<                                                       
              TensorTypes<GenericTensorImplementation>,                                
              ExtraArgs...> {                                                          
            template <typename Op>                                                     
            static bool call(Op* op, const TypeMeta) {                                
              return op->template DoRunWithOtherType<ExtraArgs...>();                  
            }                                                                          
            template <typename Op>                                                     
            static bool call(Op* op, const Tensor& tensor) {                           
              return call<Op>(op, tensor.dtype());                                     
            }                                                                          
            template <typename Op>                                                     
            static bool call(Op* op, const Blob& blob) {                               
              return call<Op>(op, blob.meta());                                        
            }                                                                          
          };
        */
    }
}
