crate::ix!();

// Forward-only Binary Functors.
#[macro_export] macro_rules! declare_forward_only_binary_functor {
    ($FunctorName:ident) => {
        /*
        template <class Context>                                   
            struct FunctorName##Functor {                              
                template <typename TIn, typename TOut>                   
                    bool Forward(                                            
                        const std::vector<int>& A_dims,                      
                        const std::vector<int>& B_dims,                      
                        const TIn* A,                                        
                        const TIn* B,                                        
                        TOut* C,                                             
                        Context* context) const {                            
                        math::FunctorName(                                     
                            A_dims.size(),                                     
                            A_dims.data(),                                     
                            B_dims.size(),                                     
                            B_dims.data(),                                     
                            A,                                                 
                            B,                                                 
                            C,                                                 
                            context);                                          
                        return true;                                           
                    }                                                        
            };
        */
    }
}

// Compare functors.
declare_forward_only_binary_functor!{EQ}
declare_forward_only_binary_functor!{NE}
declare_forward_only_binary_functor!{LT}
declare_forward_only_binary_functor!{LE}
declare_forward_only_binary_functor!{GT}
declare_forward_only_binary_functor!{GE}

// Logical functors.
declare_forward_only_binary_functor!{And}
declare_forward_only_binary_functor!{Or}
declare_forward_only_binary_functor!{Xor}

// Bitwise functors.
declare_forward_only_binary_functor!{BitwiseAnd}
declare_forward_only_binary_functor!{BitwiseOr}
declare_forward_only_binary_functor!{BitwiseXor}

