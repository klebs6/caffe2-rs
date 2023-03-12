crate::ix!();

#[macro_export] macro_rules! register_cpu_compare_operator {
    ($Op:ident) => {
        /*
        REGISTER_CPU_OPERATOR(                                      
            Op,                                                     
            BinaryElementwiseOp<                                    
            TensorTypes<bool, int32_t, int64_t, float, double>, 
            CPUContext,                                         
            Op##Functor<CPUContext>,                            
            FixedType<bool>>)
        */
    }
}

register_cpu_compare_operator!{EQ}
register_cpu_compare_operator!{NE}
register_cpu_compare_operator!{LT}
register_cpu_compare_operator!{LE}
register_cpu_compare_operator!{GT}
register_cpu_compare_operator!{GE}

///----------------------------------------
#[macro_export] macro_rules! register_cpu_logical_binary_operator {
    ($Op:ident) => {
        /*
        REGISTER_CPU_OPERATOR(                         \
            Op, BinaryElementwiseOp<BoolTypes, CPUContext, Op##Functor<CPUContext>>)
        */
    }
}

register_cpu_logical_binary_operator!{And}
register_cpu_logical_binary_operator!{Or}
register_cpu_logical_binary_operator!{Xor}

///---------------------------------------------
#[macro_export] macro_rules! register_cpu_bitwise_binary_operator {
    ($Op:ident) => {
        /*
        REGISTER_CPU_OPERATOR(                         \
            Op,                                        \
            BinaryElementwiseOp<IntBoolTypes, CPUContext, Op##Functor<CPUContext>>)
        */
    }
}

register_cpu_bitwise_binary_operator!{BitwiseAnd}
register_cpu_bitwise_binary_operator!{BitwiseOr}
register_cpu_bitwise_binary_operator!{BitwiseXor}
