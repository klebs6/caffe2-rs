crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/u8rmax.h]

#[macro_export] macro_rules! declare_pytorch_u8rmax_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL u8 fn_name(usize n, const u8* x);
        */
    }
}


declare_pytorch_u8rmax_ukernel_function!{pytorch_u8rmax_ukernel__neon}
declare_pytorch_u8rmax_ukernel_function!{pytorch_u8rmax_ukernel__sse2}

