crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/u8lut32norm.h]

#[macro_export] macro_rules! declare_pytorch_x8lut32norm_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(                 
              usize n, const u8* x, const u32* t, u8* y);
        */
    }
}

declare_pytorch_x8lut32norm_ukernel_function!{pytorch_u8lut32norm_ukernel__scalar}
