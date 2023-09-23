crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/x8lut.h]

#[macro_export] macro_rules! declare_pytorch_x8lut_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(           
              usize n, const u8* x, const u8* t, u8* y);
        */
    }
}

declare_pytorch_x8lut_ukernel_function!{pytorch_x8lut_ukernel__scalar}
