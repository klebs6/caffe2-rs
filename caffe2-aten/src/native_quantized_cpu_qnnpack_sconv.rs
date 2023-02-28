// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/sconv.h]

#[macro_export] macro_rules! declare_pytorch_sconv_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(           
              usize mr,                                
              usize nr,                                
              usize kc,                                
              usize ks,                                
              const float** a,                          
              const float* w,                           
              float* c,                                 
              usize c_stride,                          
              const struct pytorch_qnnp_fp32_clamping_params* params);
        */
    }
}

declare_pytorch_sconv_ukernel_function!{pytorch_sconv_ukernel_6x8__psimd}
