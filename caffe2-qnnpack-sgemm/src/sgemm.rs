crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/sgemm.h]

#[macro_export] macro_rules! declare_pytorch_sgemm_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(           
              usize mr,                                
              usize nr,                                
              usize k,                                 
              const float* a,                           
              usize a_stride,                          
              const float* w,                           
              float* c,                                 
              usize c_stride,                          
              const struct pytorch_qnnp_fp32_clamping_params* clamping_params);
        */
    }
}

declare_pytorch_sgemm_ukernel_function!{pytorch_sgemm_ukernel_5x8__neon}
declare_pytorch_sgemm_ukernel_function!{pytorch_sgemm_ukernel_6x8__neon}
declare_pytorch_sgemm_ukernel_function!{pytorch_sgemm_ukernel_6x8__psimd}
