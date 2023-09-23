crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/x8zip.h]

#[macro_export] macro_rules! declare_pytorch_xzipc_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(usize n, const void* x, void* y);
        */
    }
}

declare_pytorch_xzipc_ukernel_function!{pytorch_qnnp_x8zip_x2__neon}
declare_pytorch_xzipc_ukernel_function!{pytorch_qnnp_x8zip_x2__sse2}
declare_pytorch_xzipc_ukernel_function!{pytorch_qnnp_x8zip_x3__neon}
declare_pytorch_xzipc_ukernel_function!{pytorch_qnnp_x8zip_x3__sse2}
declare_pytorch_xzipc_ukernel_function!{pytorch_qnnp_x8zip_x4__neon}
declare_pytorch_xzipc_ukernel_function!{pytorch_qnnp_x8zip_x4__sse2}

#[macro_export] macro_rules! declare_pytorch_xzipv_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(           
              usize n, usize m, const void* x, void* y);
        */
    }
}

declare_pytorch_xzipv_ukernel_function!{pytorch_qnnp_x8zip_xm__neon}
declare_pytorch_xzipv_ukernel_function!{pytorch_qnnp_x8zip_xm__sse2}
