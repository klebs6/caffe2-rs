crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/init_qnnpack.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/init_qnnpack.cpp]

#[cfg(USE_PYTORCH_QNNPACK)]
pub fn initqnnpack()  {
    
    todo!();
        /*
            static once_flag once;
      static enum pytorch_qnnp_status qnnpackStatus =
          pytorch_qnnp_status_uninitialized;
      call_once(once, []() { qnnpackStatus = pytorch_qnnp_initialize(); });
      TORCH_CHECK(
          qnnpackStatus == pytorch_qnnp_status_success,
          "failed to initialize QNNPACK");
        */
}
