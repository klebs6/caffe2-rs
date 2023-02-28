// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qtanh.cpp]

define_dispatch!{qtanh_stub}

/**
  | This ALWAYS outputs scale=2.0/256,
  | zp=128, dtype=quint8
  |
  */
#[cfg(USE_PYTORCH_QNNPACK)]
pub fn qnnpack_tanh(input: Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(input.ndimension() > 0, "qnnpack_tanh(): Got empty input tensor");

      Tensor qy;
      constexpr float output_scale = 2.0f / 256.0f;
      constexpr i32 output_zero_point = 128;

      initQNNPACK();

      Tensor input_contig = input.contiguous(input.suggest_memory_format());
      usize num_elems = 1;
      for (int i = 1; i < input_contig.ndimension(); ++i) {
        num_elems *= input_contig.size(i);
      }
      const auto zero_point = input_contig.q_zero_point();
      const auto scale = input_contig.q_scale();

      pytorch_qnnp_operator_t tanh_op{nullptr};
      const pytorch_qnnp_status createStatus = pytorch_qnnp_create_tanh_nc_q8(
        num_elems /* channels */,
        zero_point /* input zero point */,
        scale /* input scale */,
        output_zero_point /* output zero point */,
        output_scale /* output scale */,
        u8::min /* output min */,
        u8::max /* output max */,
        0 /* flags */,
        &tanh_op);

      unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
          qnnpack_uniq_ptr(tanh_op);

      TORCH_INTERNAL_ASSERT(createStatus == pytorch_qnnp_status_success,
                            "failed to create QNNPACK TanH operator");
      qy = _empty_affine_quantized(
        input_contig.sizes(),
        device(kCPU).dtype(input_contig.dtype()),
        output_scale,
        output_zero_point,
        input_contig.suggest_memory_format());

      const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_tanh_nc_q8(
        tanh_op,
        input_contig.size(0) /* batch size */,
        (u8*)input_contig.data_ptr<quint8>() /* input data */,
        num_elems /* input stride */,
        (u8*)qy.data_ptr<quint8>() /* output data */,
        num_elems /* output stride */);
      TORCH_INTERNAL_ASSERT(setupStatus == pytorch_qnnp_status_success,
                            "failed to setup QNNPACK TanH operator");

      pthreadpool_t threadpool = pthreadpool_();

      const pytorch_qnnp_status runStatus =
        pytorch_qnnp_run_operator(tanh_op, threadpool);

      TORCH_INTERNAL_ASSERT(
        runStatus == pytorch_qnnp_status_success,
        "failed to run QNNPACK TanH operator");
      return qy;
        */
}

pub fn tanh_quantized_cpu(qx: &Tensor) -> Tensor {
    
    todo!();
        /*
            #ifdef USE_PYTORCH_QNNPACK
      if (globalContext().qEngine() == QEngine::QNNPACK &&
          qx.scalar_type() == kQUInt8) {
        return qnnpack_tanh(qx);
      }
    #endif  // USE_PYTORCH_QNNPACK
      Tensor qy;
      qtanh_stub(qx.device().type(), qx, qy);
      return qy;
        */
}
