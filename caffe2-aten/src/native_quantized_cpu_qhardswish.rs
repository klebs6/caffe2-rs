crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qhardswish.cpp]

define_dispatch!{qhardswish_stub}

#[cfg(USE_PYTORCH_QNNPACK)]
pub fn qnnpack_hardswish(
        qx: &Tensor,
        qy: &mut Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(qx.ndimension() > 0, "qnnpack_hardswish(): Got empty input tensor");
      initQNNPACK();

      usize num_elems = qx.numel() / qx.size(0);
      const auto i_zero_point = qx.q_zero_point();
      const auto i_scale = qx.q_scale();
      const auto o_zero_point = qy.q_zero_point();
      const auto o_scale = qy.q_scale();

      pytorch_qnnp_operator_t hardswish_op{nullptr};
      const pytorch_qnnp_status createStatus = pytorch_qnnp_create_hardswish_nc_q8(
        num_elems, // channels
        i_zero_point,
        i_scale,
        o_zero_point,
        o_scale,
        u8::min, // output min
        u8::max, // output max
        0, // flags
        &hardswish_op);

      unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
          qnnpack_uniq_ptr(hardswish_op);

      TORCH_INTERNAL_ASSERT(createStatus == pytorch_qnnp_status_success,
                            "failed to create QNNPACK Hardswish operator");

      const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_hardswish_nc_q8(
        hardswish_op,
        qx.size(0), // batch size
        (u8*)qx.data_ptr<quint8>(), // input data
        num_elems, // input stride
        (u8*)qy.data_ptr<quint8>(), // output data
        num_elems); // output stride
      TORCH_INTERNAL_ASSERT(setupStatus == pytorch_qnnp_status_success,
                            "failed to setup QNNPACK Hardswish operator");

      pthreadpool_t threadpool = pthreadpool_();

      const pytorch_qnnp_status runStatus =
        pytorch_qnnp_run_operator(hardswish_op, threadpool);

      TORCH_INTERNAL_ASSERT(
        runStatus == pytorch_qnnp_status_success,
        "failed to run QNNPACK Hardswish operator");
      return qy;
        */
}

pub fn quantized_hardswish(
        qx:                &Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
    
    todo!();
        /*
            Tensor qy = _empty_affine_quantized(
          qx.sizes(),
          device(kCPU).dtype(qx.scalar_type()),
          output_scale,
          output_zero_point,
          qx.suggest_memory_format());
    #ifdef USE_PYTORCH_QNNPACK
      if (globalContext().qEngine() == QEngine::QNNPACK &&
          qx.scalar_type() == kQUInt8) {
        Tensor qx_contig = qx.contiguous(qx.suggest_memory_format());
        qnnpack_hardswish(qx_contig, qy);
        return qy;
      }
    #endif  // USE_PYTORCH_QNNPACK
      qhardswish_stub(qx.device().type(), qx, qy);
      return qy;
        */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("quantized::hardswish"), TORCH_FN(quantized_hardswish));
    }
    */
}
