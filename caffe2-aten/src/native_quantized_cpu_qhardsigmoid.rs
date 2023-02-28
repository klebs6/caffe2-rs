crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qhardsigmoid.cpp]

define_dispatch!{qhardsigmoid_stub}

#[cfg(USE_PYTORCH_QNNPACK)]
pub fn qnnpack_hardsigmoid(input: Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(input.ndimension() > 0, "qnnpack_hardsigmoid(): Got empty input tensor");
      initQNNPACK();

      Tensor input_contig = input.contiguous(input.suggest_memory_format());
      usize num_elems = input_contig.numel() / input_contig.size(0);
      const auto i_zero_point = input_contig.q_zero_point();
      const auto i_scale = input_contig.q_scale();
      constexpr float o_scale = 1.0f / 256.0f;
      constexpr i32 o_zero_point = 0;

      pytorch_qnnp_operator_t hardsigmoid_op{nullptr};
      const pytorch_qnnp_status createStatus = pytorch_qnnp_create_hardsigmoid_nc_q8(
        num_elems, // channels
        i_zero_point,
        i_scale,
        o_zero_point,
        o_scale,
        u8::min, // output min
        u8::max, // output max
        0, // flags
        &hardsigmoid_op);

      unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
          qnnpack_uniq_ptr(hardsigmoid_op);

      TORCH_INTERNAL_ASSERT(createStatus == pytorch_qnnp_status_success,
                            "failed to create QNNPACK Hardsigmoid operator");
      Tensor qy = _empty_affine_quantized(
        input_contig.sizes(),
        device(kCPU).dtype(input_contig.dtype()),
        o_scale,
        o_zero_point,
        input_contig.suggest_memory_format());

      const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_hardsigmoid_nc_q8(
        hardsigmoid_op,
        input_contig.size(0), // batch size
        (u8*)input_contig.data_ptr<quint8>(), // input data
        num_elems, // input stride
        (u8*)qy.data_ptr<quint8>(), // output data
        num_elems); // output stride
      TORCH_INTERNAL_ASSERT(setupStatus == pytorch_qnnp_status_success,
                            "failed to setup QNNPACK Hardsigmoid operator");

      pthreadpool_t threadpool = pthreadpool_();

      const pytorch_qnnp_status runStatus =
        pytorch_qnnp_run_operator(hardsigmoid_op, threadpool);

      TORCH_INTERNAL_ASSERT(
        runStatus == pytorch_qnnp_status_success,
        "failed to run QNNPACK Hardsigmoid operator");
      return qy;
        */
}

pub fn hardsigmoid_quantized_cpu(qx: &Tensor) -> Tensor {
    
    todo!();
        /*
            #ifdef USE_PYTORCH_QNNPACK
      if (globalContext().qEngine() == QEngine::QNNPACK &&
          qx.scalar_type() == kQUInt8) {
        return qnnpack_hardsigmoid(qx);
      }
    #endif  // USE_PYTORCH_QNNPACK
      Tensor qy;
      qhardsigmoid_stub(qx.device().type(), qx, qy);
      return qy;
        */
}
