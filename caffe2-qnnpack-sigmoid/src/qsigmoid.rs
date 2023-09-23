// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qsigmoid.cpp]

define_dispatch!{qsigmoid_stub}

#[cfg(USE_PYTORCH_QNNPACK)]
pub fn qnnpack_sigmoid(
    input:             Tensor,
    output_scale:      f64,
    output_zero_point: i64) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(input.ndimension() > 0, "qnnpack_sigmoid(): Got empty input tensor");

      Tensor qy;
      initQNNPACK();

      Tensor input_contig = input.contiguous(input.suggest_memory_format());
      usize num_elems = 1;
      for (int i = 1; i < input_contig.ndimension(); ++i) {
        num_elems *= input_contig.size(i);
      }

      const auto zero_point = input_contig.q_zero_point();
      const auto scale = input_contig.q_scale();

      pytorch_qnnp_operator_t sigmoid_op{nullptr};
      const pytorch_qnnp_status createStatus = pytorch_qnnp_create_sigmoid_nc_q8(
        num_elems /* channels */,
        zero_point /* input zero point */,
        scale /* input scale */,
        output_zero_point /* output zero point */,
        output_scale /* output scale */,
        u8::min /* output min */,
        u8::max /* output max */,
        0 /* flags */,
        &sigmoid_op);

      unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
          qnnpack_uniq_ptr(sigmoid_op);

      TORCH_INTERNAL_ASSERT(createStatus == pytorch_qnnp_status_success,
                            "failed to create QNNPACK sigmoid operator");
      qy = _empty_affine_quantized(
        input_contig.sizes(),
        device(kCPU).dtype(input_contig.dtype()),
        output_scale,
        output_zero_point,
        input_contig.suggest_memory_format());

      const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_sigmoid_nc_q8(
        sigmoid_op,
        input_contig.size(0) /* batch size */,
        (u8*)input_contig.data_ptr<quint8>() /* input data */,
        num_elems /* input stride */,
        (u8*)qy.data_ptr<quint8>() /* output data */,
        num_elems /* output stride */);
      TORCH_INTERNAL_ASSERT(setupStatus == pytorch_qnnp_status_success,
                            "failed to setup QNNPACK sigmoid operator");

      pthreadpool_t threadpool = pthreadpool_();

      const pytorch_qnnp_status runStatus =
        pytorch_qnnp_run_operator(sigmoid_op, threadpool);

      TORCH_INTERNAL_ASSERT(
        runStatus == pytorch_qnnp_status_success,
        "failed to run QNNPACK sigmoid operator");
      return qy;
        */
}

/**
  | This ALWAYS outputs scale=1.0/256, dtype=quint8
  |
  | The zero_point is 0 for qint32 and quint8, but
  | -128 for qint8.
  |
  */
pub fn sigmoid_quantized_cpu(qx: &Tensor) -> Tensor {
    
    todo!();
        /*
            #ifdef USE_PYTORCH_QNNPACK
      if (globalContext().qEngine() == QEngine::QNNPACK &&
          qx.scalar_type() == kQUInt8) {
        constexpr double output_scale = 1.0f / 256.0f;
        constexpr i64 output_zero_point = 0;
        return qnnpack_sigmoid(qx, output_scale, output_zero_point);
      }
    #endif  // USE_PYTORCH_QNNPACK
      Tensor qy;
      AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qsigmoid", [&]() {
        // Naive implemenentation: uses dequantize/execute/quantize routine
        // - Output scale is set to 1.0 / 2^(BIT_NUM)
        // - For signed types output zero point is set to 0
        // - For unsigned types output zero point is set to (qmax + qmin) / 2.0
        // See https://stackoverflow.com/a/34448562/3606192 for potential
        // optimizations
        double output_scale = 0.00390625;  // 1.0 / 2^8
        i64 output_zero_point = 0;
        if (SCALAR_TYPE == kQInt32) {
          output_scale = 2.3283064365386963e-10;  // 1.0 / 2^32
        } else if (SCALAR_TYPE == kQInt8) {
          output_zero_point = -128;
        }
        qsigmoid_stub(qx.device().type(), qx, qy, output_scale, output_zero_point);
      });
      return qy;
        */
}

pub struct QSigmoid {

}

impl QSigmoid {
    
    pub fn run(
        qx:                Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            #ifdef USE_PYTORCH_QNNPACK
      if (globalContext().qEngine() == QEngine::QNNPACK &&
          qx.scalar_type() == kQUInt8) {
        return qnnpack_sigmoid(qx, output_scale, output_zero_point);
      }
    #endif  // USE_PYTORCH_QNNPACK
      Tensor qy;
      qsigmoid_stub(qx.device().type(), qx, qy, output_scale, output_zero_point);
      return qy;
        */
    }
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("quantized::sigmoid"), TORCH_FN(QSigmoid::run));
    }
    */
}
