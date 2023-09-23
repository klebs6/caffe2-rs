crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qclamp.cpp]

define_dispatch!{qclamp_stub}
define_dispatch!{qclamp_min_stub}
define_dispatch!{qclamp_max_stub}

#[cfg(USE_PYTORCH_QNNPACK)]
pub fn qnnpack_clamp(
        input: Tensor,
        min:   &Scalar,
        max:   &Scalar) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(input.ndimension() > 0, "qnnpack_clamp(): Got empty input tensor");

      initQNNPACK();

      Tensor input_contig = input.contiguous(input.suggest_memory_format());
      usize num_elems = 1;
      for (int i = 1; i < input_contig.ndimension(); ++i) {
        num_elems *= input_contig.size(i);
      }

      auto min_f = min.to<float>();
      auto max_f = max.to<float>();
      u8 min_q =
          native::quantize_val<quint8>(input.q_scale(), input.q_zero_point(), min_f).val_;
      u8 max_q =
          native::quantize_val<quint8>(input.q_scale(), input.q_zero_point(), max_f).val_;

      pytorch_qnnp_operator_t clamp_op{nullptr};
      const pytorch_qnnp_status createStatus = pytorch_qnnp_create_clamp_nc_u8(
        num_elems, // channels
        min_q,
        max_q,
        0, // flags
        &clamp_op);

      unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
          qnnpack_uniq_ptr(clamp_op);

      TORCH_INTERNAL_ASSERT(createStatus == pytorch_qnnp_status_success,
                            "failed to create QNNPACK Clamp operator");

      Tensor qy = _empty_affine_quantized(
        input_contig.sizes(),
        input_contig.options(),
        input_contig.q_scale(),
        input_contig.q_zero_point());

      const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_clamp_nc_u8(
        clamp_op,
        input_contig.size(0), // batch_size
        (u8*)input_contig.data_ptr<quint8>(), // input_data
        num_elems, // input_stride
        (u8*)qy.data_ptr<quint8>(), // output_data
        num_elems); // output_stride
      TORCH_INTERNAL_ASSERT(setupStatus == pytorch_qnnp_status_success,
                            "failed to setup QNNPACK Clamp operator");

      pthreadpool_t threadpool = pthreadpool_();

      const pytorch_qnnp_status runStatus =
        pytorch_qnnp_run_operator(clamp_op, threadpool);

      TORCH_INTERNAL_ASSERT(
        runStatus == pytorch_qnnp_status_success,
        "failed to run QNNPACK Clamp operator");
      return qy;
        */
}

pub fn quantized_clamp_impl(
        qx:  &Tensor,
        min: &Option<Scalar>,
        max: &Option<Scalar>) -> Tensor {
    
    todo!();
        /*
            Tensor qy;
      if (min && max) {
    #ifdef USE_PYTORCH_QNNPACK
        if (globalContext().qEngine() == QEngine::QNNPACK &&
            qx.scalar_type() == kQUInt8) {
          return qnnpack_clamp(qx, *min, *max);
        }
    #endif
        qclamp_stub(qx.device().type(), qx, *min, *max, qy);
      } else {
    #ifdef USE_PYTORCH_QNNPACK
        if (globalContext().qEngine() == QEngine::QNNPACK) {
          TORCH_CHECK(
              false, "Both min and max should be specified for quantized clamp!");
        }
    #endif
        if (max) {
          qclamp_max_stub(qx.device().type(), qx, *max, qy);
        } else if (min) {
          qclamp_min_stub(qx.device().type(), qx, *min, qy);
        } else {
          TORCH_CHECK(false, "At least one of 'min' or 'max' must not be None");
        }
      }
      return qy;
        */
}

/// native functions for the native_functions.yaml
///
pub fn clamp_quantized_cpu(
        qx:  &Tensor,
        min: &Option<Scalar>,
        max: &Option<Scalar>) -> Tensor {
    
    todo!();
        /*
            Tensor qy;
      AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "clamp", [&]() {
        qy = quantized_clamp_impl(qx, min, max);
      });
      return qy;
        */
}

/// hardtanh is clamp with default min==-1.0f and
/// default max==1.0f
///
pub fn hardtanh_quantized_cpu(
        qx:  &Tensor,
        min: &Scalar,
        max: &Scalar) -> Tensor {
    
    todo!();
        /*
            Tensor qy;
      qy = quantized_clamp_impl(qx, min, max);
      return qy;
        */
}

pub fn hardtanh_quantized_cpu_mut<'a>(
        self_: &mut Tensor,
        min:   &Scalar,
        max:   &Scalar) -> &'a mut Tensor {
    
    todo!();
        /*
            Tensor qy;
      qy = quantized_clamp_impl(self, min, max);
      // This can be optimized in a future PR if it becomes a bottleneck.
      self.copy_(qy);
      return self;
        */
}

pub fn hardtanh_out_quantized_cpu<'a>(
        qx:     &Tensor,
        min:    &Scalar,
        max:    &Scalar,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            result = quantized_clamp_impl(qx, min, max);
      return result;
        */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("quantized::clamp"), TORCH_FN(clamp_quantized_cpu));
    }
    */
}
