crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qthreshold.cpp]

define_dispatch!{qthreshold_stub}

/**
  | the underlying implementation for
  | quantized threshold kernel
  |
  */
pub fn quantized_threshold_impl(
    qx:        &Tensor,
    threshold: &Scalar,
    value:     &Scalar) -> Tensor {
    
    todo!();
        /*
            Tensor qy = _empty_affine_quantized(
        qx.sizes(), qx.options(), qx.q_scale(), qx.q_zero_point());
      qthreshold_stub(qx.device().type(), qx, threshold, value, qy);
      return qy;
        */
}

/**
  | native functions for the native_functions.yaml
  |
  */
pub fn threshold_quantized_cpu(
    qx:        &Tensor,
    threshold: &Scalar,
    value:     &Scalar) -> Tensor {
    
    todo!();
        /*
            Tensor qy;
      AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "threshold", [&]() {
        qy = quantized_threshold_impl(qx, threshold, value);
      });
      return qy;
        */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("quantized::threshold"), TORCH_FN(threshold_quantized_cpu));
    }
    */
}
