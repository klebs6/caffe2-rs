crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qelu.cpp]

define_dispatch!{qelu_stub}

pub fn quantized_elu(
        qx:                &Tensor,
        output_scale:      f64,
        output_zero_point: i64,
        alpha:             &Scalar,
        scale:             &Scalar,
        input_scale:       &Scalar) -> Tensor {
    
    todo!();
        /*
            Tensor qy = _empty_affine_quantized(qx.sizes(), qx.options(), output_scale, output_zero_point);
      qelu_stub(qx.device().type(), qx, alpha, scale, input_scale, qy);
      return qy;
        */
}

pub fn quantized_celu(
        qx:                &Tensor,
        output_scale:      f64,
        output_zero_point: i64,
        alpha:             &Scalar) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(alpha.to<double>() != 0,
          "ZeroDivisionError: alpha cannot be 0 for CELU");
      double inv_alpha = 1. / alpha.to<double>();
      return quantized_elu(qx, output_scale, output_zero_point, alpha, Scalar(1.0), Scalar(inv_alpha));
        */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("quantized::elu"), quantized_elu);
      m.impl(TORCH_SELECTIVE_NAME("quantized::celu"), quantized_celu);
    }
    */
}
