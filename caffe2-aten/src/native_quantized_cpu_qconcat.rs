crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qconcat.cpp]

define_dispatch!{qcat_nhwc_stub}
define_dispatch!{qcat_relu_nhwc_stub}

pub fn is_cat_nhwc_fast_path(
        qxs: &List<Tensor>,
        dim: i32) -> bool {
    
    todo!();
        /*
            TORCH_CHECK(qxs.size() > 0);
      bool is_fast_path = dim == 1;
      for (const Tensor& qx : qxs) {
        is_fast_path &= qx.dim() == 4;
        is_fast_path &= qx.is_contiguous(MemoryFormat::ChannelsLast);
      }
      return is_fast_path;
        */
}

pub fn is_valid_quantization_scheme(t: &Tensor) -> bool {
    
    todo!();
        /*
            const auto qtype = t.qscheme();
      return (qtype == kPerTensorAffine) || (qtype == kPerTensorSymmetric);
        */
}

pub fn all_inputs_sharing_qparams(qxs: TensorList) -> bool {
    
    todo!();
        /*
            bool is_valid = true;
      for (int i = 1; i < qxs.size(); ++i) {
        is_valid |= qxs[0].is_quantized();
        is_valid |= qxs[i].is_quantized() == qxs[0].is_quantized();
        is_valid |= qxs[i].qscheme() == qxs[0].qscheme();
        is_valid |= qxs[i].dtype() == qxs[0].dtype();
        if (qxs[0].qscheme() == kPerTensorAffine) {
          is_valid |= qxs[i].q_scale() == qxs[0].q_scale();
          is_valid |= qxs[i].q_zero_point() == qxs[0].q_zero_point();
        } else if (qxs[0].qscheme() == kPerChannelAffine) {
          is_valid |= qxs[i].q_per_channel_scales().equal(qxs[0].q_per_channel_scales());
          is_valid |= qxs[i].q_per_channel_zero_points().equal(qxs[0].q_per_channel_zero_points());
        } else {
          TORCH_CHECK(false, "Unrecognized qscheme:", toString(qxs[0].qscheme()));
        }
      }
      return is_valid;
        */
}

/**
  | Quantized concatenation.
  | 
  | -----------
  | @note
  | 
  | This function uses a dequantization.
  |
  */
pub fn quantized_cat_impl<const ReLUFused: bool>(
        qxs:        &List<Tensor>,
        dim:        i64,
        scale:      f64,
        zero_point: i64) -> Tensor {

    todo!();
        /*
            if (is_cat_nhwc_fast_path(qxs, dim)) {
        if (ReLUFused) {
          return qcat_relu_nhwc_stub(kCPU, qxs, dim, scale, zero_point);
        } else {
          return qcat_nhwc_stub(kCPU, qxs, dim, scale, zero_point);
        }
      }

      const auto x_dtype = qxs.get(0).scalar_type();
      const auto x_qscheme = qxs.get(0).qscheme();
      vector<Tensor> xs;
      xs.reserve(qxs.size());
      for (const Tensor& qx : qxs) {
        TORCH_CHECK(x_dtype == qx.scalar_type(), "All dtypes must be the same.");
        TORCH_CHECK(
            x_qscheme == qx.qscheme(), "Quantization schemes must be the same.");
        xs.push_back(qx.dequantize());
      }
      const Tensor y = cat(xs, dim);
      Tensor qy;
      AT_DISPATCH_QINT_TYPES(x_dtype, "qcat", [&]() {
        qy = quantize_per_tensor(y, scale, zero_point, SCALAR_TYPE);
        if (ReLUFused) {
          auto iter = TensorIterator::unary_op(qy, qy);
          cpu_kernel(iter, [&](Scalar value) -> Scalar {
            return Scalar(max<underlying_t>(value.val_, zero_point));
          });
        }
      });
      return qy;
        */
}

pub fn qcat<const ReLUFused: bool = false>(
        qxs:        &List<Tensor>,
        dim:        i64,
        scale:      Option<f64>,
        zero_point: Option<i64>) -> Tensor {

    todo!();
        /*
            TORCH_CHECK(is_valid_quantization_scheme(qxs[0]),
                  "Only per-tensor quantization is supported in 'cat'!")
      double _scale = scale.has_value() ? scale.value() : qxs.get(0).q_scale();
      i64 _zero_point =
          zero_point.has_value() ? zero_point.value() : qxs.get(0).q_zero_point();
      return quantized_cat_impl<ReLUFused>(qxs, dim, _scale, _zero_point);
        */
}

pub fn qcat_out<const ReLUFused: bool = false>(
        qxs: &List<Tensor>,
        dim: i64,
        out: Tensor) -> Tensor {

    todo!();
        /*
            TORCH_CHECK(is_valid_quantization_scheme(qxs[0]),
                  "Only per-tensor quantization is supported in 'cat'!")
      TORCH_CHECK(is_valid_quantization_scheme(out),
                  "Only per-tensor quantization is supported in 'cat'!")
      auto out_ =
          quantized_cat_impl<ReLUFused>(qxs, dim, out.q_scale(), out.q_zero_point());
      native::copy_(out, out_, /*non_blocking=*/false);
      return out;
        */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("quantized::cat"), TORCH_FN(qcat<false>));
      m.impl(TORCH_SELECTIVE_NAME("quantized::cat_relu"), TORCH_FN(qcat<true>));
      m.impl(TORCH_SELECTIVE_NAME("quantized::cat_out"), TORCH_FN(qcat_out<false>));
      m.impl(TORCH_SELECTIVE_NAME("quantized::cat_relu_out"), TORCH_FN(qcat_out<true>));
    }
    */
}

pub fn cat_quantized_cpu(
        qxs: TensorList,
        dim: i64) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(is_valid_quantization_scheme(qxs[0]),
                  "Only per-tensor quantization is supported in 'cat'!");
      TORCH_CHECK(
          all_inputs_sharing_qparams(qxs),
          "All inputs should share the same quantization parameters.");

      double _scale = qxs[0].q_scale();
      i64 _zero_point = qxs[0].q_zero_point();
      return quantized_cat_impl<false>(List<Tensor>(qxs), dim, _scale, _zero_point);
        */
}

pub fn cat_out_quantized_cpu(
        qxs: TensorList,
        dim: i64,
        out: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(is_valid_quantization_scheme(qxs[0]),
                  "Only per-tensor quantization is supported in 'cat'!")
      TORCH_CHECK(is_valid_quantization_scheme(out),
                  "Only per-tensor quantization is supported in 'cat'!")
      auto out_ = quantized_cat_impl<false>(List<Tensor>(qxs), dim, out.q_scale(),
                                            out.q_zero_point());
      native::copy_(out, out_, /*non_blocking=*/false);
      return out;
        */
}
