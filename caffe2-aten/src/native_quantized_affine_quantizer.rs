crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/affine_quantizer.h]

pub type QuantizeTensorPerTensorAffineFn = fn(
        rtensor:    &Tensor,
        qtensor:    &mut Tensor,
        scale:      f64,
        zero_point: i64
) -> c_void;

pub type QuantizeTensorPerChannelAffineFn = fn(
        rtensor:     &Tensor,
        qtensor:     &mut Tensor,
        scales:      &Tensor,
        zero_points: &Tensor,
        axis:        i64
) -> c_void;

pub type QuantizeTensorPerChannelFloatQparamsFn = fn(
        rtensor:     &Tensor,
        qtensor:     &mut Tensor,
        scales:      &Tensor,
        zero_points: &Tensor,
        axis:        i64
) -> c_void;

pub type DequantizeTensorPerTensorAffineFn = fn(
        qtensor:    &Tensor,
        rtensor:    &mut Tensor,
        scale:      f64,
        zero_point: i64
) -> c_void;

pub type DequantizeTensorPerChannelAffineFn = fn(
        qtensor:     &Tensor,
        rtensor:     &mut Tensor,
        scales:      &Tensor,
        zero_points: &Tensor,
        axis:        i64
) -> c_void;

pub type DequantizeTensorPerChannelFloatQparamsFn = fn(
        qtensor:     &Tensor,
        rtensor:     &mut Tensor,
        scales:      &Tensor,
        zero_points: &Tensor,
        axis:        i64
) -> c_void;

pub type QuantizeTensorPerTensorAffineSubByteFn = fn(
        rtensor:    &Tensor,
        qtensor:    &mut Tensor,
        scale:      f32,
        zero_point: f32
) -> c_void;

pub type DequantizeTensorPerTensorAffineSubByteFn = fn(
        qtensor:    &Tensor,
        rtensor:    &mut Tensor,
        scale:      f32,
        zero_point: f32
) -> c_void;

declare_dispatch!{quantize_tensor_per_tensor_affine_fn, quantize_tensor_per_tensor_affine_stub}
declare_dispatch!{quantize_tensor_per_channel_affine_fn, quantize_tensor_per_channel_affine_stub}
declare_dispatch!{quantize_tensor_per_channel_float_qparams_fn, quantize_tensor_per_channel_float_qparams_stub}
declare_dispatch!{dequantize_tensor_per_tensor_affine_fn, dequantize_tensor_per_tensor_affine_stub}
declare_dispatch!{dequantize_tensor_per_channel_affine_fn, dequantize_tensor_per_channel_affine_stub}
declare_dispatch!{dequantize_tensor_per_channel_float_qparams_fn, dequantize_tensor_per_channel_float_qparams_stub}
declare_dispatch!{quantize_tensor_per_tensor_affine_sub_byte_fn, quantize_tensor_per_tensor_affine_sub_byte_stub}
declare_dispatch!{dequantize_tensor_per_tensor_affine_sub_byte_fn, dequantize_tensor_per_tensor_affine_sub_byte_stub}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/affine_quantizer.cpp]

define_dispatch!{quantize_tensor_per_tensor_affine_stub}
define_dispatch!{quantize_tensor_per_channel_affine_stub}
define_dispatch!{quantize_tensor_per_channel_float_qparams_stub}
define_dispatch!{dequantize_tensor_per_tensor_affine_stub}
define_dispatch!{dequantize_tensor_per_channel_affine_stub}
define_dispatch!{dequantize_tensor_per_channel_float_qparams_stub}
define_dispatch!{quantize_tensor_per_tensor_affine_sub_byte_stub}
define_dispatch!{dequantize_tensor_per_tensor_affine_sub_byte_stub}

pub fn check_rounding_mode(fn_name: &String)  {
    
    todo!();
        /*
            // Disabling this warning message for now as it is printed incorrectly. Need
      // to fix

      /*  TORCH_WARN_ONCE(
            fegetround() != FE_TONEAREST,
            fn_name,
            " current rounding mode is not set to round-to-nearest-ties-to-even
         (FE_TONEAREST). This will cause accuracy issues in quantized models.");
      */
      return;
        */
}

pub fn check_cpu_tensor(
        fn_name: &String,
        t:       &Tensor)  {
    
    todo!();
        /*
            TORCH_CHECK(
          t.device().type() == kCPU, fn_name, " only supports CPU device type.");
        */
}

pub fn check_float_tensor(
        fn_name: &String,
        t:       &Tensor)  {
    
    todo!();
        /*
            TORCH_CHECK(t.scalar_type() == kFloat, fn_name, " expects a Float Tensor.");
        */
}

pub fn check_same_device(
        fn_name: &String,
        t1:      &Tensor,
        t2:      &Tensor)  {
    
    todo!();
        /*
            TORCH_CHECK(
          t1.device() == t2.device(),
          fn_name,
          " expects a quantized and float tensors to be on the same device.");
        */
}

pub fn check_quantized_tensor<T>(
        fn_name: &String,
        t:       &Tensor)  {

    todo!();
        /*
            TORCH_CHECK(t.is_quantized(), fn_name, " expects a quantized Tensor.");
      TORCH_CHECK(
          t.scalar_type() == TypeMeta::Make<T>(),
          fn_name,
          " expects a ",
          TypeMeta::Make<T>(),
          " Tensor, got ",
          t.scalar_type());
        */
}

pub fn check_zero_point<T>(
        fn_name:    &String,
        zero_point: i64)  {

    todo!();
        /*
            TORCH_CHECK(
          zero_point <= T::max,
          fn_name,
          " zero_point ",
          zero_point,
          " is out of range.");
      TORCH_CHECK(
          zero_point >= T::min,
          fn_name,
          " zero_point ",
          zero_point,
          " is out of range.");
        */
}

pub fn check_zero_points<T>(
        fn_name:     &String,
        zero_points: &Tensor)  {

    todo!();
        /*
            auto zero_points_data = zero_points.data_ptr<i64>();
      for (usize i = 0; i < zero_points.numel(); ++i) {
        checkZeroPoint<T>(fn_name, zero_points_data[i]);
      }
        */
}

pub fn check_same_size(
        fn_name: &String,
        qt:      &Tensor,
        rt:      &Tensor)  {
    
    todo!();
        /*
            TORCH_CHECK(
          qt.sizes().equals(rt.sizes()),
          fn_name,
          " only works with Tensors with the same shape");
        */
}


pub fn quantize_tensor_per_tensor_affine(
        rtensor:    &Tensor,
        qtensor:    &mut Tensor,
        scale:      f64,
        zero_point: i64) -> &mut Tensor {
    
    todo!();
        /*
            static const string fn_name = "quantize_tensor_per_tensor_affine";

      checkRoundingMode(fn_name);
      checkFloatTensor(fn_name, rtensor);
      checkSameDevice(fn_name, rtensor, qtensor);
      checkSameSize(fn_name, qtensor, rtensor);

      AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(qtensor.scalar_type(), fn_name, [&]() {
        checkQuantizedTensor<Scalar>(fn_name, qtensor);
        checkZeroPoint<underlying_t>(fn_name, zero_point);
      });

      // Temporary solution to pack the tensor if dtype is torch.quint4x2
      // Can move this into the fbgemm::Quantize op.
      if (qtensor.scalar_type() == ScalarType::QUInt4x2) {
        quantize_tensor_per_tensor_affine_sub_byte_stub(
            rtensor.device().type(), rtensor, qtensor, scale, zero_point);
      } else {
        quantize_tensor_per_tensor_affine_stub(
            rtensor.device().type(), rtensor, qtensor, scale, zero_point);
      }
      return qtensor;
        */
}


pub fn quantize_tensor_per_channel_affine(
        rtensor:     &Tensor,
        qtensor:     &mut Tensor,
        scales:      Tensor,
        zero_points: Tensor,
        axis:        i64) -> &mut Tensor {
    
    todo!();
        /*
            static const string fn_name = "quantize_tensor_per_channel_affine";

      checkRoundingMode(fn_name);
      checkFloatTensor(fn_name, rtensor);
      checkCPUTensor(fn_name, rtensor);
      checkSameDevice(fn_name, rtensor, qtensor);
      checkSameSize(fn_name, qtensor, rtensor);

      AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {
        checkQuantizedTensor<Scalar>(fn_name, qtensor);
        checkZeroPoints<underlying_t>(fn_name, zero_points);
      });

      TORCH_CHECK(
          0 <= axis && axis < rtensor.dim(),
          "Channel axis out of range in per channel affine quantization. Got: ",
          axis,
          "Expected: [0, ",
          rtensor.dim(),
          ")");
      i64 channel = rtensor.size(axis);
      TORCH_CHECK(
          channel == i64(scales.numel()),
          "length of scales must equal to channel");
      TORCH_CHECK(
          channel == i64(zero_points.numel()),
          "length of zero_points must equal to channel");

      quantize_tensor_per_channel_affine_stub(
          rtensor.device().type(), rtensor, qtensor, scales, zero_points, axis);
      return qtensor;
        */
}


pub fn quantize_tensor_per_channel_float_qparams(
        rtensor:     &Tensor,
        qtensor:     &mut Tensor,
        scales:      Tensor,
        zero_points: Tensor,
        axis:        i64) -> &mut Tensor {
    
    todo!();
        /*
            static const string fn_name =
          "quantize_tensor_per_channel_float_qparams";

      checkRoundingMode(fn_name);
      checkFloatTensor(fn_name, rtensor);
      checkCPUTensor(fn_name, rtensor);
      checkSameDevice(fn_name, rtensor, qtensor);
      checkSameSize(fn_name, qtensor, rtensor);

      AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(qtensor.scalar_type(), fn_name, [&]() {
        checkQuantizedTensor<Scalar>(fn_name, qtensor);
      });

      TORCH_CHECK(
          0 <= axis && axis < rtensor.dim(),
          "Channel axis out of range in per channel float qparams quantization. Got: ",
          axis,
          "Expected: [0, ",
          rtensor.dim(),
          ")");
      i64 channel = rtensor.size(axis);
      TORCH_CHECK(
          channel == i64(scales.numel()),
          "length of scales must equal to channel");
      TORCH_CHECK(
          channel == i64(zero_points.numel()),
          "length of zero_points must equal to channel");

      quantize_tensor_per_channel_float_qparams_stub(
          rtensor.device().type(), rtensor, qtensor, scales, zero_points, axis);
      return qtensor;
        */
}


pub fn dequantize_tensor_per_tensor_affine(
        qtensor:    &Tensor,
        rtensor:    &mut Tensor,
        scale:      f64,
        zero_point: i64) -> &mut Tensor {
    
    todo!();
        /*
            static const string fn_name = "dequantize_tensor_per_tensor_affine";
      checkFloatTensor(fn_name, rtensor);
      checkSameDevice(fn_name, rtensor, qtensor);
      checkSameSize(fn_name, qtensor, rtensor);

      AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(qtensor.scalar_type(), fn_name, [&]() {
        checkQuantizedTensor<Scalar>(fn_name, qtensor);
        checkZeroPoint<underlying_t>(fn_name, zero_point);
      });

      if (qtensor.scalar_type() == ScalarType::QUInt4x2) {
        dequantize_tensor_per_tensor_affine_sub_byte_stub(
            qtensor.device().type(), qtensor, rtensor, scale, zero_point);
      } else {
        dequantize_tensor_per_tensor_affine_stub(
            qtensor.device().type(), qtensor, rtensor, scale, zero_point);
      }
      return rtensor;
        */
}


pub fn dequantize_tensor_per_channel_affine(
        qtensor:     &Tensor,
        rtensor:     &mut Tensor,
        scales:      Tensor,
        zero_points: Tensor,
        axis:        i64) -> &mut Tensor {
    
    todo!();
        /*
            static const string fn_name = "dequantize_tensor_per_channel_affine";

      checkFloatTensor(fn_name, rtensor);
      checkCPUTensor(fn_name, rtensor);
      checkSameDevice(fn_name, rtensor, qtensor);
      checkSameSize(fn_name, qtensor, rtensor);

      AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {
        checkQuantizedTensor<Scalar>(fn_name, qtensor);
        checkZeroPoints<underlying_t>(fn_name, zero_points);
      });

      TORCH_CHECK(
          0 <= axis && axis < qtensor.dim(),
          "Channel axis out of range in per channel affine dequantization. Got:",
          axis,
          " Expected: [0, ",
          qtensor.dim(),
          ")");
      i64 channel = qtensor.size(axis);
      TORCH_CHECK(
          channel == i64(scales.numel()),
          "length of scales must equal to channel");
      TORCH_CHECK(
          channel == i64(zero_points.numel()),
          "length of zero_points must equal to channel");

      dequantize_tensor_per_channel_affine_stub(
          qtensor.device().type(), qtensor, rtensor, scales, zero_points, axis);
      return rtensor;
        */
}


pub fn dequantize_tensor_per_channel_float_qparams(
        qtensor:     &Tensor,
        rtensor:     &mut Tensor,
        scales:      Tensor,
        zero_points: Tensor,
        axis:        i64) -> &mut Tensor {
    
    todo!();
        /*
            static const string fn_name = "dequantize_tensor_per_channel_affine";

      checkFloatTensor(fn_name, rtensor);
      checkCPUTensor(fn_name, rtensor);
      checkSameDevice(fn_name, rtensor, qtensor);
      checkSameSize(fn_name, qtensor, rtensor);

      AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(qtensor.scalar_type(), fn_name, [&]() {
        checkQuantizedTensor<Scalar>(fn_name, qtensor);
      });

      TORCH_CHECK(
          0 <= axis && axis < qtensor.dim(),
          "Channel axis out of range in per channel float qparams dequantization. Got:",
          axis,
          " Expected: [0, ",
          qtensor.dim(),
          ")");
      i64 channel = qtensor.size(axis);
      TORCH_CHECK(
          channel == i64(scales.numel()),
          "length of scales must equal to channel");
      TORCH_CHECK(
          channel == i64(zero_points.numel()),
          "length of zero_points must equal to channel");

      dequantize_tensor_per_channel_float_qparams_stub(
          qtensor.device().type(), qtensor, rtensor, scales, zero_points, axis);
      return rtensor;
        */
}
