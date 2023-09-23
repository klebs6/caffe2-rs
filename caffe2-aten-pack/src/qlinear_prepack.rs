crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qlinear_prepack.cpp]

/**
  | Calculate the column offsets.
  |
  | Note this includes the sum of the columns as
  | well as the scalar term B_zero_point * K,
  | whereas the row_offsets created by
  | PackAWithQuantRowOffset is only the sum of the
  | A rows.
  |
  */
#[cfg(feature = "fbgemm")]
pub fn calc_col_offsets_transpose(
        K:            i32,
        N:            i32,
        bint8:        *const i8,
        b_zero_point: *mut i32,
        col_offsets:  *mut i32,
        qtype:        QScheme)  {
    
    todo!();
        /*
      for (usize i = 0; i < N; ++i) {
        i32 sum = 0;
        for (usize j = 0; j < K; ++j) {
          sum += Bint8[i * K + j];
        }
        if (qtype == kPerTensorAffine) {
          col_offsets[i] = sum - B_zero_point[0] * K;
        } else {
          col_offsets[i] = sum - B_zero_point[i] * K;
        }
      }
        */
}

pub fn saturate_weight_to_fp16(weight: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor weight_contig = weight.contiguous();
      float* weight_contig_ptr = weight_contig.data_ptr<float>();
      quant_utils::HandleWeightsSaturation(weight.size(0) * weight.size(1), weight_contig_ptr);
      return weight;
        */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("quantized::linear_prepack"), TORCH_FN(QLinearPackWeightInt8::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::linear_prepack_legacy"), TORCH_FN(QLinearPackWeightInt8Legacy::run));
    }
    */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, CPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("quantized::linear_prepack_fp16"), TORCH_FN(QLinearPackWeightFp16::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::linear_prepack_fp16_legacy"), TORCH_FN(QLinearPackWeightFp16Legacy::run));
    }
    */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(_quantized, QuantizedCPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("_quantized::linear_prepack"), TORCH_FN(QLinearPackWeightInt8::run));
    }
    */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(_quantized, CPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("_quantized::linear_prepack_fp16"), TORCH_FN(QLinearPackWeightFp16::run));
      m.impl(TORCH_SELECTIVE_NAME("_quantized::linear_prepack_fp16_legacy"), TORCH_FN(QLinearPackWeightFp16Legacy::run));
    }
    */
}
