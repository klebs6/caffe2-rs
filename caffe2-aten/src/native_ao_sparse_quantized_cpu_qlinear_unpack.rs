crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ao_sparse/quantized/cpu/qlinear_unpack.cpp]

pub fn register_linear_params() -> TorchClass<LinearPackedParamsBase> {
    
    todo!();
        /*
        
        */
}

impl PackedLinearWeight {
    
    #[cfg(USE_FBGEMM)]
    pub fn unpack(&mut self) -> LinearPackedSerializationType {
        
        todo!();
        /*
            auto packW = w.get();

      i64 N = static_cast<i64>(packW->R);
      i64 K = static_cast<i64>(packW->C);

      Tensor weight_origin;
      if (q_scheme == kPerTensorAffine) {
        weight_origin = _empty_affine_quantized(
            {N, K}, device(kCPU).dtype(kQInt8), w_scale[0], w_zp[0]);
      } else if (q_scheme == kPerChannelAffine) {
        auto scales = from_blob(
            w_scale.data(), w_scale.size(), device(kCPU).dtype(kFloat));
        auto zero_points = from_blob(
            w_zp.data(), w_zp.size(), device(kCPU).dtype(kInt));

        weight_origin = _empty_per_channel_affine_quantized(
            {N, K},
            scales.toType(kDouble),
            zero_points.toType(kLong),
            0, // The output channel axis is 0
            device(kCPU).dtype(kQInt8));
      }

      // TODO: uncomment once unpack is implemented for BCSRMatrix
      // i8* weight_ptr_int8 =
      //     reinterpret_cast<i8*>(weight_origin.data_ptr<qint8>());
      // packW->unpack(weight_ptr_int8);
      vector<i64> block_pattern(
          {out_features_block_size_, in_features_block_size_});

      return make_tuple(weight_origin, bias_, move(block_pattern));
        */
    }
}

impl PackedLinearWeightQnnp {
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn unpack(&mut self) -> LinearPackedSerializationType {
        
        todo!();
        /*
            vector<i64> block_pattern(
          {out_features_block_size_, in_features_block_size_});
      return make_tuple(orig_weight_, orig_bias_, move(block_pattern));
        */
    }
}

pub struct QLinearUnpackWeightInt8 {

}

impl QLinearUnpackWeightInt8 {
    
    pub fn run(packed_weight: &IntrusivePtr<LinearPackedParamsBase>) -> LinearPackedSerializationType {
        
        todo!();
        /*
            return packed_weight->unpack();
        */
    }
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(sparse, QuantizedCPU, m) {
      m.impl(
          TORCH_SELECTIVE_NAME("sparse::qlinear_unpack"),
          TORCH_FN(QLinearUnpackWeightInt8::run));
    }
    */
}
