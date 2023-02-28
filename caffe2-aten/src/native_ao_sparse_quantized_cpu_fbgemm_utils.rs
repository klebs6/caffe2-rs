crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.h]

#[cfg(feature = "fbgemm")]
pub struct PackedLinearWeight {
    base:        LinearPackedParamsBase,
    w:           Box<FbgemmBCSRMatrix<i8>>,
    bias:        Option<Tensor>,
    col_offsets: Vec<i32>,
    w_scale:     Vec<f32>,
    w_zp:        Vec<i32>,
    q_scheme:    QScheme,
}

#[cfg(feature = "fbgemm")]
impl PackedLinearWeight {

    pub fn new(
        w:                       Box<FbgemmBCSRMatrix<i8>>,
        bias:                    Option<Tensor>,
        col_offsets:             Vec<i32>,
        w_scale:                 Vec<f32>,
        w_zp:                    Vec<i32>,
        q_scheme:                QScheme,

        /** block sparsity size across output_features */
        out_features_block_size: i64,

        /** block sparsity size across input_features */
        in_features_block_size:  i64) -> Self {
    
        todo!();
        /*
        : linear_packed_params_base(out_features_block_size,
                    in_features_block_size),
        : w(move(w)),
        : bias(move(bias)),
        : col_offsets(move(col_offsets)),
        : w_scale(move(w_scale)),
        : w_zp(move(w_zp)),
        : q_scheme(q_scheme),

        
        */
    }
    
    pub fn apply(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
        
        */
    }
    
    pub fn apply_relu(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
        
        */
    }
    
    pub fn apply_dynamic(&mut self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(
            false,
            "Sparse quantized dynamic linear with fused relu is not yet "
            "supported on qnnpack backend.");
        return Tensor();
        */
    }
    
    pub fn apply_dynamic_relu(&mut self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(
            false,
            "Sparse quantized dynamic linear with fused relu is not yet "
            "supported on qnnpack backend.");
        return Tensor();
        */
    }
    
    pub fn unpack(&mut self) -> LinearPackedSerializationType {
        
        todo!();
        /*
        
        */
    }
    
    pub fn bias(&mut self) -> Option<Tensor> {
        
        todo!();
        /*
            return bias_;
        */
    }
    
    pub fn prepack(
        weight:                  &Tensor,
        bias:                    &Option<Tensor>,
        out_features_block_size: i64,
        in_features_block_size:  i64) -> IntrusivePtr<LinearPackedParamsBase> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn apply_impl<const ReluFused: bool>(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
    
        todo!();
        /*
        
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.cpp]

pub fn register_linear_params() -> TorchClass<LinearPackedParamsBase> {
    
    todo!();
        /*
            static auto register_linear_params =
          Torchclass_<LinearPackedParamsBase>(
              "sparse", "LinearPackedParamsBase")
              .def_pickle(
                  [](const intrusive_ptr<LinearPackedParamsBase>& params)
                      -> LinearPackedSerializationType { // __getstate__
                    return params->unpack();
                  },
                  [](LinearPackedSerializationType state)
                      -> intrusive_ptr<
                          LinearPackedParamsBase> { // __setstate__
                    Tensor weight;
                    optional<Tensor> bias;
                    i64 out_features_block_size, in_features_block_size;
                    weight = move(get<0>(state));
                    bias = move(get<1>(state));
                    out_features_block_size = get<2>(state)[0];
                    in_features_block_size = get<2>(state)[1];

    #ifdef USE_FBGEMM
                    if (globalContext().qEngine() == QEngine::FBGEMM) {
                      if (weight.scalar_type() == kQInt8) {
                        return PackedLinearWeight::prepack(
                            weight,
                            bias,
                            out_features_block_size,
                            in_features_block_size);
                      } else {
                        TORCH_CHECK(
                            false,
                            "Unsupported data type",
                            toString(weight.scalar_type()),
                            " in serialized LinearPackedParams object!");
                      }
                    }
    #endif // USE_FBGEMM
    #ifdef USE_PYTORCH_QNNPACK
                    if (globalContext().qEngine() == QEngine::QNNPACK) {
                      if (weight.scalar_type() == kQInt8) {
                        return PackedLinearWeightQnnp::prepack(
                            weight,
                            bias,
                            out_features_block_size,
                            in_features_block_size);
                      } else {
                        TORCH_CHECK(
                            false,
                            "Unsupported data type",
                            toString(weight.scalar_type()),
                            " in serialized LinearPackedParams object!");
                      }
                    }
    #endif // USE_FBGEMM
                    TORCH_CHECK(false, "Unknown qengine");
                  });
      return register_linear_params;
        */
}


lazy_static!{
    /*
    static auto linear_params = register_linear_params();
    */
}
