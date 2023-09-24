crate::ix!();

pub fn register_linear_params() -> TorchClass<LinearPackedParamsBase> {
    
    todo!();
        /*
            using SerializationType = tuple<Tensor, optional<Tensor>>;
      static auto register_linear_params =
          Torchclass_<LinearPackedParamsBase>(
              "quantized", "LinearPackedParamsBase")
              .def_pickle(
                  [](const intrusive_ptr<LinearPackedParamsBase>& params)
                      -> SerializationType { // __getstate__
                    Tensor weight;
                    optional<Tensor> bias;
                    tie(weight, bias) = params->unpack();
                    return make_tuple(move(weight), move(bias));
                  },
                  [](SerializationType state)
                      -> intrusive_ptr<
                          LinearPackedParamsBase> { // __setstate__
                    Tensor weight;
                    optional<Tensor> bias;
                    weight = move(get<0>(state));
                    bias = move(get<1>(state));

    #ifdef USE_FBGEMM
                    if (globalContext().qEngine() == QEngine::FBGEMM) {
                      if (weight.scalar_type() == kQInt8) {
                        return PackedLinearWeight::prepack(
                            move(weight), move(bias));
                      } else if (weight.scalar_type() == kFloat) {
                        // NB: fp16 weight is serialized as float
                        return PackedLinearWeightFp16::prepack(
                            move(weight), move(bias));
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
                      TORCH_CHECK(
                          weight.scalar_type() == kQInt8,
                          "QNNPACK only supports INT8 bit width currently. Got ",
                          toString(weight.scalar_type()));
                      return PackedLinearWeightsQnnp::prepack(
                          move(weight), move(bias));
                    }
    #endif // USE_PYTORCH_QNNPACK
                    TORCH_CHECK(false, "Unknown qengine");
                  });
      return register_linear_params;
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.cpp]
pub fn register_linear_params2() -> TorchClass<LinearPackedParamsBase> {
    
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
