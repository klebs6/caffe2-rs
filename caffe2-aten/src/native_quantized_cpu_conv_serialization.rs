/*! 
 | Convolution prepacked parameters serialization.
 |
 | Version 1
 |
 | - Fields:
 |  1. weight
 |  2. bias
 |  3. stride x kSpatialDim
 |  4. padding x kSpatialDim
 |  5. dilation x kSpatialDim
 |  6. groups
 |
 | Version 2
 |
 | - Fields:
 |  0. version (string)
 |  1. list of non-optional tensors
 |    0: packed parameters (i16)
 |      - kSpatialDim
 |      - stride x kSpatialDim
 |      - padding x kSpatialDim
 |      - dilation x kSpatialDim
 |      - output_padding x kSpatialDim
 |      - groups
 |      - transpose (0 or 1)
 |    1: weight
 |  2. list of optional tensors
 |    0: bias
 |
 |  Note: version is a string and conv params are
 |    packed into a Tensor to make ONNX happy
 |    (ints and containers of ints are not
 |    supported).
 */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/conv_serialization.h]

// version 2
lazy_static!{
    /*
    using ConvParamsSerializationType = tuple<
      // version, for versions 2 and up
      string,
      // non-optional tensors
      vector<Tensor>,
      // optional tensors
      vector<optional<Tensor>>>;
    */
}

/**
  | Parses any historical conv packed params
  | format into the current format.
  |
  */
pub fn parse_conv_serialized_state<const kSpatialDim: u32>(v: IValue) -> ConvParamsSerializationType {

    todo!();
        /*
            // determine the version based on IValue contents
      int version = -1;
      if (v.isTuple()) {
        auto elements = v.toTuple()->elements();
        if (elements.size() > 0) {
          auto firstElement = elements[0];
          if (firstElement.isTensor()) {
            version = 1;
          } else if (firstElement.isString()) {
            string version_str = firstElement.toStringRef();
            // note: not parsing the string to automatically handle bad
            // inputs
            if (version_str == "2") {
              version = 2;
            }
          }
        }
      }
      TORCH_INTERNAL_ASSERT(version != -1, "Unable to parse serialization version");

      if (version == 1) {
        // version 1 - convert to version 2 manually

        auto elements = v.toTuple()->elements();

        Tensor weight = elements[0].toTensor();
        optional<Tensor> bias = elements[1].toOptional<Tensor>();
        TorchList<Tensor> stride_x_kSpatialDim = elements[2].toTensorList();
        TorchList<Tensor> padding_x_kSpatialDim = elements[3].toTensorList();
        TorchList<Tensor> dilation_x_kSpatialDim = elements[4].toTensorList();
        Tensor groups = elements[5].toTensor();

        string version = "2";
        vector<Tensor> non_optional;
        vector<optional<Tensor>> optional;

        vector<i16> params_vec;
        params_vec.push_back(kSpatialDim);
        for (int i = 0; i < stride_x_kSpatialDim.size(); i++) {
          auto stride = stride_x_kSpatialDim.get(i);
          params_vec.push_back(stride[0].item<i16>());
        }
        for (int i = 0; i < padding_x_kSpatialDim.size(); i++) {
          auto padding = padding_x_kSpatialDim.get(i);
          params_vec.push_back(padding[0].item<i16>());
        }
        for (int i = 0; i < dilation_x_kSpatialDim.size(); i++) {
          auto dilation = dilation_x_kSpatialDim.get(i);
          params_vec.push_back(dilation[0].item<i16>());
        }
        // output_padding does not exist in v1, so we fill in a default value
        for (int i = 0; i < kSpatialDim; i++) {
          params_vec.push_back(0);
        }
        params_vec.push_back(groups[0].item<i16>());
        // transpose does not exist in v1, so we fill in a default value
        params_vec.push_back(0);
        i64 vec_size = params_vec.size();
        Tensor params_tensor = from_blob(params_vec.data(),
            {vec_size}, TensorOptions().dtype(kShort))
          // clone to retain ownership of the data
          .clone();

        non_optional.emplace_back(move(params_tensor));
        non_optional.emplace_back(move(weight));
        optional.emplace_back(move(bias));

        return tie(version, non_optional, optional);
      } else if (version == 2) {
        // version 2
        auto elements = v.toTuple()->elements();
        vector<Tensor> non_optional = elements[1].toTensorList().vec();
        vector<optional<Tensor>> optional;

        if (elements[2].isTensorList()) {
          for (const auto& elem : elements[2].toTensorList()) {
            optional.emplace_back(static_cast<Tensor>(elem));
          }
        } else {
          for (const auto& elem : elements[2].toList()) {
            optional.emplace_back(static_cast<IValue>(elem).toOptional<Tensor>());
          }
        }

        string version = "2";
        return tie(version, non_optional, optional);
      } else {
        TORCH_INTERNAL_ASSERT(false, "Unexpected serialized qconv version: ",
            version);
      }
        */
}

pub fn serialize_conv<const kSpatialDim: u32>(params: &IntrusivePtr<ConvPackedParamsBase<SpatialDim>>) -> ConvParamsSerializationType {

    todo!();
        /*
            string version = "2";
      vector<Tensor> non_optional;
      vector<optional<Tensor>> optional;

      // create a packed i8 tensor for conv params
      vector<i16> params_vec;
      params_vec.push_back(kSpatialDim);
      auto stride = params->stride().vec();
      params_vec.insert(params_vec.end(), stride.begin(), stride.end());
      auto padding = params->padding().vec();
      params_vec.insert(params_vec.end(), padding.begin(), padding.end());
      auto dilation = params->dilation().vec();
      params_vec.insert(params_vec.end(), dilation.begin(), dilation.end());
      auto output_padding = params->output_padding().vec();
      params_vec.insert(params_vec.end(), output_padding.begin(),
                        output_padding.end());
      params_vec.push_back(params->groups());
      params_vec.push_back(params->transpose());
      i64 vec_size = params_vec.size();
      Tensor params_tensor = from_blob(
          params_vec.data(), {vec_size},
          TensorOptions().dtype(kShort))
        // clone to retain ownership of the data
        .clone();

      Tensor weight;
      optional<Tensor> bias;
      tie(weight, bias) = params->unpack();

      non_optional.emplace_back(move(params_tensor));
      non_optional.emplace_back(move(weight));
      optional.emplace_back(move(bias));

      return tie(version, non_optional, optional);
        */
}

pub fn deserialize_conv<const kSpatialDim: u32>(state: ConvParamsSerializationType) -> IntrusivePtr<ConvPackedParamsBase<SpatialDim>> {

    todo!();
        /*
            string version;
      vector<Tensor> non_optional;
      vector<optional<Tensor>> optional;

      tie(version, non_optional, optional) = state;
      TORCH_INTERNAL_ASSERT(version == "2", "Unexpected serialized qconv version: ",
          version);

      Tensor conv_params_packed = non_optional[0];
      Tensor weight = non_optional[1];
      optional<Tensor> bias = optional[0];

      TorchList<i64> stride, padding, output_padding, dilation;
      // skip kSpatialDim
      int idx = 1;
      for (int i = 0; i < kSpatialDim; ++i) {
        stride.emplace_back(conv_params_packed[idx].item<i64>());
        idx++;
      }
      for (int i = 0; i < kSpatialDim; ++i) {
        padding.emplace_back(conv_params_packed[idx].item<i64>());
        idx++;
      }
      for (int i = 0; i < kSpatialDim; ++i) {
        dilation.emplace_back(conv_params_packed[idx].item<i64>());
        idx++;
      }
      for (int i = 0; i < kSpatialDim; ++i) {
        output_padding.emplace_back(conv_params_packed[idx].item<i64>());
        idx++;
      }
      i64 groups = conv_params_packed[idx].item<i64>();
      idx++;
      bool transpose = conv_params_packed[idx].item<bool>();
      idx++;
      TORCH_INTERNAL_ASSERT(idx == conv_params_packed.numel(),
          "Unexpected length of conv_params_packed, expected ",
          idx,
          " got ",
          conv_params_packed.numel());

      auto& ctx = globalContext();

    #ifdef USE_FBGEMM
      if (ctx.qEngine() == QEngine::FBGEMM) {
        return PackedConvWeight<kSpatialDim>::prepack(
          weight,
          bias,
          stride,
          padding,
          output_padding,
          dilation,
          groups,
          transpose
        );
      }
    #endif // USE_FBGEMM
    #ifdef USE_PYTORCH_QNNPACK
      if (ctx.qEngine() == QEngine::QNNPACK) {
        TORCH_CHECK(
            kSpatialDim == 2,
            "prepack/__setstate__: QNNPACK only supports Conv2d "
            "now.");
        return PackedConvWeightsQnnp<kSpatialDim>::prepack(
          weight,
          bias,
          stride,
          padding,
          output_padding,
          dilation,
          groups,
          transpose
        );
      }
    #endif // USE_PYTORCH_QNNPACK
    TORCH_CHECK(
      false,
      "Didn't find engine for when deserializing ConvPackedParams: ",
      toString(ctx.qEngine()));
        */
}
