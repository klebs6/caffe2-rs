crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qembeddingbag.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qembeddingbag.cpp]

pub fn register_embedding_params() -> TorchClass<EmbeddingPackedParamsBase> {
    
    todo!();
        /*
        
        */
}

/**
  | Fallback implementation when FBGEMM
  | is not available.
  |
  */
pub fn embedding_lookup_fallback_impl<'a, IndexType, OffsetType, const BIT_RATE: i32, const NUM_ELEM_PER_BYTE: i32>(
    weight:                     &Tensor,
    indices:                    &Tensor,
    offsets:                    &Tensor,
    per_sample_weights:         &Option<Tensor>,
    compressed_indices_mapping: &Option<Tensor>,
    output:                     &mut Tensor,
    block_size:                 i64,
    output_size:                i64,
    include_last_offset:        bool,
    pruned:                     bool) -> &'a mut Tensor {

    todo!();
        /*
            auto* output_data = output.data_ptr<float>();
      const auto weight_data = weight.data_ptr<u8>();
      const auto indices_data = indices.data_ptr<IndexType>();
      i32* compressed_indices_mapping_data = nullptr;
      const auto weight_sizes = weight.sizes();
      const i64 N = weight_sizes[0];
      const i64 weight_size = weight_sizes[1];
      const int index_size = indices.numel();

      auto accessor = offsets.accessor<OffsetType, 1>();
      vector<OffsetType> lengths_data;

      i64 lower = accessor[0];
      for (i64 i = 1; i < offsets.numel(); ++i) {
        lengths_data.push_back(accessor[i] - lower);
        lower = accessor[i];
      }
      if (!include_last_offset) {
        lengths_data.push_back(indices.numel() - lower);
      }

      i64 current = 0;
      float* per_sample_weights_data;
      if (per_sample_weights_.has_value()) {
        per_sample_weights_data = per_sample_weights_.value().data_ptr<float>();
      }
      for (int m = 0; m < output_size; ++m) {
        memset(output_data, 0, block_size * sizeof(float));
        TORCH_CHECK(
            current + lengths_data[m] <= index_size,
            "Expect the lengths data to be less than indices size");

        for (int i = 0; i < lengths_data[m]; ++i, ++current) {
          i64 idx;
          if (!pruned) {
            idx = indices_data[current];
            TORCH_CHECK((idx >= 0 && idx < N), "Invalid indices data");
          } else {
            i64 uncompressed_idx = indices_data[current];
            int compressed_index_size = compressed_indices_mapping.value().numel();
            compressed_indices_mapping_data =
                compressed_indices_mapping.value().data_ptr<i32>();
            TORCH_CHECK(
                uncompressed_idx >= 0 && uncompressed_idx < compressed_index_size,
                "Invalid indices data for Sparse Op.")
            idx = compressed_indices_mapping_data[uncompressed_idx];
            if (idx == -1) {
              continue;
            }
          }

          float weight_val = 1.0f;
          if (per_sample_weights_.has_value()) {
            weight_val = per_sample_weights_data[current];
          }
          float scale, bias;
          if (BIT_RATE == 8) {
            const u8* scale_bias =
                weight_data + (idx + 1) * weight_size - 2 * sizeof(float);
            u32 scale_val_int32 = 0;
            scale_val_int32 = scale_val_int32 |
              (scale_bias[0]) |
              (scale_bias[1] << 8) |
              (scale_bias[2] << 16) |
              (scale_bias[3] << 24);
            float scale_val = (reinterpret_cast<float*>(&scale_val_int32))[0];
            u32 bias_val_int32 = 0;
            bias_val_int32 = bias_val_int32 |
              (scale_bias[4]) |
              (scale_bias[5] << 8) |
              (scale_bias[6] << 16) |
              (scale_bias[7] << 24);
            float bias_val = (reinterpret_cast<float*>(&bias_val_int32))[0];
            scale = weight_val * scale_val;
            bias = weight_val * bias_val;
          } else {
            const u8* scale_bias =
                weight_data + (idx + 1) * weight_size - 2 * sizeof(Half);
            u16 scale_val_int16 = 0;
            scale_val_int16 = scale_val_int16 |
              (scale_bias[0]) |
              (scale_bias[1] << 8);
            Half scale_val = (reinterpret_cast<Half*>(&scale_val_int16))[0];
            u16 bias_val_int16 = 0;
            bias_val_int16 = bias_val_int16 |
              (scale_bias[2]) |
              (scale_bias[3] << 8);
            Half bias_val = (reinterpret_cast<Half*>(&bias_val_int16))[0];
            scale = weight_val * scale_val;
            bias = weight_val * bias_val;
          }

          for (int j = 0; j < block_size; ++j) {
            u8 quantized =
                weight_data[idx * weight_size + j / NUM_ELEM_PER_BYTE];
            quantized >>= (j % NUM_ELEM_PER_BYTE) * BIT_RATE;
            quantized &= (1 << BIT_RATE) - 1;

            output_data[j] = fma(scale, quantized, output_data[j] + bias);
          }
        } // for each i
        output_data += block_size;
      } // for each m
      return output;
        */
}

pub fn embedding_bag_4bit_impl<'a, IndexType, OffsetType>(
    output:                     &mut Tensor,
    weight:                     &Tensor,
    indices:                    &Tensor,
    offsets:                    &Tensor,
    pruned_weights:             bool,
    per_sample_weights:         &Option<Tensor>,
    compressed_indices_mapping: &Option<Tensor>,
    include_last_offset:        bool) -> &'a mut Tensor {

    todo!();
        /*
            TORCH_CHECK(weight.dim() == 2);
      TORCH_CHECK(offsets.dim() == 1);

      const auto weight_data = weight.data_ptr<u8>();
      const auto indices_data = indices.data_ptr<IndexType>();
      auto offsets_data = offsets.data_ptr<OffsetType>();

      // Get compressed indices for pruned_weights op.
      i32* compressed_indices_mapping_data = nullptr;
      int compressed_index_size = 0;
      bool fallback_to_no_sparse = false;
      if (pruned_weights) {
        compressed_index_size = compressed_indices_mapping.value().numel();
        compressed_indices_mapping_data =
            compressed_indices_mapping.value().data_ptr<i32>();

        // if compressed_indices_mapping is [0], it is a indicator that
        // we should fallback to non sparse embedding look up kernel.
        if ((compressed_index_size == 1 &&
             compressed_indices_mapping_data[0] == 0)) {
          fallback_to_no_sparse = true;
        }
      }

      const auto weight_sizes = weight.sizes();
      const i64 N = weight_sizes[0];
      const i64 weight_size = weight_sizes[1];
      const i64 D =
          (weight_size - 4) * 2; // NB: 2-byte fp16 scale and 2-byte zero_offset
      const i64 M = offsets.sizes()[0];

      i64 output_size = M - 1;
      vector<OffsetType> offsets_include_last_val;
      if (!include_last_offset) {
        output_size = M;
        offsets_include_last_val.resize(M + 1);
        // Avoid `null pointer passed as argument 2` ASAN violation when offsets
        // tensor is empty.
        if (M > 0) {
          memcpy(
              offsets_include_last_val.data(),
              offsets_data,
              sizeof(OffsetType) * M);
        }
        offsets_include_last_val[M] = indices.numel();
        offsets_data = offsets_include_last_val.data();
      }

      const vector<i64> shape = {output_size, D};
      native::resize_(output, shape, nullopt);
      auto* output_data = output.data_ptr<float>();

      const i64 block_size = D;
      const int index_size = indices.numel();
      constexpr int prefetch_distance = 16;

    #ifdef USE_FBGEMM
      if (!pruned_weights || fallback_to_no_sparse) {
        // Generate the fbgemm kernel
        auto kernel = fbgemm::GenerateEmbeddingSpMDMNBit<IndexType, OffsetType>(
            /*bit rate=*/4,
            /*block size=*/block_size,
            /*has weights=*/per_sample_weights_.has_value(),
            /*normalize_by_lengths=*/false,
            /*prefetch distance=*/prefetch_distance,
            /*is_weight_positional=*/false,
            /*use_offsets=*/true);

        bool success = kernel(
            /*output_size=*/output_size,
            /*index_size=*/index_size,
            /*data_size=*/N,
            /*input=*/weight_data,
            /*indices=*/indices_data,
            /*offsets=*/offsets_data,
            /*weights=*/
            per_sample_weights_.has_value()
                ? per_sample_weights_.value().data_ptr<float>()
                : nullptr,
            /*output=*/output_data);

        TORCH_CHECK(
            success,
            "FBGEMM GenerateEmbeddingSpMDMNBit kernel failed for 4-bit input");
      } else {
        auto kernel =
            fbgemm::GenerateEmbeddingSpMDMNBitRowWiseSparse<IndexType, OffsetType>(
                /*bit rate=*/4,
                /*block_size=*/block_size,
                /*has weights=*/per_sample_weights_.has_value(),
                /*normalize_by_lengths=*/false,
                /*prefetch distance*/ prefetch_distance,
                /*is_weight_positional*/ false,
                /*use_offsets*/ true);
        bool success = kernel(
            /*output_size=*/output_size,
            /*index_size=*/index_size,
            /*data_size=*/compressed_index_size,
            /*input=*/weight_data,
            /*indices=*/indices_data,
            /*offsets=*/offsets_data,
            /*weights=*/
            per_sample_weights_.has_value()
                ? per_sample_weights_.value().data_ptr<float>()
                : nullptr,
            /*output=*/output_data,
            /*compressed_indices_table=*/compressed_indices_mapping_data);
        TORCH_CHECK(
            success,
            "FBGEMM GenerateEmbeddingSpMDMNBitRowWiseSparse kernel failed for 4-bit input");
      }
      return output;
    #else
      return embedding_lookup_fallback_impl<IndexType, OffsetType, 4, 2>(
          weight,
          indices,
          offsets,
          per_sample_weights_,
          compressed_indices_mapping,
          output,
          D,
          output_size,
          include_last_offset,
          (pruned_weights && !fallback_to_no_sparse));
    #endif
        */
}

pub fn embedding_bag_byte_impl<'a, IndexType, OffsetType>(
        output:                     &mut Tensor,
        weight:                     &Tensor,
        indices:                    &Tensor,
        offsets:                    &Tensor,
        pruned_weights:             bool,
        per_sample_weights:         &Option<Tensor>,
        compressed_indices_mapping: &Option<Tensor>,
        include_last_offset:        bool,
        is_embedding_op:            bool) -> &'a mut Tensor {

    todo!();
        /*
            TORCH_CHECK(weight.scalar_type() == kByte);
      TORCH_CHECK(weight.dim() == 2);
      TORCH_CHECK(offsets.dim() == 1);
      const auto weight_data = weight.data_ptr<u8>();
      const auto indices_data = indices.data_ptr<IndexType>();
      auto offsets_data = offsets.data_ptr<OffsetType>();

      // Get compressed indices for pruned_weights.
      i32* compressed_indices_mapping_data = nullptr;
      int compressed_index_size = 0;
      bool fallback_to_no_sparse = false;
      if (pruned_weights) {
        compressed_index_size = compressed_indices_mapping.value().numel();
        compressed_indices_mapping_data =
            compressed_indices_mapping.value().data_ptr<i32>();

        // if compressed_indices_mapping is [0], it is a indicator that
        // we should fallback to non sparse embedding look up kernel.
        if ((compressed_index_size == 1 &&
             compressed_indices_mapping_data[0] == 0)) {
          fallback_to_no_sparse = true;
        }
      }

      const auto weight_sizes = weight.sizes();
      const i64 N = weight_sizes[0];
      const i64 D = weight_sizes[1] - 8; // NB: -8 to account for scale and bias
      const i64 M = offsets.sizes()[0];

      i64 output_size = M - 1;
      vector<OffsetType> offsets_include_last_val;

      if (!include_last_offset) {
        output_size = M;
        offsets_include_last_val.resize(M + 1);
        // Avoid `null pointer passed as argument 2` ASAN violation when offsets
        // tensor is empty.
        if (M > 0) {
          memcpy(
              offsets_include_last_val.data(),
              offsets_data,
              sizeof(OffsetType) * M);
        }
        offsets_include_last_val[M] = indices.numel();
        offsets_data = offsets_include_last_val.data();
      }
      vector<i64> shape;
      if (indices.dim() == 2 && is_embedding_op) {
        const auto indices_sizes = indices.sizes();
        shape = {indices_sizes[0], indices_sizes[1], D};
      } else {
        shape = {output_size, D};
      }
      native::resize_(output, shape, nullopt);
      auto* output_data = output.data_ptr<float>();

      const int index_size = indices.numel();
    #ifdef USE_FBGEMM
      if (!pruned_weights || fallback_to_no_sparse) {
        auto kernel_i8 =
            fbgemm::GenerateEmbeddingSpMDM<u8, IndexType, OffsetType>(
                /*block_size=*/D,
                /*has_weight=*/per_sample_weights_.has_value(),
                /*normalize_by_lengths=*/false,
                /*prefetch=*/16, // NOLINT(cppcoreguidelines-avoid-magic-numbers)
                /*is_weight_positional=*/false,
                /*use_offsets=*/true);

        parallel_for(
            0, output_size, 1, [&](i64 start_idx, i64 end_idx) {
              bool success = kernel_i8(
                  /*output_size=*/end_idx - start_idx,
                  /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
                  /*data_size=*/N,
                  /*input=*/weight_data,
                  /*indices=*/indices_data + offsets_data[start_idx],
                  /*offsets_or_lengths=*/offsets_data + start_idx,
                  /*weights=*/
                  per_sample_weights_
                      ? per_sample_weights_.value().data_ptr<float>() +
                          offsets_data[start_idx]
                      : nullptr,
                  /*out=*/output_data + start_idx * D);

              TORCH_CHECK(
                  success,
                  "FBGEMM GenerateEmbeddingSpMDM kernel failed for 8-bit input");
            });
      } else {
        // pruned weights
        auto kernel_i8_sparse = fbgemm::
            GenerateEmbeddingSpMDMRowWiseSparse<u8, IndexType, OffsetType>(
                /*block_size=*/D,
                /*has_weight=*/per_sample_weights_.has_value(),
                /*normalize_by_lengths=*/false,
                /*prefetch=*/16, // NOLINT(cppcoreguidelines-avoid-magic-numbers)
                /*is_weight_positional=*/false,
                /*use_offsets=*/true);

        auto success = kernel_i8_sparse(
            /*output_size=*/output_size,
            /*index_size=*/index_size,
            /*data_size=*/compressed_index_size,
            /*input=*/weight_data,
            /*indices=*/indices_data,
            /*offsets=*/offsets_data,
            /*weights=*/
            per_sample_weights_.has_value()
                ? per_sample_weights_.value().data_ptr<float>()
                : nullptr,
            /*output=*/output_data,
            /*compressed_indices_table=*/compressed_indices_mapping_data);
        TORCH_CHECK(
            success,
            "FBGEMM GenerateEmbeddingSpMDMRowWiseSparse kernel failed for 8-bit input");
      }
      return output;
    #else
      return embedding_lookup_fallback_impl<IndexType, OffsetType, 8, 1>(
          weight,
          indices,
          offsets,
          per_sample_weights_,
          compressed_indices_mapping,
          output,
          D,
          output_size,
          include_last_offset,
          (pruned_weights && !fallback_to_no_sparse));
    #endif
        */
}

pub fn embedding_bag_byte_helper<'a>(
        output:                     &mut Tensor,
        weight:                     &Tensor,
        indices:                    &Tensor,
        offsets_in:                 &Option<Tensor>,
        pruned_weights:             bool,
        per_sample_weights:         &Option<Tensor>,
        compressed_indices_mapping: &Option<Tensor>,
        include_last_offset:        bool,
        is_embedding_op:            bool) -> &'a mut Tensor {
    
    todo!();
        /*
            MaybeOwned<Tensor> offsets;
      TORCH_CHECK(
          indices.dim() == 1 || indices.dim() == 2,
          "qembedding/qembedding_bag operator supports 1 or 2d indices, got ",
          indices.dim());
      // For embedding_bag operator with 2D indices, we set the offsets explicitly
      // here.
      if (indices.dim() == 2 && !is_embedding_op) {
        TORCH_CHECK(
            !offsets_in.has_value(),
            "embedding_bag_byte operator: input is 2D, then offsets has to be None, as input is treated is a mini-batch of fixed length sequences.");

        offsets = MaybeOwned<Tensor>::owned(arange(0, indices.numel(), indices.sizes()[1], indices.scalar_type()));

      } else {
        TORCH_CHECK(
            offsets_in.has_value(),
            "embedding_bag_byte expects offsets to be set for 1D indices.");
        offsets = MaybeOwned<Tensor>::borrowed(offsets_in.value());
      }

      TORCH_CHECK(
          indices.scalar_type() == kInt || indices.scalar_type() == kLong,
          "Expect 32 or 64 bit indices, but found ",
          indices.scalar_type(),
          " instead.");
      TORCH_CHECK(
          offsets->scalar_type() == kInt || offsets->scalar_type() == kLong,
          "Expect 32 or 64 bit offsets, but found ",
          offsets->scalar_type(),
          " instead.");
      TORCH_CHECK(
          weight.is_contiguous() && indices.is_contiguous() &&
              offsets->is_contiguous(),
          "Expect weight, indices, and offsets to be contiguous.");

      // Using helper function to support different type combination without the
      // need to cast, which can be additional performance overhead
      if (indices.scalar_type() == kInt && offsets->scalar_type() == kInt) {
        return embedding_bag_byte_impl<int, int>(
            output,
            weight,
            indices,
            *offsets,
            pruned_weights,
            per_sample_weights_,
            compressed_indices_mapping,
            include_last_offset,
            is_embedding_op);
      } else if (
          indices.scalar_type() == kInt && offsets->scalar_type() == kLong) {
        return embedding_bag_byte_impl<int, i64>(
            output,
            weight,
            indices,
            *offsets,
            pruned_weights,
            per_sample_weights_,
            compressed_indices_mapping,
            include_last_offset,
            is_embedding_op);
      } else if (
          indices.scalar_type() == kLong && offsets->scalar_type() == kInt) {
        return embedding_bag_byte_impl<i64, int>(
            output,
            weight,
            indices,
            *offsets,
            pruned_weights,
            per_sample_weights_,
            compressed_indices_mapping,
            include_last_offset,
            is_embedding_op);
      }

      // default case given the TORCH_CHECK above
      return embedding_bag_byte_impl<i64, i64>(
          output,
          weight,
          indices,
          *offsets,
          pruned_weights,
          per_sample_weights_,
          compressed_indices_mapping,
          include_last_offset,
          is_embedding_op);
        */
}

pub fn embedding_bag_4bit_helper<'a>(
        output:                     &mut Tensor,
        weight:                     &Tensor,
        indices:                    &Tensor,
        offsets_in:                 &Option<Tensor>,
        pruned_weights:             bool,
        per_sample_weights:         &Option<Tensor>,
        compressed_indices_mapping: &Option<Tensor>,
        include_last_offset:        bool) -> &'a mut Tensor {
    
    todo!();
        /*
            MaybeOwned<Tensor> offsets;
      TORCH_CHECK(
          indices.dim() == 1 || indices.dim() == 2,
          "qembedding/qembedding_bag operator supports 1 or 2d indices, got ",
          indices.dim());

      // For embedding_bag operator with 2D indices, we need to set the offsets
      // explicitly here.
      if (indices.dim() == 2) {
        TORCH_CHECK(
            !offsets_in.has_value(),
            "embedding_bag_4bit operator: input is 2D, then offsets has to be None, as input is treated is a mini-batch of fixed length sequences.");

        offsets = MaybeOwned<Tensor>::owned(arange(
            0, indices.numel(), indices.sizes()[1], indices.scalar_type()));
      } else {
        TORCH_CHECK(
            offsets_in.has_value(),
            "embedding_bag_4bit operator expects offsets to be set for 1D indices.");
        offsets = MaybeOwned<Tensor>::borrowed(offsets_in.value());
      }

      TORCH_CHECK(
          indices.scalar_type() == kInt || indices.scalar_type() == kLong,
          "Expect 32 or 64 bit indices, but found ",
          indices.scalar_type(),
          " instead.");
      TORCH_CHECK(
          offsets->scalar_type() == kInt || offsets->scalar_type() == kLong,
          "Expect 32 or 64 bit offsets, but found ",
          offsets->scalar_type(),
          " instead.");
      TORCH_CHECK(
          weight.is_contiguous() && indices.is_contiguous() &&
              offsets->is_contiguous(),
          "Expect weight, indices, and offsets to be contiguous.");

      // Using helper function to support different type combination without the
      // need to cast, which can be additional performance overhead
      if (indices.scalar_type() == kInt && offsets->scalar_type() == kInt) {
        return embedding_bag_4bit_impl<int, int>(
            output,
            weight,
            indices,
            *offsets,
            pruned_weights,
            per_sample_weights_,
            compressed_indices_mapping,
            include_last_offset);
      } else if (
          indices.scalar_type() == kInt && offsets->scalar_type() == kLong) {
        return embedding_bag_4bit_impl<int, i64>(
            output,
            weight,
            indices,
            *offsets,
            pruned_weights,
            per_sample_weights_,
            compressed_indices_mapping,
            include_last_offset);
      } else if (
          indices.scalar_type() == kLong && offsets->scalar_type() == kInt) {
        return embedding_bag_4bit_impl<i64, int>(
            output,
            weight,
            indices,
            *offsets,
            pruned_weights,
            per_sample_weights_,
            compressed_indices_mapping,
            include_last_offset);
      }
      return embedding_bag_4bit_impl<i64, i64>(
          output,
          weight,
          indices,
          *offsets,
          pruned_weights,
          per_sample_weights_,
          compressed_indices_mapping,
          include_last_offset);
        */
}

impl PackedEmbeddingBagWeight {
    
    pub fn embeddingbag_byte(&mut self, 
        indices:                    &Tensor,
        offsets_in:                 &Option<Tensor>,
        pruned_weights:             bool,
        per_sample_weights:         &Option<Tensor>,
        compressed_indices_mapping: &Option<Tensor>,
        include_last_offset:        bool,
        is_embedding_op:            bool) -> Tensor {
        
        todo!();
        /*
            auto output = empty({0}, packed_w.options().dtype(kFloat));
      return embedding_bag_byte_helper(
          output,
          packed_w,
          indices,
          offsets_in,
          pruned_weights,
          per_sample_weights_,
          compressed_indices_mapping,
          include_last_offset,
          is_embedding_op);
        */
    }
    
    pub fn embeddingbag_4bit(&mut self, 
        indices:                    &Tensor,
        offsets_in:                 &Option<Tensor>,
        pruned_weights:             bool,
        per_sample_weights:         &Option<Tensor>,
        compressed_indices_mapping: &Option<Tensor>,
        include_last_offset:        bool) -> Tensor {
        
        todo!();
        /*
            if (per_sample_weights_.has_value()) {
        TORCH_CHECK(
            (per_sample_weights_.value().scalar_type() == kFloat ||
             per_sample_weights_.value().scalar_type() == kHalf),
            "Expect fp32 or fp16 weights, but found",
            per_sample_weights_.value().scalar_type(),
            " instead")
      }

      auto output = empty({0}, packed_w.options().dtype(kFloat));
      return embedding_bag_4bit_helper(
        output,
        packed_w,
        indices,
        offsets_in,
        pruned_weights,
        per_sample_weights_.has_value()
            ? per_sample_weights_.value().to(kFloat)
            : per_sample_weights_,
        compressed_indices_mapping,
        include_last_offset);
        */
    }
}

pub fn embedding_bag_byte_rowwise_offsets_out<'a>(
        output:                     &mut Tensor,
        weight:                     &Tensor,
        indices:                    &Tensor,
        offsets_in:                 &Option<Tensor>,
        scale_grad_by_freq:         bool,
        mode:                       i64,
        pruned_weights:             bool,
        per_sample_weights:         &Option<Tensor>,
        compressed_indices_mapping: &Option<Tensor>,
        include_last_offset:        bool) -> &'a mut Tensor {
    
    todo!();
        /*
            return embedding_bag_byte_helper(
          output,
          weight,
          indices,
          offsets_in,
          pruned_weights,
          per_sample_weights_,
          compressed_indices_mapping,
          include_last_offset,
          false /* is_embedding_op */);
        */
}

pub fn embedding_bag_4bit_rowwise_offsets_out<'a>(
    output:                     &mut Tensor,
    weight:                     &Tensor,
    indices:                    &Tensor,
    offsets_in:                 &Option<Tensor>,
    scale_grad_by_freq:         bool,
    mode:                       i64,
    pruned_weights:             bool,
    per_sample_weights:         &Option<Tensor>,
    compressed_indices_mapping: &Option<Tensor>,
    include_last_offset:        bool) -> &'a mut Tensor {

    todo!();
    /*
       if (per_sample_weights_.has_value()) {
        TORCH_CHECK(
            (per_sample_weights_.value().scalar_type() == kFloat ||
             per_sample_weights_.value().scalar_type() == kHalf),
            "Expect fp32 or fp16 weights, but found",
            per_sample_weights_.value().scalar_type(),
            " instead")
      }
      return embedding_bag_4bit_helper(
          output,
          weight,
          indices,
          offsets_in,
          pruned_weights,
          per_sample_weights_.has_value()
              ? per_sample_weights_.value().to(kFloat)
              : per_sample_weights_,
          compressed_indices_mapping,
          include_last_offset);
        */
}

#[inline] pub fn create_empty_from(
        t:     &Tensor,
        dtype: ScalarType) -> Tensor {
    
    todo!();
        /*
            return empty_cpu(
          {0}, dtype, t.layout(), t.device(), nullopt, nullopt);
        */
}

pub fn embedding_bag_byte_rowwise_offsets(
        weight:                     &Tensor,
        indices:                    &Tensor,
        offsets_in:                 &Option<Tensor>,
        scale_grad_by_freq:         bool,
        mode:                       i64,
        pruned_weights:             bool,
        per_sample_weights:         &Option<Tensor>,
        compressed_indices_mapping: &Option<Tensor>,
        include_last_offset:        bool) -> Tensor {
    
    todo!();
        /*
            auto output = create_empty_from(weight, kFloat);
      embedding_bag_byte_rowwise_offsets_out(
          output,
          weight,
          indices,
          offsets_in,
          false /*unused scale_grad_by_freq*/,
          0 /*unused mode*/,
          pruned_weights,
          per_sample_weights_,
          compressed_indices_mapping,
          include_last_offset);
      return output;
        */
}

pub fn embedding_bag_4bit_rowwise_offsets(
        weight:                     &Tensor,
        indices:                    &Tensor,
        offsets_in:                 &Option<Tensor>,
        scale_grad_by_freq:         bool,
        mode:                       i64,
        pruned_weights:             bool,
        per_sample_weights:         &Option<Tensor>,
        compressed_indices_mapping: &Option<Tensor>,
        include_last_offset:        bool) -> Tensor {
    
    todo!();
        /*
            auto output = create_empty_from(weight, kFloat);
      embedding_bag_4bit_rowwise_offsets_out(
        output,
        weight,
        indices,
        offsets_in,
        false, // unused scale_grad_by_freq
        0, // unused mode
        pruned_weights,
        per_sample_weights_,
        compressed_indices_mapping,
        include_last_offset
      );
      return output;
        */
}

pub struct QEmbeddingBag<const bit_rate: i32> {

}

impl<const bit_rate: i32> QEmbeddingBag<bit_rate> {
    
    pub fn run(
        packed_weight:              &IntrusivePtr<EmbeddingPackedParamsBase>,
        indices:                    &Tensor,
        offsets:                    &Option<Tensor>,
        scale_grad_by_freq:         bool,
        mode:                       i64,
        pruned_weights:             bool,
        per_sample_weights:         &Option<Tensor>,
        compressed_indices_mapping: &Option<Tensor>,
        include_last_offset:        bool) -> Tensor {
        
        todo!();
        /*
            if (bit_rate == 8) {
          return packed_weight->embeddingbag_byte(
              indices,
              offsets,
              pruned_weights,
              per_sample_weights_,
              compressed_indices_mapping,
              include_last_offset,
              false /* is_embedding_op */);
        } else if (bit_rate == 4) {
          return packed_weight->embeddingbag_4bit(
              indices,
              offsets,
              pruned_weights,
              per_sample_weights_,
              compressed_indices_mapping,
              include_last_offset);
        } else {
          TORCH_INTERNAL_ASSERT(
              "Currently only support 8-bit embedding_bag quantization");
        }
        */
    }
}

pub struct QEmbedding<const bit_rate: i32> {

}

impl<const bit_rate: i32> QEmbedding<bit_rate> {
    
    pub fn run(
        packed_weight:  &IntrusivePtr<EmbeddingPackedParamsBase>,
        indices:        &Tensor,
        pruned_weights: bool) -> Tensor {
        
        todo!();
        /*
            // Set default offsets here since the FBGEMM lookup op expects it.
        const auto offsets_size = indices.numel();
        Tensor offsets = arange(0, offsets_size, indices.scalar_type());
        Tensor output;
        if (bit_rate == 8) {
          return packed_weight->embeddingbag_byte(
              indices,
              offsets,
              pruned_weights,
              nullopt,
              nullopt,
              false /* include_last_offset */,
              true /* is_embedding_op */);

        } else {
          TORCH_INTERNAL_ASSERT(
              "Currently only support 8-bit embedding quantization");
        }
        return output;
        */
    }
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, CPU, m) {
      // Function that works on TorchBind packed weights.
      m.impl(
          TORCH_SELECTIVE_NAME("quantized::embedding_bag_byte"),
          TORCH_FN(QEmbeddingBag<8>::run));
      m.impl(
          TORCH_SELECTIVE_NAME("quantized::embedding_bag_4bit"),
          TORCH_FN(QEmbeddingBag<4>::run));
      m.impl(
          TORCH_SELECTIVE_NAME("quantized::embedding_byte"),
          TORCH_FN(QEmbedding<8>::run));

      // Functions that work on Tensor packed weight.
      m.impl(
          TORCH_SELECTIVE_NAME("quantized::embedding_bag_byte_rowwise_offsets"),
          embedding_bag_byte_rowwise_offsets);
      m.impl(
          TORCH_SELECTIVE_NAME("quantized::embedding_bag_4bit_rowwise_offsets"),
          embedding_bag_4bit_rowwise_offsets);
    }
    */
}
