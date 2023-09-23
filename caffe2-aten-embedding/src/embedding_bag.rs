crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/EmbeddingBag.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/EmbeddingBag.cpp]

pub const MODE_SUM:  i32 = 0;
pub const MODE_MEAN: i32 = 1;
pub const MODE_MAX:  i32 = 2;

pub fn make_offset2bag_a(
        offsets:    &Tensor,
        offset2bag: &mut Tensor)  {
    
    todo!();
        /*
            offset2bag.index_add_(
          0, offsets, ones_like(offsets, LEGACY_CONTIGUOUS_MEMORY_FORMAT)); // offset2bag = [1 0 1 0 1]
      offset2bag[0] -= 1;                     // offset2bag = [0 0 1 0 1]
      offset2bag = offset2bag.cumsum(0, offset2bag.scalar_type());     // offset2bag = [0 0 1 1 2]
        */
}

pub fn promote_indices_and_offsets(
        indices: &Tensor,
        offsets: &Tensor) -> (Tensor,Tensor) {
    
    todo!();
        /*
            const auto commonType =
          promoteTypes(offsets.scalar_type(), indices.scalar_type());
      return {
          indices.scalar_type() == commonType ? indices
                                              : indices.toType(commonType),
          offsets.scalar_type() == commonType ? offsets
                                              : offsets.toType(commonType)};
        */
}

/**
  | Determines if we can use a fast implementation
  | for index_select_add, which is only
  | applicable if special conditions are
  | met
  |
  */
pub fn is_fast_path_index_select<Index>(
        src:         &Tensor,
        output:      &mut Tensor,
        padding_idx: Index) -> bool {

    todo!();
        /*
            return src.scalar_type() == kFloat && src.strides()[1] == 1 && output.strides()[1] == 1 && padding_idx < static_cast<Index>(0);
        */
}

/**
  | Determines if we can use a fast implementation
  | for index_select_scale_add, which is only
  | applicable if special conditions are met
  |
  */
pub fn is_fast_path_index_select_scale<Index>(
        src:         &Tensor,
        scale:       &Tensor,
        output:      &mut Tensor,
        padding_idx: Index) -> bool {

    todo!();
        /*
            return src.scalar_type() == kFloat && src.strides()[1] == 1 && output.strides()[1] == 1 && scale.strides()[0] == 1 && padding_idx < static_cast<Index>(0);
        */
}

pub fn is_fast_path<Index>(
        src:         &Tensor,
        scale:       &Option<Tensor>,
        output:      &mut Tensor,
        padding_idx: Index) -> bool {

    todo!();
        /*
            return (scale.has_value() && scale.value().defined()) ?
             is_fast_path_index_select_scale(src, scale.value(), output, padding_idx) :
             is_fast_path_index_select(src, output, padding_idx);
        */
}

/**
  | This function combines index_select (using
  | select_indices as the index) and index_add
  | (using add_indices as the index), without
  | creating an intermediary tensor to hold the
  | selected embeddings
  |
  */
//typename enable_if<!is_same<Data, float>::value, void>::type
pub fn index_select_add<Data, Index>(
    select_indices:      &Tensor,
    add_indices:         &Tensor,
    src:                 &Tensor,
    output:              &mut Tensor,
    offsets:             &Tensor,
    include_last_offset: bool,
    bag_size:            &mut Tensor,
    padding_idx:         Index)  {

    todo!();
        /*
            TORCH_CHECK(select_indices.numel() == add_indices.numel());
      auto* add_indices_data = add_indices.data_ptr<Index>();
      auto* select_indices_data = select_indices.data_ptr<Index>();
      auto* src_data = src.data_ptr<Data>();
      auto* output_data = output.data_ptr<Data>();
      Index* bag_size_data;
      if (bag_size.defined()) {
        bag_size_data = bag_size.data_ptr<Index>();
      }
      auto numel = add_indices.numel();
      i64 ddim = src.sizes()[1];
      auto src_stride0 = src.strides()[0];
      auto src_stride1 = src.strides()[1];
      auto output_stride0 = output.strides()[0];
      auto output_stride1 = output.strides()[1];

      for (i64 i = 0; i < numel; i++) {
        // We can skip indices equal to padding_idx so they are not included in
        // the reduction
        if (select_indices_data[i] != padding_idx) {
          native::cpublas::axpy<Data>(ddim, 1,
                  src_data + src_stride0 * select_indices_data[i], src_stride1,
                  output_data + output_stride0 * add_indices_data[i], output_stride1);
        } else if (bag_size.defined()) {
          // Decrement bag_size to reflect that the index is padded
          bag_size_data[add_indices_data[i]]--;
        }
      }
        */
}

pub fn index_select_add_float<Data: Float, Index>(
    select_indices:      &Tensor,
    add_indices:         &Tensor,
    src:                 &Tensor,
    output:              &mut Tensor,
    offsets:             &Tensor,
    include_last_offset: bool,
    bag_size:            &mut Tensor,
    padding_idx:         Index)  {

    todo!();
        /*
            i64 ddim = src.sizes()[1];
      auto* select_indices_data = select_indices.data_ptr<Index>();
      auto* output_data = output.data_ptr<float>();

      if (is_fast_path_index_select(src, output, padding_idx)) {
        auto src_contig = src.contiguous();
        auto* src_data = src_contig.data_ptr<float>();
        i64 output_size = offsets.numel() - 1;
        auto* offsets_data = offsets.data_ptr<Index>();
        vector<Index> offsets_include_last;

        if (include_last_offset) {
          output_size = offsets.numel() - 1;
        } else {
          output_size = offsets.numel();
          offsets_include_last.resize(offsets.numel() + 1);
          if (offsets.numel() > 0) {
            memcpy(
                offsets_include_last.data(),
                offsets.data_ptr<Index>(),
                sizeof(Index) * offsets.numel());
          }
          offsets_include_last[offsets.numel()] = select_indices.numel();
          offsets_data = offsets_include_last.data();
        }

    #ifdef USE_FBGEMM
        auto kernel_fp32_index_t =
          fbgemm::GenerateEmbeddingSpMDM<float, Index, Index>(
            /* block_size */ddim,
            /* has_weight */false,
            /* normalize_by_lengths */false,
            /* prefetch */16,
            /* is_weight_positional */false,
            /* use_offsets */true
          );
    #endif
        parallel_for(
            0, output_size, 1, [&](Index start_idx, Index end_idx) {
    #ifdef USE_FBGEMM
              kernel_fp32_index_t(
                /* output_size */end_idx - start_idx,
                /* index_size */offsets_data[end_idx] - offsets_data[start_idx],
                /* data_size */src.sizes()[0],
                /* input */src_data,
                /* indices */select_indices_data + offsets_data[start_idx],
                /* offsets_or_lengths */offsets_data + start_idx,
                /* weights */nullptr,
                /* output */output_data + start_idx * ddim);
    #else
              EmbeddingLookupIdx(
                  /*block_size=*/ddim,
                  /*output_size=*/end_idx - start_idx,
                  /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
                  /*data_size=*/src.sizes()[0],
                  /*input=*/src_data,
                  /*indices=*/select_indices_data + offsets_data[start_idx],
                  /*offsets=*/offsets_data + start_idx,
                  /*weights=*/nullptr,
                  /*scale_bias=*/nullptr,
                  /*normalize_by_lengths=*/false,
                  /*out=*/output_data + start_idx * ddim);
    #endif
            });
      } else {
        AT_ASSERT(select_indices.numel() == add_indices.numel());
        auto* src_data = src.data_ptr<float>();
        auto* add_indices_data = add_indices.data_ptr<Index>();
        Index* bag_size_data;
        if (bag_size.defined()) {
          bag_size_data = bag_size.data_ptr<Index>();
        }
        auto src_stride0 = src.strides()[0];
        auto src_stride1 = src.strides()[1];
        auto output_stride0 = output.strides()[0];
        auto output_stride1 = output.strides()[1];
        auto numel = add_indices.numel();
        for (i64 i = 0; i < numel; i++) {
          // We can skip indices equal to padding_idx so they are not included in
          // the reduction
          if (select_indices_data[i] != padding_idx) {
            native::cpublas::axpy<float>(
                ddim,
                1,
                src_data + src_stride0 * select_indices_data[i],
                src_stride1,
                output_data + output_stride0 * add_indices_data[i],
                output_stride1);
          } else if (bag_size.defined()) {
            // Decrement bag_size to reflect that the index is padded
            bag_size_data[add_indices_data[i]]--;
          }
        }
      }
        */
}

/**
  | This function fuses the following three fns:
  | index_select (using select_indices as the
  | index) mul (scaling by per_sample_weights)
  | index_add (using add_indices as the index)
  |
  |typename enable_if<!is_same<Data, float>::value, void>::type
  */
pub fn index_select_scale_add<Data, Index>(
        select_indices:      &Tensor,
        add_indices:         &Tensor,
        scale:               &Tensor,
        src:                 &Tensor,
        output:              &mut Tensor,
        offsets:             &Tensor,
        include_last_offset: bool,
        bag_size:            &mut Tensor,
        padding_idx:         Index)  {

    todo!();
        /*
            AT_ASSERT(select_indices.numel() == add_indices.numel());
      auto* add_indices_data = add_indices.data_ptr<Index>();
      auto* select_indices_data = select_indices.data_ptr<Index>();
      auto* src_data = src.data_ptr<Data>();
      auto* output_data = output.data_ptr<Data>();
      Index* bag_size_data;
      if (bag_size.defined()) {
        bag_size_data = bag_size.data_ptr<Index>();
      }
      auto numel = add_indices.numel();
      i64 ddim = src.sizes()[1];
      auto src_stride0 = src.strides()[0];
      auto src_stride1 = src.strides()[1];
      auto output_stride0 = output.strides()[0];
      auto output_stride1 = output.strides()[1];

      auto* scale_data = scale.data_ptr<Data>();
      auto scale_stride = scale.strides()[0];

      for (i64 i = 0; i < numel; i++) {
        // We can skip indices equal to padding_idx so they are not included in
        // the reduction
        if (select_indices_data[i] != padding_idx) {
          auto* src_base = src_data + src_stride0 * select_indices_data[i];
          auto* output_base = output_data + output_stride0 * add_indices_data[i];
          auto scale = scale_data[i * scale_stride];
          for (i64 j = 0; j < ddim; j++) {
            output_base[j * output_stride1] += src_base[j * src_stride1] * scale;
          }
        } else if (bag_size.defined()) {
          // Decrement bag_size to reflect that the index is padded
          bag_size_data[add_indices_data[i]]--;
        }
      }
        */
}

pub fn index_select_scale_add_float<Data: Float, Index>(
    select_indices:      &Tensor,
    add_indices:         &Tensor,
    scale:               &Tensor,
    src:                 &Tensor,
    output:              &mut Tensor,
    offsets:             &Tensor,
    include_last_offset: bool,
    bag_size:            &mut Tensor,
    padding_idx:         Index)  {

    todo!();
        /*
            i64 ddim = src.sizes()[1];
      auto* scale_data = scale.data_ptr<float>();
      auto* select_indices_data = select_indices.data_ptr<Index>();
      auto* output_data = output.data_ptr<float>();

      if (is_fast_path_index_select_scale(src, scale, output, padding_idx)) {
        auto src_contig = src.contiguous();
        auto* src_data = src_contig.data_ptr<float>();
        i64 output_size = offsets.numel() - 1;
        auto* offsets_data = offsets.data_ptr<Index>();
        vector<Index> offsets_include_last;

        if (include_last_offset) {
          output_size = offsets.numel() - 1;
        } else {
          output_size = offsets.numel();
          offsets_include_last.resize(offsets.numel() + 1);
          memcpy(
              offsets_include_last.data(),
              offsets.data_ptr<Index>(),
              sizeof(Index) * offsets.numel());
          offsets_include_last[offsets.numel()] = select_indices.numel();
          offsets_data = offsets_include_last.data();
        }

    #ifdef USE_FBGEMM
        auto kernel_fp32_index_t =
          fbgemm::GenerateEmbeddingSpMDM<float, Index, Index>(
            /* block_size */ddim,
            /* has_weight */true,
            /* normalize_by_lengths */false,
            /* prefetch */16,
            /* is_weight_positional */false,
            /* use_offsets */true
          );
    #endif
        parallel_for(
            0, output_size, 1, [&](Index start_idx, Index end_idx) {
    #ifdef USE_FBGEMM
              kernel_fp32_index_t(
                /* output_size */end_idx - start_idx,
                /* index_size */offsets_data[end_idx] - offsets_data[start_idx],
                /* data_size */src.sizes()[0],
                /* input */src_data,
                /* indices */select_indices_data + offsets_data[start_idx],
                /* offsets_or_lengths */offsets_data + start_idx,
                /* weights */scale_data + offsets_data[start_idx],
                /* output */output_data + start_idx * ddim);
    #else
              EmbeddingLookupIdx(
                  /*block_size=*/ddim,
                  /*output_size=*/end_idx - start_idx,
                  /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
                  /*data_size=*/src.sizes()[0],
                  /*input=*/src_data,
                  /*indices=*/select_indices_data + offsets_data[start_idx],
                  /*offsets=*/offsets_data + start_idx,
                  /*weights=*/scale_data + offsets_data[start_idx],
                  /*scale_bias=*/nullptr,
                  /*normalize_by_lengths=*/false,
                  /*out=*/output_data + start_idx * ddim);
    #endif
            });
      } else {
        AT_ASSERT(select_indices.numel() == add_indices.numel());
        auto* src_data = src.data_ptr<float>();
        auto* add_indices_data = add_indices.data_ptr<Index>();
        Index* bag_size_data;
        if (bag_size.defined()) {
          bag_size_data = bag_size.data_ptr<Index>();
        }
        auto src_stride0 = src.strides()[0];
        auto src_stride1 = src.strides()[1];
        auto output_stride0 = output.strides()[0];
        auto output_stride1 = output.strides()[1];
        auto scale_stride = scale.strides()[0];
        auto numel = add_indices.numel();

        for (i64 i = 0; i < numel; i++) {
          // We can skip indices equal to padding_idx so they are not included in
          // the reduction
          if (select_indices_data[i] != padding_idx) {
            auto* src_base = src_data + src_stride0 * select_indices_data[i];
            auto* output_base = output_data + output_stride0 * add_indices_data[i];
            auto scale = scale_data[i * scale_stride];
            for (i64 j = 0; j < ddim; j++) {
              output_base[j * output_stride1] += src_base[j * src_stride1] * scale;
            }
          } else if (bag_size.defined()) {
            // Decrement bag_size to reflect that the index is padded
            bag_size_data[add_indices_data[i]]--;
          }
        }
      }
        */
}

pub fn check_arguments(
        weight:              &Tensor,
        indices:             &Tensor,
        offsets:             &Tensor,
        mode:                i64,
        per_sample_weights:  &Option<Tensor>,
        include_last_offset: bool)  {
    
    todo!();
        /*
            auto indices_arg = TensorArg(indices, "indices", 1);
      checkScalarTypes("embedding_bag", indices_arg, {kLong, kInt});
      auto offsets_arg = TensorArg(offsets, "offsets", 1);
      checkScalarTypes("embedding_bag", offsets_arg, {kLong, kInt});
      checkSameType("embedding_bag", indices_arg, offsets_arg);
      auto weight_arg = TensorArg(weight, "weight", 1);
      checkScalarTypes("embedding_bag", weight_arg, {kFloat, kDouble});

      AT_DISPATCH_INDEX_TYPES(offsets.scalar_type(), "_embedding_bag_cpu_impl", [&]() {
        if (offsets.sizes()[0] > 0) {
          Index offset_0 = offsets.data_ptr<Index>()[0];
          Index offset_n = offsets.data_ptr<Index>()[offsets.sizes()[0]-1];
          TORCH_CHECK(offset_0 == 0, "offsets[0] has to be 0, i.e., the first sequence "
                                    "in the mini-batch has to start from position 0. "
                                    "However, got ", offsets[0]);
          TORCH_CHECK(offset_n <= indices.sizes()[0], "offsets[-1] can not "
                      "be greater than input's length ", indices.sizes()[0], " but got offsets[-1] of ",
                      offset_n);
        }
      });

      if (per_sample_weights.has_value() && per_sample_weights.value().defined()) {
        TORCH_CHECK(mode == MODE_SUM,
            "embedding_bag: per_sample_weights only supported with mode='sum'");
        auto per_input_weights_arg = TensorArg(
            per_sample_weights.value(),"per_sample_weights", 1);
        checkSameType("embedding_bag", weight_arg, per_input_weights_arg);
        TORCH_CHECK(per_sample_weights.value().dim() == 1);
        TORCH_CHECK(per_sample_weights.value().numel() == indices.numel());
      }

      if (include_last_offset) {
        TORCH_CHECK(
            offsets.sizes()[0] >= 1,
            "include_last_offset: number of offset should be at least 1");
      }
        */
}

pub fn make_bag_size_out(
        bag_size_out:        &mut Tensor,
        offsets:             &Tensor,
        indices:             &Tensor,
        mode:                i64,
        include_last_offset: bool,
        requires_grad:       bool)  {
    
    todo!();
        /*
            if (requires_grad || mode == MODE_MEAN || mode == MODE_MAX) {
        auto num_bags = offsets.size(0) - (include_last_offset ? 1 : 0);
        bag_size_out = zeros({num_bags}, offsets.options());
        // Compute this for MODE_MEAN and MODE_MAX (latter needed for backwards)
        if (num_bags != 1) {
          bag_size_out.slice(0, 0, bag_size_out.sizes()[0] - 1, 1) =
              offsets.slice(0, 1, num_bags, 1) -
              offsets.slice(0, 0, num_bags - 1, 1);
        }
        if (num_bags > 0) {
          bag_size_out[-1] = indices.sizes()[0] - offsets[num_bags - 1];
        }
      }
        */
}

pub fn make_max_indices_out(
        max_indices_out:     &mut Tensor,
        weight:              &Tensor,
        indices:             &Tensor,
        offsets:             &Tensor,
        bag_size:            &Tensor,
        mode:                i64,
        include_last_offset: bool)  {
    
    todo!();
        /*
            i64 numBags = offsets.sizes()[0];
      if (mode == MODE_MAX) {
        if (include_last_offset) {
          TORCH_CHECK(
            numBags >= 1, "include_last_offset: numBags should be at least 1");
          numBags -= 1;
        }
        native::resize_(max_indices_out, {numBags, weight.sizes()[1]}, nullopt);
        native::zero_(max_indices_out);
      } else {
          native::resize_(max_indices_out, bag_size.sizes(), nullopt);
      }
        */
}

pub fn make_offset2bag_out(
        offset2bag:         &mut Tensor,
        output:             &mut Tensor,
        weight:             &Tensor,
        indices:            &Tensor,
        offsets:            &Tensor,
        mode:               i64,
        per_sample_weights: &Option<Tensor>,
        padding_idx:        i64)  {

    let padding_idx: i64 = padding_idx.unwrap_or(-1);
    
    todo!();
        /*
            // To save compute, if we are going to go down the fast path case for the 'sum'
      // mode, we skip calculating offset2bag, since it is not going to be used.
      bool fast_path_sum = is_fast_path(weight, per_sample_weights, output, padding_idx);

      if (mode == MODE_MEAN || mode == MODE_MAX || !fast_path_sum) {
        native::resize_(offset2bag, {indices.sizes()[0] + 1}, nullopt);
        native::zero_(offset2bag);
      }

      if (mode == MODE_MEAN || mode == MODE_MAX || !fast_path_sum) {
        make_offset2bag(offsets, offset2bag);
        native::resize_(offset2bag, {indices.sizes()[0]}, nullopt);
        // only initialize output in slow path
        native::zero_(output);
      }
        */
}

pub fn make_bag_size(
    offsets:             &Tensor,
    indices:             &Tensor,
    mode:                i64,
    include_last_offset: bool,
    requires_grad:       bool) -> Tensor {

    todo!();
        /*
            Tensor bag_size = empty(offsets.sizes(), offsets.options());
      make_bag_size_out(bag_size, offsets, indices, mode, include_last_offset, requires_grad);
      return bag_size;
        */
}

pub fn make_max_indices(
    weight:              &Tensor,
    indices:             &Tensor,
    offsets:             &Tensor,
    bag_size:            &Tensor,
    mode:                i64,
    include_last_offset: bool) -> Tensor {

    todo!();
        /*
            Tensor max_indices = empty(bag_size.sizes(), offsets.options());
      make_max_indices_out(max_indices, weight, indices, offsets, bag_size, mode, include_last_offset);
      return max_indices;
        */
}

pub fn make_offset2bag_b(
    output:             &mut Tensor,
    weight:             &Tensor,
    indices:            &Tensor,
    offsets:            &Tensor,
    mode:               i64,
    per_sample_weights: &Option<Tensor>,
    padding_idx:        i64) -> Tensor {

    todo!();
        /*
            Tensor offset2bag = empty({0}, offsets.options());
      make_offset2bag_out(offset2bag, output, weight, indices, offsets, mode, per_sample_weights, padding_idx);
      return offset2bag;
        */
}

pub fn apply_bag_size(
    mode:     i64,
    output:   &mut Tensor,
    bag_size: &Tensor) -> Tensor {
    
    todo!();
        /*
            if (mode == MODE_MEAN) {
        auto bag_size_ = max(bag_size, ones_like(bag_size, LEGACY_CONTIGUOUS_MEMORY_FORMAT))
                             .to(output.options())
                             .unsqueeze(1)
                             .expand_as(output);
        output /= bag_size_;
      }
      return output;
        */
}

pub fn apply_bag_size_backward(
    mode:       i64,
    output:     &mut Tensor,
    offset2bag: &Tensor,
    bag_size:   &Tensor) -> Tensor {
    
    todo!();
        /*
            if (mode == MODE_MEAN) {
        auto inv_bag_size_ = (1 / bag_size.to(output.options()))
                               .unsqueeze(1)
                               .index_select(0, offset2bag);
        output *= inv_bag_size_;
      }
      return output;
        */
}

pub fn embedding_bag_cpu_max_out<Scalar>(
    max_indices:         &mut Tensor,
    weight:              &Tensor,
    indices:             &Tensor,
    offset2bag:          &Tensor,
    output:              &Tensor,
    include_last_offset: bool,
    bag_size:            &mut Tensor,
    padding_idx:         i64)  {

    todo!();
        /*
            i64 numIndices = indices.numel();
      i64 featureSize = weight.sizes()[1];
      AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_bag_cpu_max_out", [&] {
        auto* indices_data = indices.data_ptr<Index>();
        auto* offset2bag_data = offset2bag.data_ptr<Index>();

        auto* max_indices_data = max_indices.data_ptr<Index>();
        auto max_indices_stride = max_indices.strides()[0];

        auto* weight_data = weight.data_ptr<Scalar>();
        auto* output_data = output.data_ptr<Scalar>();
        auto* bag_size_data = bag_size.data_ptr<Index>();
        auto weight_stride0 = weight.strides()[0];
        auto weight_stride1 = weight.strides()[1];
        auto output_stride = output.strides()[0];
        i64 numBags = bag_size.size(0);
        vector<bool> bag_empty(numBags, true);

        for (const auto i : irange(numIndices)) {
          auto bag = offset2bag_data[i];
          auto word_idx = indices_data[i];

          if (word_idx != static_cast<Index>(padding_idx)) {
            bool is_first_for_bag = bag_empty[bag];
            for (const auto dim : irange(featureSize)) {
              auto& current_item = output_data[output_stride * bag + dim];
              auto weight_item =
                  weight_data[weight_stride0 * word_idx + dim * weight_stride1];

              if (is_first_for_bag || (weight_item > current_item)) {
                current_item = weight_item;
                max_indices_data[max_indices_stride * bag + dim] = word_idx;
              }
            }
            if (is_first_for_bag) {
              bag_empty[bag] = false;
            }
          } else {
            // Decrement bag_size to reflect that the index is padded
            bag_size_data[bag]--;
          }
        }
      });
        */
}

pub fn embedding_bag_cpu_impl_out_b(
    output:              &mut Tensor,
    offset2bag:          &mut Tensor,
    bag_size:            &mut Tensor,
    max_indices:         &mut Tensor,
    weight:              &Tensor,
    indices:             &Tensor,
    offsets:             &Tensor,
    mode:                i64,
    per_sample_weights:  &Option<Tensor>,
    include_last_offset: bool,
    padding_idx:         i64)  {

    let mode: i64 = mode.unwrap_or(0);
    let per_sample_weights: &Option<Tensor> = per_sample_weights.unwrap_or(nullopt);
    let include_last_offset: bool = include_last_offset.unwrap_or(false);
    let padding_idx: i64 = padding_idx.unwrap_or(-1);
    
    todo!();
        /*
            if (mode == MODE_MEAN || mode == MODE_SUM) {
        AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "embedding_bag_no_grad_cpu_out",
          [&indices, &offset2bag, &per_sample_weights, &weight, &output, &offsets, &include_last_offset, &mode, &bag_size, &padding_idx]() {
          AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_bag_no_grad_cpu_out",
            [&indices, &offset2bag, &per_sample_weights, &weight, &output, &offsets, &include_last_offset, &mode, &bag_size, &padding_idx]() {
            if (per_sample_weights.has_value() && per_sample_weights.value().defined()) {
              TORCH_INTERNAL_ASSERT(mode == MODE_SUM);
              index_select_scale_add<Scalar, Index>(
                indices, offset2bag, per_sample_weights.value(), weight, output, offsets, include_last_offset, bag_size, padding_idx);
            } else {
              index_select_add<Scalar, Index>(indices, offset2bag, weight, output, offsets, include_last_offset, bag_size, padding_idx);
            }
          });
        });
        apply_bag_size(mode, output, bag_size);
        if (mode == MODE_SUM) {
          // make bag_size output deterministic
          native::zero_(bag_size);
        }
        max_indices = bag_size;
      } else { // MODE_MAX
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
          weight.scalar_type(), "embedding_bag_cpu_max_out", [&]() {
            embedding_bag_cpu_max_out<Scalar>(
              max_indices, weight, indices, offset2bag, output, include_last_offset, bag_size, padding_idx);
          }
        );
      }
        */
}

/**
  | Assumes all input tensors except for `weight`
  | are contiguous.
  |
  | See NOTE [ embedding_bag Native Functions ] in
  | native_functions.yaml for details
  |
  */
pub fn embedding_bag_cpu_impl(
        weight:              &Tensor,
        indices:             &Tensor,
        offsets:             &Tensor,
        mode:                i64,
        per_sample_weights:  &Tensor,
        include_last_offset: bool,
        padding_idx:         i64,
        requires_grad:       bool) -> (Tensor,Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            Tensor indices, offsets;
      tie(indices, offsets) = promoteIndicesAndOffsets(indices_, offsets_);
      check_arguments(weight, indices, offsets, mode, per_sample_weights, include_last_offset);

      Tensor output = empty(
          {include_last_offset ? offsets.sizes()[0] - 1 : offsets.sizes()[0],
           weight.sizes()[1]},
          weight.options());

      Tensor offset2bag = make_offset2bag(output, weight, indices, offsets, mode, per_sample_weights, padding_idx);

      Tensor bag_size = make_bag_size(offsets, indices, mode, include_last_offset, requires_grad);

      Tensor max_indices = make_max_indices(weight, indices, offsets, bag_size, mode, include_last_offset);

      _embedding_bag_cpu_impl_out(output, offset2bag,
                              bag_size, max_indices,
                              weight, indices, offsets,
                              mode, per_sample_weights,
                              include_last_offset, padding_idx);

      return make_tuple(move(output), move(offset2bag), move(bag_size), move(max_indices));
        */
}

/**
  | embedding_bag wrapper to enforce contiguity in
  | tensors other than `weight`.
  |
  | This is created to save extra `.contiguous()`
  | call in backward.
  |
  | See NOTE [ embedding_bag Native Functions ] in
  | native_functions.yaml for details
  |
  */
pub fn embedding_bag_a(
        weight:                 &Tensor,
        indices:                &Tensor,
        offsets:                &Tensor,
        scale_grad_by_freq:     bool,
        mode:                   i64,
        sparse:                 bool,
        per_sample_weights_opt: &Option<Tensor>,
        include_last_offset:    bool,
        padding_idx_opt:        Option<i64>) -> (Tensor,Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> per_sample_weights_maybe_owned = borrow_from_optional_tensor(per_sample_weights_opt);
      const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;
      i64 padding_idx = -1;

      if (padding_idx_opt.has_value()) {
        auto num_embeddings = weight.size(0);
        padding_idx = padding_idx_opt.value();
        TORCH_CHECK(
          (padding_idx >= -num_embeddings) && (padding_idx < num_embeddings),
          "padding_idx must be within the number of embeddings, -", num_embeddings,
          " through ", num_embeddings - 1, ", but got ", padding_idx);
        padding_idx = maybe_wrap_dim(padding_idx, weight.size(0));
      }
      tuple<Tensor, Tensor, Tensor, Tensor> out;
      if (!weight.requires_grad()) {
        out = _embedding_bag_forward_only(
          weight, indices.contiguous(), offsets.contiguous(), scale_grad_by_freq,
          mode, sparse, per_sample_weights, include_last_offset, padding_idx);
      } else {
        out = _embedding_bag(
          weight, indices.contiguous(), offsets.contiguous(), scale_grad_by_freq,
          mode, sparse, per_sample_weights, include_last_offset, padding_idx);
      }
      return out;
        */
}

pub fn embedding_bag_b(
        weight:                 &Tensor,
        indices:                &Tensor,
        offsets:                &Tensor,
        scale_grad_by_freq:     bool,
        mode:                   i64,
        sparse:                 bool,
        per_sample_weights_opt: &Option<Tensor>,
        include_last_offset:    bool) -> (Tensor,Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            return native::embedding_bag(weight, indices, offsets, scale_grad_by_freq,
          mode, sparse, per_sample_weights_opt, include_last_offset, nullopt);
        */
}

/**
  | Assumes all input tensors except for `weight`
  | are contiguous.
  |
  | See NOTE [ embedding_bag Native Functions ] in
  | native_functions.yaml for details
  |
  */
pub fn embedding_bag_forward_only_cpu(
        weight:                 &Tensor,
        indices:                &Tensor,
        offsets:                &Tensor,
        scale_grad_by_freq:     bool,
        mode:                   i64,
        sparse:                 bool,
        per_sample_weights_opt: &Option<Tensor>,
        include_last_offset:    bool,
        padding_idx:            i64) -> (Tensor,Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> per_sample_weights_maybe_owned = borrow_from_optional_tensor(per_sample_weights_opt);
      const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;
      ignore = scale_grad_by_freq;
      ignore = sparse;
      return _embedding_bag_cpu_impl(
          weight,
          indices,
          offsets,
          mode,
          per_sample_weights,
          include_last_offset,
          padding_idx,
          /*requires_grad=*/false);
        */
}

/**
  | Assumes all input tensors except for `weight`
  | are contiguous.
  |
  | See NOTE [ embedding_bag Native Functions ] in
  | native_functions.yaml for details
  |
  */
pub fn embedding_bag_cpu(
    weight:                 &Tensor,
    indices:                &Tensor,
    offsets:                &Tensor,
    scale_grad_by_freq:     bool,
    mode:                   i64,
    sparse:                 bool,
    per_sample_weights_opt: &Option<Tensor>,
    include_last_offset:    bool,
    padding_idx:            i64) -> (Tensor,Tensor,Tensor,Tensor) {

    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> per_sample_weights_maybe_owned = borrow_from_optional_tensor(per_sample_weights_opt);
      const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;

      ignore = scale_grad_by_freq;
      ignore = sparse;
      return _embedding_bag_cpu_impl(
          weight,
          indices,
          offsets,
          mode,
          per_sample_weights,
          include_last_offset,
          padding_idx,
          /*requires_grad=*/true);
        */
}

/**
  | Assumes all input tensors are contiguous.
  |
  | See NOTE [ embedding_bag Native Functions ] in
  | native_functions.yaml for details
  |
  */
pub fn embedding_bag_backward(
        grad:                   &Tensor,
        indices:                &Tensor,
        offsets:                &Tensor,
        offset2bag:             &Tensor,
        bag_size:               &Tensor,
        max_indices:            &Tensor,
        num_weights:            i64,
        scale_grad_by_freq:     bool,
        mode:                   i64,
        sparse:                 bool,
        per_sample_weights_opt: &Option<Tensor>,
        padding_idx:            i64) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> per_sample_weights_maybe_owned = borrow_from_optional_tensor(per_sample_weights_opt);
      const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;

      Tensor indices, offsets;
      tie(indices, offsets) = promoteIndicesAndOffsets(indices_, offsets_);
      auto indices_arg = TensorArg(indices, "indices", 1);
      checkScalarTypes("embedding_bag", indices_arg, {kLong, kInt});
      checkContiguous("embedding_bag", indices_arg);
      auto offsets_arg = TensorArg(offsets, "offsets", 1);
      checkScalarTypes("embedding_bag", offsets_arg, {kLong, kInt});
      checkSameType("embedding_bag", indices_arg, offsets_arg);
      checkContiguous("embedding_bag", offsets_arg);

      Tensor offset2bag_;
      if (indices.numel() != 0 && offset2bag.numel() == 0) {
        offset2bag_ = zeros(
           {indices.sizes()[0] + 1}, offsets.options()); // offset2bag = [0 0 0 0 0]

        make_offset2bag(offsets, offset2bag_);
        offset2bag_.resize_({indices.sizes()[0]});
      } else {
        auto offset2bag_arg = TensorArg(offset2bag, "offset2bag", 1);
        checkScalarTypes("embedding_bag", offset2bag_arg, {kLong, kInt});
        checkContiguous("embedding_bag", offset2bag_arg);
        offset2bag_ = offset2bag;
      }

      if (sparse) {
        return _embedding_bag_sparse_backward(
            grad, indices, offsets, offset2bag_, bag_size_, num_weights,
            scale_grad_by_freq, mode, per_sample_weights, padding_idx);
      } else {
        return _embedding_bag_dense_backward(
            grad, indices, offset2bag_, bag_size_, max_indices_, num_weights,
            scale_grad_by_freq, mode, per_sample_weights, padding_idx);
      }
        */
}

pub fn embedding_bag_dense_backward_cpu_max(
        grad:        &Tensor,
        bag_size:    &Tensor,
        max_indices: &Tensor,
        num_weights: i64) -> Tensor {
    
    todo!();
        /*
            AT_ASSERT(max_indices.defined());
      auto index_grad_weight =
          zeros({num_weights, grad.sizes()[1]}, grad.options());
      auto nonempty_max_indices = max_indices.index_select(0, bag_size.nonzero().view(-1));
      auto nonempty_grad = grad.index_select(0, bag_size.nonzero().view(-1));

      for (i64 dim = 0; dim < grad.sizes()[1]; dim++) {
        index_grad_weight.select(1, dim).index_add_(
          0, nonempty_max_indices.select(1, dim), nonempty_grad.select(1, dim));
      }
      return index_grad_weight;
        */
}

pub fn compute_counts<Index>(
        num_weights:    i64,
        indices_data:   *mut Index,
        indices_length: i64) -> Vec<Index> {

    todo!();
        /*
            vector<Index> counts(num_weights, 0);
      for (const auto i : irange(indices_length)) {
        counts[indices_data[i]]++;
      }
      return counts;
        */
}

/**
  | counts_uniq stores the index of the NEXT unique element
  | of the (sorted) indices vector.
  |
  | For example:
  | indices: [0, 0, 0, 1, 3, 3, 4]
  | counts: [3, 1, 0, 2, 1, 0]
  | counts_uniq: [3, 4, 6, 7]
  |
  | The unique indices can be found at index 0, 3, 4, 6.
  */
pub fn compute_counts_uniq<Index>(
        num_weights:    i64,
        indices_data:   *mut Index,
        indices_length: i64,
        counts:         &Vec<Index>) -> Vec<Index> {

    todo!();
        /*
            vector<Index> counts_uniq;
      counts_uniq.reserve(num_weights);
      i64 o = 0;
      for (i64 i = 0; i < indices_length; i += counts[indices_data[i]]) {
        counts_uniq.push_back(counts[indices_data[i]]);
        if (o > 0) {
          counts_uniq[o] += counts_uniq[o - 1];
        }
        o++;
      }
      return counts_uniq;
        */
}

pub fn embedding_bag_dense_backward_cpu_sum_mean<Scalar>(
        grad:               &Tensor,
        indices:            &Tensor,
        offset2bag:         &Tensor,
        bag_size:           &Tensor,
        num_weights:        i64,
        scale_grad_by_freq: bool,
        mode:               i64,
        per_sample_weights: &Tensor,
        index_grad_weight:  &mut Tensor,
        padding_idx:        i64)  {

    todo!();
        /*
      Tensor &offset2bag_ = const_cast<Tensor &>(offset2bag__);

      auto ind_sort_ = indices_.sort();
      auto indices = get<0>(ind_sort_);
      auto ind_sort = get<1>(ind_sort_);
      auto offset2bag = offset2bag_.index_select(0, ind_sort);

      optional<Tensor> per_sample_weights;
      Scalar* per_sample_weights_data;
      optional<i64> per_sample_weights_stride;
      if (per_sample_weights_.defined()) {
        per_sample_weights = per_sample_weights_.index_select(0, ind_sort);
        per_sample_weights_data = per_sample_weights->data_ptr<Scalar>();
        per_sample_weights_stride = per_sample_weights->strides()[0];
      }

      i64 numel = indices.numel();

      // explicitly capture all required variables to work around windows build
      // TODO: fix this when windows can correctly capture variables in nested lambda
      AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "_embedding_bag_dense_backward_cpu_sum_mean",
        [&indices, &offset2bag, &bag_size_, &num_weights, &numel, &per_sample_weights,
          &per_sample_weights_data, &per_sample_weights_stride, &mode, &scale_grad_by_freq,
          &grad, &index_grad_weight, &padding_idx] {
        auto* indices_data = indices.data_ptr<Index>();
        auto* offset2bag_data = offset2bag.data_ptr<Index>();
        auto* bag_size_data = bag_size_.data_ptr<Index>();

        auto counts = compute_counts(num_weights, indices_data, numel);
        auto next_unique_index_idx =
            compute_counts_uniq(num_weights, indices_data, numel, counts);

        auto loop =
          [&next_unique_index_idx, &indices_data, &offset2bag_data, &bag_size_data, &per_sample_weights,
            &mode, &per_sample_weights_data, &per_sample_weights_stride, &scale_grad_by_freq,
            &counts, &grad, &index_grad_weight, &padding_idx
          ](Index start, Index end) {
          for (Index i = start; i < end; i++) {
            Index start = i == 0 ? 0 : next_unique_index_idx[i - 1];
            Index index = indices_data[start];

            if (index != static_cast<Index>(padding_idx)) {
              for (Index j = start; j < next_unique_index_idx[i]; j++) {
                Index source = offset2bag_data[j];
                double scale = 1.0;
                if (per_sample_weights) {
                  AT_ASSERT(mode == MODE_SUM);
                  scale = per_sample_weights_data[*per_sample_weights_stride * j];
                }
                if (scale_grad_by_freq) {
                  scale /= counts[indices_data[i]];
                }
                if (mode == MODE_MEAN) {
                  auto bag_size = bag_size_data[source];
                  if (bag_size != 0) {
                    scale /= bag_size;
                  }
                }
                i64 ddim = grad.size(1);
                auto igwd = index_grad_weight.data_ptr<Scalar>();
                auto gd = grad.data_ptr<Scalar>();
                native::cpublas::axpy<Scalar>(ddim, (Scalar)scale, gd + ddim * source, 1,
                            igwd + ddim * index, 1);
              }
            }
          }
        };

        if (numel > 1000) {
          parallel_for(0, (i64)next_unique_index_idx.size(), 0, loop);
        } else {
          loop(0, (i64)next_unique_index_idx.size());
        }
      });
        */
}

pub fn embedding_bag_dense_backward_cpu(
        grad:                   &Tensor,
        indices:                &Tensor,
        offset2bag:             &Tensor,
        bag_size:               &Tensor,
        max_indices:            &Tensor,
        num_weights:            i64,
        scale_grad_by_freq:     bool,
        mode:                   i64,
        per_sample_weights_opt: &Option<Tensor>,
        padding_idx:            i64) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> per_sample_weights__maybe_owned = borrow_from_optional_tensor(per_sample_weights__opt);
      const Tensor& per_sample_weights_ = *per_sample_weights__maybe_owned;

      // indices_, offsets_ and offset2bag__ are assumed having correct dtypes and
      // contiguous here due to the checks in _embedding_bag_backward above.
      // Also see NOTE [ embedding_bag Native Functions ] in native_functions.yaml
      // for more details.
      auto grad = grad_.contiguous();
      auto grad_arg = TensorArg(grad, "grad_", 1);
      checkScalarTypes("embedding_bag", grad_arg, {kFloat, kDouble});

      if (mode == MODE_MAX) {
        return _embedding_bag_dense_backward_cpu_max(
            grad_, bag_size_, max_indices_, num_weights);
      }
      AT_ASSERT(mode == MODE_MEAN || mode == MODE_SUM);

      auto index_grad_weight =
          zeros({num_weights, grad.sizes()[1]}, grad.options());

      AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "embedding_bag_backward", [&] {
          _embedding_bag_dense_backward_cpu_sum_mean<Scalar>(
              grad, indices_, offset2bag__, bag_size_, num_weights,
              scale_grad_by_freq, mode, per_sample_weights_, index_grad_weight,
              padding_idx);
      });
      return index_grad_weight;
        */
}

pub fn embedding_bag_per_sample_weights_backward_cpu_template<Scalar>(
        grad:        &Tensor,

        // NB: embedding table, not per_sample_weights
        weight:      &Tensor,
        indices:     &Tensor,
        offsets:     &Tensor,
        offset2bag:  &Tensor,
        mode:        i64,
        padding_idx: i64) -> Tensor {

    todo!();
        /*
            TORCH_CHECK(
          mode == MODE_SUM,
          "embedding_bag_backward: per_sample_weights only supported for mode='sum'");

      AT_ASSERT(grad.dim() == 2);
      auto embedding_features = grad.sizes()[1];

      Tensor indices, offsets;
      tie(indices, offsets) = promoteIndicesAndOffsets(indices_, offsets_);
      AT_ASSERT(indices.dim() == 1);
      auto num_samples = indices.sizes()[0];

      AT_ASSERT(weight.dim() == 2);
      AT_ASSERT(weight.sizes()[1] == embedding_features);

      auto output = zeros({num_samples}, grad.options());

      auto indices_arg = TensorArg(indices, "indices", 1);
      checkScalarTypes("embedding_bag", indices_arg, {kLong, kInt});
      checkContiguous("embedding_bag", indices_arg);

      Tensor offset2bag_;
      if (indices.numel() != 0 && offset2bag.numel() == 0) {
        offset2bag_ = zeros(
           {indices.sizes()[0] + 1}, offset2bag.options()); // offset2bag = [0 0 0 0 0]

        make_offset2bag(offsets, offset2bag_);

        native::resize_(offset2bag_, {indices.sizes()[0]}, nullopt);
      } else {
        auto offset2bag_arg = TensorArg(offset2bag, "offset2bag", 1);
        checkScalarTypes("embedding_bag", offset2bag_arg, {kLong, kInt});
        checkContiguous("embedding_bag", offset2bag_arg);
        offset2bag_ = offset2bag;
      }

      auto* grad_data = grad.data_ptr<Scalar>();
      auto grad_stride0 = grad.strides()[0];
      auto grad_stride1 = grad.strides()[1];

      auto* weight_data = weight.data_ptr<Scalar>();
      auto weight_stride0 = weight.strides()[0];
      auto weight_stride1 = weight.strides()[1];

      // explicitly capture all required variables to work around windows build
      // TODO: fix this when windows can correctly capture variables in nested lambda
      AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "_embedding_bag_per_sample_weights_backward_cpu_template",
        [&indices, &output, &offset2bag_, &num_samples, &embedding_features,
          &grad_data, &grad_stride0, &grad_stride1, &weight_data, &weight_stride0, &weight_stride1,
          &padding_idx] () {
        auto* indices_data = indices.data_ptr<Index>();

        // The following are contiguous
        auto* output_data = output.data_ptr<Scalar>();
        auto* offset2bag_data = offset2bag_.data_ptr<Index>();

        // XXX: 64 was arbitrarily chosen. There is probably a sweet spot for this number.
        parallel_for(0, num_samples, 64,
          [&embedding_features, &grad_data, &grad_stride0, &grad_stride1, &weight_data, &weight_stride0,
            &weight_stride1, &offset2bag_data, &indices_data, &output_data, &padding_idx](Index begin, Index end) {
          for (Index sample_idx = begin; sample_idx < end; sample_idx++) {
            auto bag_idx = offset2bag_data[sample_idx];
            auto embedding_idx = indices_data[sample_idx];

            if (embedding_idx != static_cast<Index>(padding_idx)) {
              output_data[sample_idx] = dot_impl<Scalar>(
                  embedding_features,
                  grad_data + grad_stride0 * bag_idx, grad_stride1,
                  weight_data + weight_stride0 * embedding_idx, weight_stride1);
            }
          }
        });
      });
      return output;
        */
}

pub fn embedding_bag_per_sample_weights_backward_cpu(
        grad:        &Tensor,

        // NB: embedding table, not per_sample_weights
        weight:      &Tensor,
        indices:     &Tensor,
        offsets:     &Tensor,
        offset2bag:  &Tensor,
        mode:        i64,
        padding_idx: i64) -> Tensor {
    
    todo!();
        /*
            return AT_DISPATCH_FLOATING_TYPES(
        grad.scalar_type(), "_embedding_bag_per_sample_weights_backward_cpu", [&]() {
          return _embedding_bag_per_sample_weights_backward_cpu_template<Scalar>(
              grad, weight, indices, offsets, offset2bag, mode, padding_idx);
        }
      );
        */
}

pub fn embedding_bag_sparse_backward(
        grad:                   &Tensor,
        indices:                &Tensor,
        offsets:                &Tensor,
        offset2bag:             &Tensor,
        bag_size:               &Tensor,
        num_weights:            i64,
        scale_grad_by_freq:     bool,
        mode:                   i64,
        per_sample_weights_opt: &Option<Tensor>,
        padding_idx:            i64) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> per_sample_weights_maybe_owned = borrow_from_optional_tensor(per_sample_weights_opt);
      const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;

      // indices, offsets and offset2bag are assumed having correct dtypes and
      // contiguous here due to the checks in _embedding_bag_backward above.
      // Also see NOTE [ embedding_bag Native Functions ] in native_functions.yaml
      // for more details.

      Tensor grad = grad_;
      Tensor index_grad = grad_.index_select(0, offset2bag);

      index_grad = apply_bag_size_backward(mode, index_grad, offset2bag, bag_size_);

      if (per_sample_weights.defined()) {
        AT_ASSERT(mode == MODE_SUM);
        index_grad.mul_(per_sample_weights.unsqueeze(1));
      }
      return native::embedding_backward(index_grad, indices, num_weights, padding_idx,
                                        scale_grad_by_freq, true);
        */
}
