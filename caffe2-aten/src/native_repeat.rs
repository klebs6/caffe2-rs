crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Repeat.h]

//template < typename Index, void compute(Index*, i64*, Index*, i64, i64)>
#[inline] pub fn repeat_interleave_common(
        repeats:     &Tensor,
        output_size: Option<i64>) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          repeats.dim() == 1, "repeat_interleave only accept 1D vector as repeat");
      TORCH_CHECK(
          repeats.scalar_type() == kLong || repeats.scalar_type() == kInt,
          "repeats has to be Long or Int tensor");
      if (repeats.size(0) == 0) {
        return empty_like(repeats, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      }
      Tensor repeats_ = repeats.contiguous();
      Tensor cumsum = repeats.cumsum(0);
      i64 total;
      if (output_size.has_value()) {
        total = output_size.value();
      } else {
        total = cumsum[-1].item<i64>();
        TORCH_CHECK(
            (repeats >= 0).all().item<u8>(), "repeats can not be negative");
      }

      Tensor result = empty({total}, repeats.options());
      Index* repeat_ptr = repeats_.data_ptr<Index>();
      i64* cumsum_ptr = cumsum.data_ptr<i64>();
      Index* result_ptr = result.data_ptr<Index>();
      compute(repeat_ptr, cumsum_ptr, result_ptr, repeats.size(0), total);
      return result;
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Repeat.cpp]

pub fn compute_cpu<Index>(
        repeat_ptr:  *mut Index,
        cumsum_ptr:  *mut i64,
        result_ptr:  *mut Index,
        size:        i64,
        result_size: i64)  {

    todo!();
        /*
            TORCH_CHECK(
          (result_size == cumsum_ptr[size - 1]),
          "allocated size does not match required size");
      parallel_for(0, size, 1, [&](i64 i_begin, i64 i_end) {
        for (i64 i = i_begin; i < i_end; i++) {
          i64 end = cumsum_ptr[i];
          Index size = repeat_ptr[i];
          TORCH_CHECK((size >= 0), "repeats can not be negative");
          i64 start = end - size;
          for (i64 j = start; j < end; j++) {
            result_ptr[j] = i;
          }
        }
      });
        */
}

pub fn repeat_interleave_cpu(
        repeat:      &Tensor,
        output_size: Option<i64>) -> Tensor {
    
    todo!();
        /*
            Tensor output;
      AT_DISPATCH_INDEX_TYPES(repeat.scalar_type(), "repeat_interleave_cpu", [&]() {
        output = repeat_interleave_common<Index, compute_cpu<Index>>(
            repeat, output_size);
      });

      return output;
        */
}

pub fn repeat_interleave_a(
        self_:       &Tensor,
        repeats:     &Tensor,
        dim:         Option<i64>,
        output_size: Option<i64>) -> Tensor {
    
    todo!();
        /*
            Tensor input = self;
      if (!dim) {
        input = self.flatten();
        dim = 0;
      }

      Tensor repeats_ = repeats;
      if (repeats.dim() == 0 || (repeats.dim() == 1 && repeats.size(0) == 1)) {
        repeats_ = repeats.reshape({1}).expand({input.size(dim.value())});
      } else if (repeats.dim() == 1) {
        TORCH_CHECK(
            repeats.size(0) == input.size(dim.value()),
            "repeats must have the same size as input along dim")
      } else {
        AT_ERROR("repeats must be 0-dim or 1-dim tensor");
      }

      return input.index_select(
          dim.value(), repeat_interleave(repeats_, output_size));
        */
}

pub fn repeat_interleave_b(
        self_:       &Tensor,
        repeats:     i64,
        dim:         Option<i64>,
        output_size: Option<i64>) -> Tensor {
    
    todo!();
        /*
            return native::repeat_interleave(
          self,
          tensor({repeats}, self.options().dtype(kLong)),
          dim,
          output_size);
        */
}
