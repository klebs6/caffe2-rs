crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/SumKernel.cpp]

pub struct LoadImpl<Scalar> {

}

impl LoadImpl<Scalar> {
    
    pub fn load(
        data:   *const u8,
        stride: i64,
        index:  i64) -> Scalar {
        
        todo!();
        /*
            auto *ptr = reinterpret_cast<const Scalar*>(data + index * stride);
        return *ptr;
        */
    }
}

lazy_static!{
    /*
    template <typename Scalar>
    struct LoadImpl<Vectorized<Scalar>> {

      static Vectorized<Scalar> load(const char *  data, i64 stride, i64 index) {
        auto *ptr = data + index * stride;
        return Vectorized<Scalar>::loadu(ptr);
      }
    };
    */
}

pub fn load<T>(
        data:   *const u8,
        stride: i64,
        index:  i64) -> T {

    todo!();
        /*
            return LoadImpl<T>::load(data, stride, index);
        */
}

pub fn accumulate_result<Scalar>(
        data:   *mut u8,
        stride: i64,
        index:  i64,
        value:  Scalar)  {

    todo!();
        /*
            auto *const ptr = reinterpret_cast<Scalar*>(data + index * stride);
      *ptr += value;
        */
}

pub fn accumulate_result_array<Scalar, const N: usize>(
    data:   *mut u8,
    stride: i64,
    index:  i64,
    values: &Array<Scalar,N>)  {

    todo!();
        /*
            auto *const base_ptr = data + stride * index;
      for (const auto k : irange(numel)) {
        accumulate_result(base_ptr, stride, k, values[k]);
      }
        */
}


/** Simultaneously sum over n rows at once

This algorithm calculates the sum without loss of precision over large axes. It
does this by chunking the sum into groups of 16 or more elements. The sums of
these chunks are also summed in chunks and so on until there is just a single sum
value remaining. This means only numbers of a similar order of magnitude are
added together, thus minimising rounding errors.

This is done in a single linear pass over the data and with O(1) extra storage.
A simplified recursive implementation would look like this:

  Scalar row_sum(const Scalar * data, i64 n) {
    // Note, in practice the chunk size can increase with n
    // This allows the recursion depth to be limited to O(1).
    constexpr i64 min_chunk_size = 16;

    Scalar sum = 0;
    if (n <= min_chunk_size) {
      // Recursive base case, calculate a simple running sum
      for (i64 i = 0; i < n; ++i) {
        sum += data[i];
      }
      return sum;
    }

    // Recursively sum larger chunks of elements
    const i64 chunk_size = max(divup(n, min_chunk_size), min_chunk_size);
    for (i64 i = 0; i < n; i += chunk_size) {
      sum += row_sum(data + i, min(chunk_size, n - i));
    }
    return sum;
  }
*/
pub fn multi_row_sum<Scalar, const nrows: i64>(
    in_data:    *const u8,
    row_stride: i64,
    col_stride: i64,
    size:       i64) -> Array<Scalar,NRows> {

    todo!();
        /*
            constexpr i64 num_levels = 4;

      const i64 level_power =
          max(i64(4), utils::CeilLog2(size) / num_levels);
      const i64 level_step = (1 << level_power);
      const i64 level_mask = level_step - 1;

      Scalar acc[num_levels][nrows];
      fill_n(&acc[0][0], num_levels * nrows, Scalar(0));

      i64 i = 0;
      for (; i + level_step <= size;) {
        for (i64 j = 0; j < level_step; ++j, ++i) {
          const char * sum_base = in_data + i * row_stride;
          #if !defined(COMPILING_FOR_MIN_SIZE)
          # pragma unroll
          #endif
          for (i64 k = 0; k < nrows; ++k) {
            acc[0][k] += load<Scalar>(sum_base, col_stride, k);
          }
        }

        for (i64 j = 1; j < num_levels; ++j) {
          #if !defined(COMPILING_FOR_MIN_SIZE)
          # pragma unroll
          #endif
          for (i64 k = 0; k < nrows; ++k) {
            acc[j][k] += acc[j-1][k];
            acc[j-1][k] = Scalar(0);
          }

          const auto mask = (level_mask << (j * level_power));
          if ((i & mask) != 0) {
            break;
          }
        }
      }

      for (; i < size; ++i) {
        const char * sum_base = in_data + i * row_stride;
        #if !defined(COMPILING_FOR_MIN_SIZE)
        # pragma unroll
        #endif
        for (i64 k = 0; k < nrows; ++k) {
          acc[0][k] += load<Scalar>(sum_base, col_stride, k);
        }
      }

      for (i64 j = 1; j < num_levels; ++j) {
        #if !defined(COMPILING_FOR_MIN_SIZE)
        # pragma unroll
        #endif
        for (i64 k = 0; k < nrows; ++k) {
          acc[0][k] += acc[j][k];
        }
      }

      array<Scalar, nrows> ret;
      for (i64 k = 0; k < nrows; ++k) {
        ret[k] = acc[0][k];
      }
      return ret;
        */
}

pub fn row_sum<Scalar>(
        in_data:   *const u8,
        in_stride: i64,
        size:      i64) -> Scalar {

    todo!();
        /*
            constexpr i64 ilp_factor = 4;

      // Interpret row as a (-1, ilp_factor) shaped array to find partial sums
      const i64 size_ilp = size / ilp_factor;
      auto partial_sums = multi_row_sum<Scalar, ilp_factor>(
          in_data, in_stride * ilp_factor, in_stride, size_ilp);

      for (i64 i = size_ilp * ilp_factor; i < size; ++i) {
        partial_sums[0] += load<Scalar>(in_data, in_stride, i);
      }

      for (i64 k = 1; k < ilp_factor; ++k) {
        partial_sums[0] += partial_sums[k];
      }

      return partial_sums[0];
        */
}

pub fn vectorized_inner_sum<Scalar>(
        data:         [*mut u8; 2],
        outer_stride: i64,
        out_stride:   i64,
        size0:        i64,
        size1:        i64)  {

    todo!();
        /*
            using vec_t = Vectorized<Scalar>;
      constexpr i64 vec_stride = vec_t::size() * sizeof(Scalar);
      const i64 vec_size = size0 / vec_t::size();

      // Input is contiguous over the first (reduced) dimension
      for (i64 j = 0; j < size1; ++j) {
        const auto *row_in = data[1] + j * outer_stride;
        auto vec_acc = row_sum<vec_t>(row_in, vec_stride, vec_size);

        Scalar final_acc = 0;
        for (i64 k = vec_size * vec_t::size(); k < size0; ++k) {
          final_acc += load<Scalar>(row_in, sizeof(Scalar), k);
        }

        Scalar partials[vec_t::size()];
        vec_acc.store(partials);
        for (i64 k = 0; k < vec_t::size(); ++k) {
          final_acc += partials[k];
        }
        accumulate_result(data[0], out_stride, j, final_acc);
      }
        */
}

pub fn scalar_inner_sum<Scalar>(
        data:       [*mut u8; 2],
        in_strides: [i64; 2],
        out_stride: i64,
        size0:      i64,
        size1:      i64)  {

    todo!();
        /*
            for (i64 j = 0; j < size1; ++j) {
        const auto *row_in = data[1] + j * in_strides[1];
        Scalar ans = row_sum<Scalar>(row_in, in_strides[0], size0);
        accumulate_result(data[0], out_stride, j, ans);
      }
        */
}

pub fn vectorized_outer_sum<Scalar>(
        data:         [*mut u8; 2],
        inner_stride: i64,
        out_stride:   i64,
        size0:        i64,
        size1:        i64)  {

    todo!();
        /*
            using vec_t = Vectorized<Scalar>;
      constexpr i64 nrows = 4;
      constexpr i64 vec_stride = vec_t::size() * sizeof(Scalar);

      // Input is contiguous over the second (non-reduced) dimension
      i64 j = 0;
      for (; j + nrows * vec_t::size() <= size1; j += nrows * vec_t::size()) {
        const auto *row_in = data[1] + j * sizeof(Scalar);
        auto sums = multi_row_sum<vec_t, nrows>(row_in, inner_stride, vec_stride, size0);

        for (i64 i = 0; i < nrows; ++i) {
          const i64 base_idx = j + i * vec_t::size();

          array<Scalar, vec_t::size()> ans;
          sums[i].store(ans.data());
          accumulate_result(data[0], out_stride, base_idx, ans);
        }
      }

      for (; j + vec_t::size() <= size1; j += vec_t::size()) {
        const auto *row_in = data[1] + j * sizeof(Scalar);
        const vec_t sums = row_sum<vec_t>(row_in, inner_stride, size0);

        array<Scalar, vec_t::size()> ans;
        sums.store(ans.data());
        accumulate_result(data[0], out_stride, j, ans);
      }

      for (; j < size1; ++j) {
        const auto *row_in = data[1] + j * sizeof(Scalar);
        Scalar ans = row_sum<Scalar>(row_in, inner_stride, size0);
        accumulate_result(data[0], out_stride, j, ans);
      }
        */
}

pub fn scalar_outer_sum<Scalar>(
        data:       [*mut u8; 2],
        in_strides: [i64; 2],
        out_stride: i64,
        size0:      i64,
        size1:      i64)  {

    todo!();
        /*
            constexpr i64 nrows = 4;
      i64 j = 0;
      for (; j + (nrows - 1) < size1; j += nrows) {
        const auto *row_in = data[1] + j * in_strides[1];
        auto sums = multi_row_sum<Scalar, nrows>(
            row_in, in_strides[0], in_strides[1], size0);
        accumulate_result(data[0], out_stride, j, sums);
      }

      for (; j < size1; ++j) {
        const auto *row_in = data[1] + j * in_strides[1];
        Scalar ans = row_sum<Scalar>(row_in, in_strides[0], size0);
        accumulate_result(data[0], out_stride, j, ans);
      }
        */
}

pub fn sum_kernel_impl(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            if (isIntegralType(iter.dtype(), /*includeBool=*/ true)) {
        AT_DISPATCH_INTEGRAL_TYPES_AND(ScalarType::Bool, iter.dtype(), "sum_cpu",
          [&] {
            binary_kernel_reduce_vec(
                iter, [=](Scalar a, Scalar b) -> Scalar { return a + b; },
                [=](Vectorized<Scalar> a, Vectorized<Scalar> b) { return a + b; });
          });
        return;
      }

      // Custom floating point sum for better accuracy
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        ScalarType::BFloat16, ScalarType::Half, iter.dtype(), "sum_cpu",
        [&] {
          iter.output().fill_(Scalar(0));
          iter.parallel_reduce(
            [&](char** data, const i64* strides, i64 size0, i64 size1) {
              i64 in_strides[] = { strides[1], strides[3] };
              i64 out_strides[] = { strides[0], strides[2] };

              // Move reduction to be the 1st dim
              if (out_strides[0] != 0 && out_strides[1] == 0) {
                swap(in_strides[0], in_strides[1]);
                swap(out_strides[0], out_strides[1]);
                swap(size0, size1);
              }

              // Special case? - not a true reduction
              if (out_strides[0] != 0 && out_strides[1] != 0) {
                i64 outer_strides[] = { strides[2], strides[3] };
                UNARY_OUTER_LOOP(data, outer_strides, size1, [&] {
                  char* ptrs[3] = { data[0], data[0], data[1] };
                  i64 inner_strides[3] = { strides[0], strides[0], strides[1] };
                  basic_loop(ptrs, inner_strides, 0, size0, [](Scalar a, Scalar b) { return a + b; });
                });
                return;
              }

              const i64 out_stride = out_strides[1];
              TORCH_INTERNAL_ASSERT(out_strides[0] == 0);

              if (in_strides[0] == sizeof(Scalar) && size0 >= Vectorized<Scalar>::size()) {
                // Contiguous inner reduction
                vectorized_inner_sum<Scalar>(data, in_strides[1], out_stride, size0, size1);
              } else if (in_strides[1] == sizeof(Scalar) && size1 >= Vectorized<Scalar>::size()) {
                // Contiguous outer reduction
                vectorized_outer_sum<Scalar>(data, in_strides[0], out_stride, size0, size1);
              } else if (in_strides[0] < in_strides[1]) {
                scalar_inner_sum<Scalar>(data, in_strides, out_stride, size0, size1);
              } else {
                scalar_outer_sum<Scalar>(data, in_strides, out_stride, size0, size1);
              }
            });
        });
        */
}

register_dispatch!{sum_stub, &sum_kernel_impl}
