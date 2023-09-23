crate::ix!();

/**
  | Different combinations of row, col, and offset
  | can lead to two cases:
  |
  | Case 1 - Trapezoid (Triangle as a special case): row + offset <= col
  |    Example A: offset > 0
  |      1 1 0 0 0
  |      1 1 1 0 0
  |      1 1 1 1 0
  |    Example B: offset <= 0
  |      0 0 0
  |      1 0 0
  |      1 1 0
  |    In this case, we calculate the number of elements in the first row and
  |    last row of the tril respectively, and then compute the tril size.
  |
  | Case 2 - Trapezoid + Rectangle: row + offset > col
  |    Example:
  |      1 1 0
  |      1 1 1
  |      1 1 1
  |    In this case, we first calculate the size of top trapezoid, and then
  |    calculate the size of the bottom rectangle.
  */
#[inline] pub fn get_tril_size(
        row:    i64,
        col:    i64,
        offset: i64) -> i64 {
    
    todo!();
        /*
            // number of elements in the first row of the tril
      auto m_first_row = offset > 0 ?
        min<i64>(col, 1 + offset) : // upper bounded by col
        row + offset > 0; // either 0 or 1
      // number of elements in the last row of the tril, bounded by [0, col]
      auto m_last_row = max<i64>(0, min<i64>(col, row + offset));
      // number of rows, bounded by [0, row]
      auto n_row_all = max<i64>(0, min<i64>(row, row + offset));
      auto n_row_trapezoid = (m_last_row - m_first_row + 1);

      // calculate # of elements in the top trapezoid
      auto tril_size = (m_first_row + m_last_row) * n_row_trapezoid >> 1;

      // calculate # of elements in the bottom rectangle if there is any
      auto diff_row = n_row_all - n_row_trapezoid;
      if (diff_row > 0) {
        tril_size += diff_row * col;
      }

      return tril_size;
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triangle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn tril_indices_cpu(
        row:            i64,
        col:            i64,
        offset:         i64,
        dtype_opt:      Option<ScalarType>,
        layout_opt:     Option<Layout>,
        device_opt:     Option<Device>,
        pin_memory_opt: Option<bool>) -> Tensor {
    
    todo!();
        /*
            if (!dtype_opt.has_value()) {
        dtype_opt = ScalarType::Long;
      }

      check_args(row, col, layout_opt);

      auto tril_size = get_tril_size(row, col, offset);

      // create an empty Tensor with correct size
      auto result = native::empty_cpu({2, tril_size}, dtype_opt, layout_opt, device_opt, pin_memory_opt);

      // The following three approaches result in very little performance
      // differences. Hence, the 2nd option is taken for simpler code, and to return
      // contiguous tensors. Refer to #14904 for more details.
      //
      // 1. sequential RAM access: fill row coordinates first, then columns. This
      //    results in two for-loop and more arithmetic operations.
      //
      // 2. interleaved RAM access: fill in index coordinates one by one, which
      //    jumps between the two output Tensor rows in every iteration.
      //
      // 3. sequential RAM + transpose: create an n X 2 Tensor, fill the Tensor
      //    sequentially, and then transpose it.
      AT_DISPATCH_ALL_TYPES(result.scalar_type(), "tril_indices", [&]() -> void {
        // fill the Tensor with correct values
        Scalar* result_data = result.data_ptr<Scalar>();
        i64 i = 0;

        Scalar r = max<i64>(0, -offset), c = 0;
        while (i < tril_size) {
          result_data[i] = r;
          result_data[tril_size + i++] = c;

          // move to the next column and check if (r, c) is still in bound
          c += 1;
          if (c > r + offset || c >= col) {
            r += 1;
            c = 0;
            // NOTE: not necessary to check if r is less than row here, because i
            // and tril_size provide the guarantee
          }
        }
      });

      return result;
        */
}

pub fn triu_indices_cpu(
        row:            i64,
        col:            i64,
        offset:         i64,
        dtype_opt:      Option<ScalarType>,
        layout_opt:     Option<Layout>,
        device_opt:     Option<Device>,
        pin_memory_opt: Option<bool>) -> Tensor {
    
    todo!();
        /*
            if (!dtype_opt.has_value()) {
        dtype_opt = ScalarType::Long;
      }

      check_args(row, col, layout_opt);

      auto triu_size = row * col - get_tril_size(row, col, offset - 1);

      // create an empty Tensor with correct size
      auto result = native::empty_cpu({2, triu_size}, dtype_opt, layout_opt, device_opt, pin_memory_opt);

      AT_DISPATCH_ALL_TYPES(result.scalar_type(), "triu_indices", [&]() -> void {
        // fill the Tensor with correct values
        Scalar* result_data = result.data_ptr<Scalar>();
        i64 i = 0;
        // not typing max with Scalar as it could be an unsigned type
        // NOTE: no need to check if the returned value of max overflows
        // Scalar, as i and triu_size act as a guard.
        Scalar c = max<i64>(0, offset), r = 0;
        while (i < triu_size) {
          result_data[i] = r;
          result_data[triu_size + i++] = c;

          // move to the next column and check if (r, c) is still in bound
          c += 1;
          if (c >= col) {
            r += 1;
            // not typing max with Scalar as it could be an unsigned type
            // NOTE: not necessary to check if c is less than col or overflows here,
            // because i and triu_size act as a guard.
            c = max<i64>(0, r + offset);
          }
        }
      });

      return result;
        */
}
