crate::ix!();

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ eye ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn eye_a(
        n:          i64,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // the default value of `m` equals to `n`
      return native::eye(n, n, dtype, layout, device, pin_memory);
        */
}

pub fn eye_b(
        n:          i64,
        m:          i64,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      auto tensor = empty({0}, options); // to be resized
      return eye_out(tensor, n, m);
        */
}


pub fn eye_out_cpu_a<'a>(
        n:      i64,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            // the default value of `m` equals to `n`
      return native::eye_out_cpu(n, n, result);
        */
}

pub fn eye_out_cpu_b<'a>(
        n:      i64,
        m:      i64,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);
      TORCH_CHECK(m >= 0, "m must be greater or equal to 0, got ", m);

      result.resize_({n, m});
      result.zero_();

      i64 sz = min<i64>(n, m);
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(ScalarType::Half, ScalarType::Bool, result.scalar_type(), "eye", [&]() -> void {
        Scalar* result_data = result.data_ptr<Scalar>();
        parallel_for(0, sz, internal::GRAIN_SIZE, [&](i64 p_begin, i64 p_end) {
          for(i64 i = p_begin; i < p_end; i++)
            result_data[i*(result.strides()[0] + result.strides()[1])] = 1;
        });
      });

      return result;
        */
}
