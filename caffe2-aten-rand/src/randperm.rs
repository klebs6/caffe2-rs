crate::ix!();

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ randperm ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn randperm_cpu<Scalar>(
    result:    &mut Tensor,
    n:         i64,
    generator: *mut CPUGeneratorImpl)  {

    todo!();
        /*
            Scalar *r__data = result.data_ptr<Scalar>();

      result.resize_({n});
      i64 r__stride_0 = result.stride(0);

      parallel_for(0, n, internal::GRAIN_SIZE,
                      [&r__data, &r__stride_0](i64 p_begin, i64 p_end) {
        for(i64 i = p_begin; i < p_end; i++)
          r__data[i*r__stride_0] = static_cast<Scalar>(i);
      });

      for(i64 i = 0; i < n - 1; i++)
      {
        i64 z = generator->random() % (n-i);
        Scalar sav = r__data[i*r__stride_0];
        r__data[i*r__stride_0] = r__data[(z+i)*r__stride_0];
        r__data[(z+i)*r__stride_0] = sav;
      }
        */
}

pub fn randperm_a(
        n:          i64,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::randperm(n, nullopt, dtype, layout, device, pin_memory);
        */
}

pub fn randperm_b(
        n:          i64,
        generator:  Option<dyn GeneratorInterface>,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            if (!dtype.has_value()) {
        dtype = ScalarType::Long;
      }

      // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      auto tensor = empty(n, options);
      return randperm_out(tensor, n, generator);
        */
}

pub fn randperm_out<'a>(
        n:      i64,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return randperm_out(result, n, nullopt);
        */
}

pub fn randperm_out_cpu<'a>(
        n:         i64,
        generator: Option<dyn GeneratorInterface>,
        result:    &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
      TORCH_CHECK(!generator.has_value() || (generator.has_value() && result.device() == generator->device()), "Expected a '", result.device(), "' generator device but found '", generator->device(), "'");
      check_supported_max_int_with_precision(n, result);
      result.resize_({n});
      auto gen = get_generator_or_default<CPUGeneratorImpl>(generator, getDefaultCPUGenerator());
      // See Note [Acquire lock when using random generators]
      lock_guard<mutex> lock(gen->mutex_);
      AT_DISPATCH_ALL_TYPES_AND(ScalarType::Half, result.scalar_type(), "randperm", [&]() -> void {
        randperm_cpu<Scalar>(result, n, gen);
      });

      return result;
        */
}
