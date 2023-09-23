crate::ix!();

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ arange ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn arange_a(
        end:        &Scalar,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::arange(/*start=*/0, end, dtype, layout, device, pin_memory);
        */
}

pub fn arange_b(
        start:      &Scalar,
        end:        &Scalar,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::arange(
          start, end, /*step=*/1, dtype, layout, device, pin_memory);
        */
}

pub fn arange_c(
        start:      &Scalar,
        end:        &Scalar,
        step:       &Scalar,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      bool set_to_integral_dtype = !options.has_dtype() &&
           // bool inputs are considered integral
           start.isIntegral(true) &&
           end.isIntegral(true) &&
           step.isIntegral(true);

      Tensor result = set_to_integral_dtype
          ? empty({0}, options.dtype(ScalarType::Long))
          : empty({0}, options);
      return arange_out(result, start, end, step);
        */
}

pub fn arange_out_a<'a>(
        end:    &Scalar,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return arange_out(result, /*start=*/0, end);
        */
}

pub fn arange_out_b<'a>(
        result: &mut Tensor,
        start:  &Scalar,
        end:    &Scalar) -> &'a mut Tensor {
    
    todo!();
        /*
            return arange_out(result, start, end, /*step=*/1);
        */
}

pub fn dim_arange(
        like: &Tensor,
        dim:  i64) -> Tensor {
    
    todo!();
        /*
            return arange(like.size(dim), like.options().dtype(kLong));
        */
}
