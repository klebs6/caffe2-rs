crate::ix!();

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ complex / polar ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn complex_check_floating(
        a: &Tensor,
        b: &Tensor)  {
    
    todo!();
        /*
            TORCH_CHECK((a.scalar_type() == kFloat || a.scalar_type() == kDouble) &&
                  (b.scalar_type() == kFloat || b.scalar_type() == kDouble),
                  "Expected both inputs to be Float or Double tensors but got ",
                  a.scalar_type(), " and ", b.scalar_type());
        */
}

pub fn complex_check_dtype(
        result: &Tensor,
        a:      &Tensor,
        b:      &Tensor)  {
    
    todo!();
        /*
            complex_check_floating(a, b);
      TORCH_CHECK(a.scalar_type() == b.scalar_type(),
                  "Expected object of scalar type ", a.scalar_type(),
                  " but got scalar type ", b.scalar_type(), " for second argument");
      TORCH_CHECK(result.scalar_type() == toComplexType(a.scalar_type()),
                  "Expected object of scalar type ", toComplexType(a.scalar_type()),
                  " but got scalar type ", result.scalar_type(),
                  " for argument 'out'");
        */
}

pub fn complex_out<'a>(
        real:   &Tensor,
        imag:   &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            complex_check_dtype(result, real, imag);
      auto iter = TensorIteratorConfig()
          .add_output(result)
          .add_input(real)
          .add_input(imag)
          .check_all_same_dtype(false)
          .build();
      complex_stub(iter.device_type(), iter);
      return result;
        */
}

pub fn complex(
        real: &Tensor,
        imag: &Tensor) -> Tensor {
    
    todo!();
        /*
            complex_check_floating(real, imag);
      TensorOptions options = real.options();
      options = options.dtype(toComplexType(real.scalar_type()));
      Tensor result = empty(0, options);
      return complex_out(result, real, imag);
        */
}


