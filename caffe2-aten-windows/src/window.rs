crate::ix!();

pub fn window_function_checks(
    function_name: *const u8,
    options:       &TensorOptions,
    window_length: i64

) {

    todo!();
        /*
            TORCH_CHECK(
          options.layout() != kSparse,
          function_name,
          " is not implemented for sparse types, got: ",
          options);
      TORCH_CHECK(
          isFloatingType(typeMetaToScalarType(options.dtype())) || isComplexType(typeMetaToScalarType(options.dtype())),
          function_name,
          " expects floating point dtypes, got: ",
          options);
      TORCH_CHECK(
          window_length >= 0,
          function_name,
          " requires non-negative window_length, got window_length=",
          window_length);
        */
}
