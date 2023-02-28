crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/TestOps.cpp]

/**
  | If addends is nullopt, return values.
  | 
  | Else, return a new tensor containing
  | the elementwise sums.
  |
  */
pub fn test_optional_intlist(
        values:  &Tensor,
        addends: Option<&[i32]>) -> Tensor {
    
    todo!();
        /*
            if (!addends) {
        return values;
      }
      TORCH_CHECK(values.dim() == 1);
      Tensor output = empty_like(values);
      auto inp = values.accessor<int,1>();
      auto out = output.accessor<int,1>();
      for (const auto i : irange(values.size(0))) {
        out[i] = inp[i] + addends->at(i);
      }
      return output;
        */
}

/**
  | If addends is nullopt, return values.
  | 
  | Else, return a new tensor containing
  | the elementwise sums.
  |
  */
pub fn test_optional_floatlist(
        values:  &Tensor,
        addends: Option<&[f64]>) -> Tensor {
    
    todo!();
        /*
            if (!addends) {
        return values;
      }
      TORCH_CHECK(values.dim() == 1);
      Tensor output = empty_like(values);
      auto inp = values.accessor<float,1>();
      auto out = output.accessor<float,1>();
      for (const auto i : irange(values.size(0))) {
        out[i] = inp[i] + addends->at(i);
      }
      return output;
        */
}

/**
  | Test default strings can handle escape
  | sequences properly (although commas
  | are broken)
  |
  */
pub fn test_string_default(
        dummy: &Tensor,
        a:     StringView,
        b:     StringView) -> Tensor {
    
    todo!();
        /*
            const string_view expect = "\"'\\";
      TORCH_CHECK(a == expect, "Default A failed");
      TORCH_CHECK(b == expect, "Default B failed");
      return dummy;
        */
}


/**
  | Test that overloads with ambiguity
  | created by defaulted parameters work.
  | 
  | The operator declared first should
  | have priority always
  | 
  | Overload a
  |
  */
pub fn test_ambiguous_defaults_a(
        dummy: &Tensor,
        a:     i64,
        b:     i64) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(a == 1);
      TORCH_CHECK(b == 1);
      return scalar_to_tensor(1);
        */
}

/**
  | Overload b
  |
  */
pub fn test_ambiguous_defaults_b(
        dummy: &Tensor,
        a:     i64,
        b:     StringView) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(a == 2);
      TORCH_CHECK(b == "2");
      return scalar_to_tensor(2);
        */
}
