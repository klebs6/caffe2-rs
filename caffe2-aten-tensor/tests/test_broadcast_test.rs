crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/broadcast_test.cpp]

/// can't expand empty tensor
pub fn test_empty_tensor(T: &mut DeprecatedTypeProperties)  {
    
    todo!();
        /*
            auto empty = randn({0}, T);
      ASSERT_ANY_THROW(empty.expand({3}));
        */
}

/// out-place function with 2 args
pub fn test_out_2basic(T: &mut DeprecatedTypeProperties)  {
    
    todo!();
        /*
            auto a = randn({3, 1}, T);
      auto b = randn({5}, T);
      vector<i64> expanded_sizes = {3, 5};
      ASSERT_TRUE(
          (a + b).equal(a.expand(expanded_sizes) + b.expand(expanded_sizes)));
        */
}

/// with scalar
pub fn test_out_2with_scalar(T: &mut DeprecatedTypeProperties)  {
    
    todo!();
        /*
            auto aScalar = ones({}, T);
      auto b = randn({3, 5}, T);
      ASSERT_TRUE(
          (aScalar + b).equal(aScalar.expand(b.sizes()) + b.expand(b.sizes())));
        */
}

/// old fallback behavior yields error
pub fn test_out_2old_fallback(T: &mut DeprecatedTypeProperties)  {
    
    todo!();
        /*
            auto a = randn({3, 5}, T);
      auto b = randn({5, 3}, T);
      ASSERT_ANY_THROW(a + b);
        */
}

/// with mismatched sizes
pub fn test_out_2mismatched_sizes(T: &mut DeprecatedTypeProperties)  {
    
    todo!();
        /*
            auto a = randn({3, 5}, T);
      auto b = randn({7, 5}, T);
      ASSERT_ANY_THROW(a + b);
        */
}

/// out-place function with 3 args
pub fn test_out_3basic(T: &mut DeprecatedTypeProperties)  {
    
    todo!();
        /*
            auto a = randn({3, 1, 1}, T);
      auto b = randn({1, 2, 1}, T);
      auto c = randn({1, 1, 5}, T);
      vector<i64> expanded_sizes = {3, 2, 5};
      ASSERT_TRUE((a + b + c).equal(
          a.expand(expanded_sizes) + b.expand(expanded_sizes) +
          c.expand(expanded_sizes)));
        */
}

/// with scalar
pub fn test_out_3with_scalar(T: &mut DeprecatedTypeProperties)  {
    
    todo!();
        /*
            auto aTensorScalar = ones({}, T);
      auto b = randn({3, 2, 1}, T);
      auto c = randn({1, 2, 5}, T);
      vector<i64> expanded_sizes = {3, 2, 5};
      ASSERT_TRUE(aTensorScalar.addcmul(b, c).equal(
          aTensorScalar.expand(expanded_sizes)
              .addcmul(b.expand(expanded_sizes), c.expand(expanded_sizes))));
        */
}

/// old fallback behavior yields error
pub fn test_out_3old_fallback(T: &mut DeprecatedTypeProperties)  {
    
    todo!();
        /*
            auto a = randn({3, 2, 5}, T);
      auto b = randn({2, 3, 5}, T);
      auto c = randn({5, 3, 2}, T);
      ASSERT_ANY_THROW(a.addcmul(b, c));
        */
}

/// with mismatched sizes
pub fn test_out_3mismatched_sizes(T: &mut DeprecatedTypeProperties)  {
    
    todo!();
        /*
            auto a = randn({3, 2, 5}, T);
      auto b = randn({2, 3, 5}, T);
      auto c = randn({5, 5, 5}, T);
      ASSERT_ANY_THROW(a.addcmul(b, c));
        */
}

/**
  | in-place function with 2 args
  |
  */
pub fn test_in_2basic(T: &mut DeprecatedTypeProperties)  {
    
    todo!();
        /*
            auto a = randn({3, 5}, T);
      auto b = randn({3, 1}, T);
      ASSERT_TRUE((a + b).equal(a + b.expand({3, 5})));
        */
}

/**
  | with scalar
  |
  */
pub fn test_in_2with_scalar(T: &mut DeprecatedTypeProperties)  {
    
    todo!();
        /*
            auto a = randn({3, 5}, T);
      auto bScalar = ones({}, T);
      ASSERT_TRUE((a + bScalar).equal(a + bScalar.expand(a.sizes())));
        */
}

/**
  | error: would have to expand inplace
  | arg
  |
  */
pub fn test_in_2expand_error(T: &mut DeprecatedTypeProperties)  {
    
    todo!();
        /*
            auto a = randn({1, 5}, T);
      auto b = randn({3, 1}, T);
      ASSERT_ANY_THROW(a.add_(b));
        */
}

/**
  | in-place function with 3 args
  |
  */
pub fn test_in_3basic(T: &mut DeprecatedTypeProperties)  {
    
    todo!();
        /*
            auto a = randn({3, 5, 2}, T);
      auto b = randn({3, 1, 2}, T);
      auto c = randn({1, 5, 1}, T);
      auto aClone = a.clone();
      ASSERT_TRUE(a.addcmul_(b, c).equal(
          aClone.addcmul_(b.expand(a.sizes()), c.expand(a.sizes()))));
        */
}


/**
  | with scalar
  |
  */
pub fn test_in_3with_scalar(T: &mut DeprecatedTypeProperties)  {
    
    todo!();
        /*
            auto a = randn({3, 5, 2}, T);
      auto b = randn({3, 1, 2}, T);
      auto c = randn({1, 5, 1}, T);
      auto aClone = a.clone();
      auto bScalar = ones({}, T);
      ASSERT_TRUE(a.addcmul_(bScalar, c)
                      .equal(aClone.addcmul_(
                          bScalar.expand(a.sizes()), c.expand(a.sizes()))));
        */
}


/**
  | error: would have to expand inplace
  | arg
  |
  */
pub fn test_in_3expand_error(T: &mut DeprecatedTypeProperties)  {
    
    todo!();
        /*
            auto a = randn({1, 3, 5}, T);
      auto b = randn({4, 1, 1}, T);
      auto c = randn({1, 3, 1}, T);
      ASSERT_ANY_THROW(a.addcmul_(b, c));
        */
}

/**
  | explicit dim specification
  |
  */
pub fn test_explicit_dim_basic(T: &mut DeprecatedTypeProperties)  {
    
    todo!();
        /*
            auto a = randn({1}, T);
      auto b = randn({5, 3}, T);
      auto c = randn({3, 7}, T);
      ASSERT_TRUE(a.addmm(b, c).equal(a.expand({5, 7}).addmm(b, c)));
        */
}

/**
  | with scalar
  |
  */
pub fn test_explicit_dim_with_scalar(T: &mut DeprecatedTypeProperties)  {
    
    todo!();
        /*
            auto a = randn({1}, T);
      auto b = randn({5, 3}, T);
      auto c = randn({3, 7}, T);
      Tensor aScalar = ones({}, T);
      ASSERT_TRUE(aScalar.addmm(b, c).equal(aScalar.expand({5, 7}).addmm(b, c)));
        */
}

/**
  | with mismatched sizes
  |
  */
pub fn test_explicit_dim_with_mismatched_sizes(T: &mut DeprecatedTypeProperties)  {
    
    todo!();
        /*
            auto b = randn({5, 3}, T);
      auto c = randn({3, 7}, T);
      auto a = randn({3, 3}, T);
      ASSERT_ANY_THROW(a.addmm(b, c));
        */
}

#[test] fn broadcast_test() {
    todo!();
    /*
    
      manual_seed(123);
      DeprecatedTypeProperties& T = CPU(kFloat);

      TestEmptyTensor(T);

      TestOut2Basic(T);
      TestOut2WithScalar(T);
      TestOut2OldFallback(T);
      TestOut2MismatchedSizes(T);

      TestOut3Basic(T);
      TestOut3WithScalar(T);
      TestOut3OldFallback(T);
      TestOut3MismatchedSizes(T);

      TestIn2Basic(T);
      TestIn2WithScalar(T);
      TestIn2ExpandError(T);

      TestIn3Basic(T);
      TestIn3WithScalar(T);
      TestIn3ExpandError(T);

      TestExplicitDimBasic(T);
      TestExplicitDimWithScalar(T);
      TestExplicitDimWithMismatchedSizes(T);

    */
}
