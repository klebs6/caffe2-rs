crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/wrapdim_test.cpp]

pub fn test_simple_case(T: &mut DeprecatedTypeProperties)  {
    
    todo!();
        /*
            auto a = randn({2, 3, 4, 5}, T);
      ASSERT_TRUE(a.prod(-4).equal(a.prod(0)));
      ASSERT_TRUE(a.prod(3).equal(a.prod(-1)));
        */
}

pub fn test_expression_specification(T: &mut DeprecatedTypeProperties)  {
    
    todo!();
        /*
            auto a = randn({2, 3, 4, 5}, T);
      ASSERT_TRUE(a.unsqueeze(-5).equal(a.unsqueeze(0)));
      ASSERT_TRUE(a.unsqueeze(4).equal(a.unsqueeze(-1)));

      // can unsqueeze scalar
      auto b = randn({}, T);
      ASSERT_TRUE(b.unsqueeze(0).equal(b.unsqueeze(-1)));
        */
}

pub fn test_empty_tensor(T: &mut DeprecatedTypeProperties)  {
    
    todo!();
        /*
            auto a = randn(0, T);
      ASSERT_TRUE(a.prod(0).equal(ones({}, T)));
        */
}

pub fn test_scalar_vs_1dim_1size(T: &mut DeprecatedTypeProperties)  {
    
    todo!();
        /*
            auto a = randn(1, T);
      ASSERT_TRUE(a.prod(0).equal(a.prod(-1)));
      a.resize_({});
      ASSERT_EQ(a.dim(), 0);
      ASSERT_TRUE(a.prod(0).equal(a.prod(-1)));
        */
}

#[test] fn test_wrapdim() {
    todo!();
    /*
    
      manual_seed(123);
      DeprecatedTypeProperties& T = CPU(kFloat);

      TestSimpleCase(T);
      TestEmptyTensor(T);
      TestScalarVs1Dim1Size(T);
      TestExpressionSpecification(T);

    */
}
