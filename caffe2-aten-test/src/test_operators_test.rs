crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/operators_test.cpp]

pub fn pass_through_wrapper<F, const Func: F, Output, Args>(args: Args) -> Output {

    todo!();
        /*
            return Func(forward<Args>(args)...);
        */
}

#[test] fn operators_test_function_decltype() {
    todo!();
    /*
    
      Tensor a = randn({5, 5});
      Tensor b = randn({5, 5});
      auto expected = a * b;

      auto result = pass_through_wrapper<
        decltype(&ATEN_FN2(mul, Tensor)), &ATEN_FN2(mul, Tensor),
        Tensor, const Tensor&, const Tensor&>(a, b);
      ASSERT_TRUE(allclose(result, a * b));

    */
}

#[test] fn operators_test_method_only_decltype() {
    todo!();
    /*
    
      Tensor a = randn({5, 5});
      Tensor b = randn({5, 5});
      auto expected = a * b;

      // NB: add_ overloads are guaranteed to be method-only
      // because that is how the tensor API works.
      auto& result = pass_through_wrapper<
        decltype(&ATEN_FN2(mul_, Tensor)), &ATEN_FN2(mul_, Tensor),
        Tensor&, Tensor&, const Tensor&>(a, b);
      ASSERT_TRUE(allclose(result, expected));

    */
}

#[test] fn operators_test_aten_fn() {
    todo!();
    /*
    
      Tensor a = rand({5, 5});

      auto result = pass_through_wrapper<
        decltype(&ATEN_FN(sin)), &ATEN_FN(sin),
        Tensor, const Tensor&>(a);
      ASSERT_TRUE(allclose(result, a.sin()));

    */
}

#[test] fn operators_test_out_variant_is_faithful() {
    todo!();
    /*
    
      Tensor a = rand({5, 5});
      Tensor b = empty({5, 5});

      auto& result = pass_through_wrapper<
        decltype(&ATEN_FN2(sin, out)), &ATEN_FN2(sin, out),
        Tensor&, const Tensor&, Tensor&>(a, b);
      ASSERT_TRUE(allclose(result, a.sin()));

    */
}
