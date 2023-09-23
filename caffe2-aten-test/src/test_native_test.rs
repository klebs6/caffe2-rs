crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/native_test.cpp]

pub fn require_equal_tensor_list(
    t1: &[Tensor],
    t2: &[Tensor])  {
    
    todo!();
        /*
            ASSERT_EQ(t1.size(), t2.size());
      for (usize i = 0; i < t1.size(); ++i) {
        ASSERT_EQUAL(t1[i], t2[i]);
      }
        */
}

/// split: test method, namespace give same result
pub fn test_split(
        T: TensorOptions,
        t: &mut Tensor)  {
    
    todo!();
        /*
            auto splitMethod = t.split(1, 0);
      auto splitNs = split(t, 1, 0);
      requireEqualTensorList(splitMethod, splitNs);

      // test rebuilding with cat
      ASSERT_EQUAL(cat(splitMethod, 0), t);
        */
}

/// chunk: test method, namespace give same result
///
pub fn test_chunk(
    T: TensorOptions,
    t: &mut Tensor)  {

    todo!();
        /*
            // test method, type, namespace give same result
      auto chunkMethod = t.chunk(3, 0);
      auto chunkNs = chunk(t, 3, 0);
      requireEqualTensorList(chunkMethod, chunkNs);

      // test rebuilding with cat
      ASSERT_EQUAL(cat(chunkMethod, 0), t);
        */
}

pub type StackFunc = fn(_0: &[Tensor], _1: i64) -> Tensor;

/// helper function for TestStack
pub fn test_stack_a(
        inputs:     &[Tensor],
        dim:        i64,
        stack_func: StackFunc)  {
    
    todo!();
        /*
            auto const &x = inputs[0];

      auto res = stack_func(inputs, dim);
      auto res_neg = stack_func(inputs, dim - x.dim() - 1);
      vector<i64> expected_size;
      expected_size.insert(
          expected_size.end(), x.sizes().begin(), x.sizes().begin() + dim);
      expected_size.insert(expected_size.end(), inputs.size());
      expected_size.insert(
          expected_size.end(), x.sizes().begin() + dim, x.sizes().end());

      ASSERT_EQUAL(res, res_neg);
      ASSERT_TRUE(res.sizes().equals(expected_size));

      int d = 0;
      for (auto& t : inputs) {
        ASSERT_EQUAL(res.select(dim, d), t);
        d++;
      }
        */
}

pub fn test_stack_b(
    T: TensorOptions,
    t: &mut Tensor)  {
    
    todo!();
        /*
            { // stack
        auto x = rand({2, 3, 4});
        auto y = rand({2, 3, 4});
        auto z = rand({2, 3, 4});

        auto inputs = {x, y, z};
        for (i64 dim = 0; dim < 4; ++dim) {
          _test_stack(inputs, dim, stack);
        }
      }

      { // native::_stack
        auto x = rand({2, 3, 4});
        auto y = rand({2, 3, 4});
        auto z = rand({2, 3, 4});

        auto inputs = {x, y, z};
        for (i64 dim = 0; dim < 4; ++dim) {
          _test_stack(inputs, dim, native::_stack);
        }
      }

      { // native::_stack_cpu
        auto x = rand({2, 3, 4});
        auto y = rand({2, 3, 4});
        auto z = rand({2, 3, 4});

        auto inputs = {x, y, z};
        for (i64 dim = 0; dim < 4; ++dim) {
          _test_stack(inputs, dim, native::_stack_cpu);
        }
      }
        */
}

/// size / stride
pub fn test_size(
        T: TensorOptions,
        t: &mut Tensor)  {
    
    todo!();
        /*
            auto scalar = randn({}, T);
      // Throw StartsWith("dimension specified as 0 but tensor has no dimensions")
      ASSERT_ANY_THROW(scalar.size(0));
      // Throw StartsWith("dimension specified as -1 but tensor has no dimensions")
      ASSERT_ANY_THROW(scalar.size(-1));
      // Throw StartsWith("dimension specified as 0 but tensor has no dimensions")
      ASSERT_ANY_THROW(scalar.stride(0));
      // Throw StartsWith("dimension specified as -1 but tensor has no dimensions")
      ASSERT_ANY_THROW(scalar.stride(-1));

      auto empty = randn({0}, T);
      ASSERT_EQ(empty.size(0), 0);
      ASSERT_EQ(empty.size(-1), 0);
      ASSERT_EQ(empty.stride(0), 1);
      ASSERT_EQ(empty.stride(-1), 1);
        */
}

pub fn test_matmul(
    T:    TensorOptions,
    t:    &mut Tensor,
    acct: TensorOptions)  {
    
    todo!();
        /*
            auto scalar = randn({}, T);
      auto d1 = randn({3}, T);
      auto d2 = randn({2, 3}, T);

      // 0-d
      // Throw StartsWith("both arguments to matmul need to be at least 1D")
      ASSERT_ANY_THROW(scalar.matmul(d2));
      // Throw StartsWith("both arguments to matmul need to be at least 1D")
      ASSERT_ANY_THROW(d2.matmul(scalar));

      // 1-d
      ASSERT_ALLCLOSE(d1.matmul(d1), d1.dot(d1));
      ASSERT_ALLCLOSE(d2.matmul(d1), d2.mv(d1));
      auto d1o = randn({2}, T);
      ASSERT_ALLCLOSE(d1o.matmul(d2), d1o.unsqueeze(0).mm(d2).squeeze(0));

      // 2-d
      auto d2o = randn({3, 5}, T);
      ASSERT_ALLCLOSE(d2.matmul(d2o), d2.mm(d2o));

      // > 2-d, 1-d
      auto d3 = randn({5, 2, 3}, T);
      ASSERT_ALLCLOSE(
          d3.matmul(d1), d3.bmm(d1.view({1, 3, 1}).expand({5, 3, 1})).view({5, 2}));
      ASSERT_ALLCLOSE(d1o.matmul(d3), d1o.expand({5, 1, 2}).bmm(d3).view({5, 3}));

      auto d5 = randn({3, 2, 4, 2, 3}, T);
      ASSERT_ALLCLOSE(
          d5.matmul(d1),
          d5.view({24, 2, 3})
              .bmm(d1.view({1, 3, 1}).expand({24, 3, 1}))
              .view({3, 2, 4, 2}));
      ASSERT_ALLCLOSE(
          d1o.matmul(d5),
          d1o.expand({24, 1, 2}).bmm(d5.view({24, 2, 3})).view({3, 2, 4, 3}));

      // > 2-d, 2-d
      // we use a "folding" algorithm in this case of matmul, so the direct
      // comparison to bmm doesn't work; instead, compare to the higher precision
      // computation (technically, we should always do this). Tolerances are
      // selected empirically.
      double atol = 1e-04;
      double rtol = 1e-06;
      d2 = randn({3, 4}, T);
      d2o = randn({4, 2}, T);
      auto result = d5.matmul(d2).to(AccT);

      auto d5Acc = d5.to(AccT);
      auto d2Acc = d2.to(AccT);
      auto acc_result = d5Acc.view({24, 2, 3})
                            .bmm(d2Acc.expand({24, 3, 4}))
                            .view({3, 2, 4, 2, 4});
      ASSERT_ALLCLOSE_TOLERANCES(result, acc_result, atol, rtol);
      ASSERT_ALLCLOSE(
          d2o.matmul(d5),
          d2o.expand({24, 4, 2}).bmm(d5.view({24, 2, 3})).view({3, 2, 4, 4, 3}));

      // > 2-d, > 2-d
      auto d5o = randn({2, 1, 2, 4, 3, 2}, T);
      auto d5_bmm_view =
          d5.expand({2, 3, 2, 4, 2, 3}).contiguous().view({48, 2, 3});
      auto d5o_bmm_view =
          d5o.expand({2, 3, 2, 4, 3, 2}).contiguous().view({48, 3, 2});
      ASSERT_ALLCLOSE(
          d5.matmul(d5o), d5_bmm_view.bmm(d5o_bmm_view).view({2, 3, 2, 4, 2, 2}));

      // non-expandable case
      auto d5wrong = randn({2, 4, 2, 4, 3, 2}, T);
      // Throw Contains("must match the size")
      ASSERT_ANY_THROW(d5.matmul(d5wrong));
        */
}

pub fn test_standard_gamma_grad(
    T: TensorOptions,
    t: &mut Tensor)  {
    
    todo!();
        /*
            // check empty
      auto empty = ones({0}, T);
      ASSERT_EQUAL(empty, _standard_gamma_grad(empty, empty));

      // check scalar equals one element
      auto one_scalar = ones({}, T).mul(5);
      auto one_with_dim = ones({1}, T).mul(5);
      ASSERT_ALLCLOSE(
          _standard_gamma_grad(one_scalar, one_scalar),
          _standard_gamma_grad(one_with_dim, one_with_dim).sum());

      // check mixing types
      auto t1 = randn({3, 4}, T);
      auto t2 = randn({3, 4}, T).toType(kDouble);
      // Throw StartsWith("expected scalar type")
      ASSERT_ANY_THROW(_standard_gamma_grad(t1, t2));
        */
}

pub fn test_where(
        T: TensorOptions,
        t: &mut Tensor)  {
    
    todo!();
        /*
            // empty
      auto empty = ones({0}, T);
      auto bT = T.dtype(kByte);
      auto empty_byte = ones({0}, bT);
      ASSERT_EQUAL(empty, where(empty_byte, empty, empty));

      // check scalar equals one element
      auto x_scalar = ones({}, T).mul(5);
      auto y_scalar = ones({}, T).mul(7);
      auto cond_scalar = zeros({}, bT);
      auto x_1d = x_scalar.unsqueeze(0);
      auto y_1d = y_scalar.unsqueeze(0);
      auto cond_1d = cond_scalar.unsqueeze(0);
      ASSERT_ALLCLOSE(
          where(cond_scalar, x_scalar, y_scalar).unsqueeze(0),
          where(cond_1d, x_1d, y_1d));
        */
}

pub fn test(
        T:    TensorOptions,
        acct: TensorOptions)  {
    
    todo!();
        /*
            auto t = randn({3, 3}, T);
      TestSplit(T, t);
      TestChunk(T, t);
      TestStack(T, t);
      TestSize(T, t);
      TestMatmul(T, t, AccT);
      TestStandardGammaGrad(T, t);
      TestWhere(T, t);
        */
}

#[test] fn test_native_cpu() {
    todo!();
    /*
    
      manual_seed(123);

      test(device(kCPU).dtype(kFloat),
           device(kCPU).dtype(kDouble));

    */
}

#[test] fn test_native_gpu() {
    todo!();
    /*
    
      manual_seed(123);

      if (hasCUDA()) {
        test(device(kCUDA).dtype(kFloat),
             device(kCUDA).dtype(kDouble));
      }

    */
}
