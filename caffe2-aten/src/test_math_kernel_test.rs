crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/math_kernel_test.cpp]

pub fn all_close(
        t1:   &Tensor,
        t2:   &Tensor,
        rtol: f64,
        atol: f64) -> bool {

    let rtol: f64 = rtol.unwrap_or(1e-5);
    let atol: f64 = atol.unwrap_or(1e-8);

    todo!();
        /*
            if (!t1.is_same_size(t2)) {
        cerr << "Difference in tensor shapes: "
          << t1.sizes() << " v.s. " << t2.sizes() << endl;
        return false;
      }
      bool equal = t1.allclose(t2, rtol, atol);
      if (!equal) {
        cerr << "Difference in tensor value: \nFirst tensor:\n"
            << t1 << "\nSecond tensor:\n" << t2 << endl;
      }
      return equal;
        */
}

macro_rules! assert_allclose_tolerances {
    ($t1:ident, 
    $t2:ident, 
    $rtol:ident, 
    $atol:ident) => {
        /*
        
          ASSERT_TRUE(allClose(t1, t2, rtol, atol));
        */
    }
}

/**
  | Ideally we want to test both forward and
  | backward on math kernels but I haven't found an
  | easy way to do it.  Currently we only test
  | forward here and rely on backward tests of each
  | function used in math kernels. 
  */
#[test] fn math_kernel_test_native_group_norm() {
    todo!();
    /*
    
      int num_channels = 6;
      int N = 2;
      int H = 2, W = 2;
      int HxW = H * W;

      const auto input = randn({N, num_channels, H, W});
      const auto weight = randn({num_channels});
      const auto bias = randn({num_channels});
      double eps = 1e-05;
      for (bool undef_weight: {true, false}) {
        for (int num_groups: {3, 6, 1}) {
          Tensor undef;
          auto out = native::native_group_norm(
                input, undef_weight ? undef : weight, undef_weight ? undef : bias,
                N, num_channels, HxW, num_groups, eps);
          auto math_out = native::math_group_norm(
                input, undef_weight ? undef : weight, undef_weight ? undef : bias,
                N, num_channels, HxW, num_groups, eps);
          ASSERT_ALLCLOSE_TOLERANCES(get<0>(out), get<0>(math_out), 1e-4, 1e-6);
          ASSERT_ALLCLOSE_TOLERANCES(get<1>(out), get<1>(math_out), 1e-4, 1e-6);
          ASSERT_ALLCLOSE_TOLERANCES(get<2>(out), get<2>(math_out), 1e-4, 1e-6);
        }
      }

    */
}

#[test] fn math_kernel_test_native_layer_norm() {
    todo!();
    /*
    
      const auto input = rand({20, 10, 10, 10});
      const auto input_shape = input.sizes();
      const auto input_ndim = input.dim();

      double eps = 1e-05;
      for (bool undef_weight: {true, false}) {
        for (int normalized_size: {2, 3}) {
          Tensor undef;
          vector<i64> normalized_shape(normalized_size, 10);
          const auto weight = rand(normalized_shape);
          const auto bias = rand(normalized_shape);

          auto out = native_layer_norm(
                input, normalized_shape, undef_weight ? undef : weight, undef_weight ? undef : bias,
                eps);
          auto math_out = native::math_native_layer_norm(
                input, normalized_shape, undef_weight ? undef : weight, undef_weight ? undef : bias,
                eps);
          ASSERT_ALLCLOSE_TOLERANCES(get<0>(out), get<0>(math_out), 1e-3, 1e-5);
          ASSERT_ALLCLOSE_TOLERANCES(get<1>(out), get<1>(math_out), 1e-3, 1e-5);
          ASSERT_ALLCLOSE_TOLERANCES(get<2>(out), get<2>(math_out), 1e-3, 1e-5);
        }
      }

    */
}

#[test] fn math_kernel_test_addr() {
    todo!();
    /*
    
      const auto vec1 = arange(1., 4.);
      const auto vec2 = arange(1., 3.);
      const auto M = zeros({3, 2});

      for (float beta: {1., 1.2, 0.}) {
        // nans and infs are not propagated to the output when beta == 0
        if (beta == 0) {
          M[0][0] = numeric_limits<float>::infinity();
          M[2][0] = numeric_limits<float>::quiet_NaN();
        }
        for (float alpha: {1., 2., 0.}) {
          auto out = native::addr(M, vec1, vec2, beta, alpha);
          auto math_out = native::math_addr(M, vec1, vec2, beta, alpha);
          ASSERT_ALLCLOSE_TOLERANCES(out, math_out, 1e-4, 1e-6);
        }
      }

    */
}

#[test] fn math_kernel_test_silu_backward() {
    todo!();
    /*
    
      const auto input = rand({20, 10});
      const auto grad_output = rand({20, 10});
      auto out = native::silu_backward(grad_output, input);
      auto math_out = native::math_silu_backward(grad_output, input);
      ASSERT_ALLCLOSE_TOLERANCES(out, math_out, 1e-4, 1e-6);

    */
}

#[test] fn math_kernel_test_mish_backward() {
    todo!();
    /*
    
      const auto input = rand({20, 10});
      const auto grad_output = rand({20, 10});
      auto out = native::mish_backward(grad_output, input);
      auto math_out = native::math_mish_backward(grad_output, input);
      ASSERT_ALLCLOSE_TOLERANCES(out, math_out, 1e-4, 1e-6);

    */
}

#[test] fn math_kernel_test_narrow_copy() {
    todo!();
    /*
    
      auto x = rand({5, 8, 7});
      for (i64 dim = 0; dim < 3; ++dim) {
        const i64 start = 1, length = 4;
        auto y_ref = x.narrow(dim, start, length);
        auto y_test = native::narrow_copy_dense(x, dim, start, length);
        ASSERT_ALLCLOSE_TOLERANCES(y_ref, y_test, 0, 0);
      }

    */
}

#[test] fn math_kernel_test_bmm() {
    todo!();
    /*
    
      auto test_bmm = [](i64 last_dim) {
        auto x = rand({1, 4, 4}, kFloat);
        auto y = rand({1, 4, last_dim}, kDouble);
        EXPECT_THROW(auto z = bmm(x, y), exception);
      };

      test_bmm(5);
      test_bmm(1000);

    */
}
