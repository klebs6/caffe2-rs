crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/apply_utils_test.cpp]

pub fn fill_tensor(
        scalar: i64,
        t:      &mut Tensor)  {
    
    todo!();
        /*
            auto t = t_.view(-1);
      for (i64 i = 0; i < t.numel(); i++) {
        t[i] = (i + 1) * scalar;
      }
        */
}

/**
  | This test exercises all sequential applyX
  | functions. Given a shape and two transpose
  | dimensions we create 5 tensors (a0, ..., a4) of
  | the given shape and transpose the dimension
  | a with b for each tensor.
  |
  | Then we call the applyX function on each
  | floating type. a4 is allocated in doubles only,
  | whereas a0, ..., a3 are allocated in the given
  | type. For each applyX function we once write
  | the same type as we read (using a0, ..., aX-1)
  | and we once write to double (using a4 as
  | a target).
  |
  | We also exercise on a zero_dim and empty
  | tensor.
  |
  */
pub fn test(
        ty:    &mut DeprecatedTypeProperties,
        shape: &[i32],
        a:     i64,
        b:     i64)  {

    let a: i64 = a.unwrap_or(0);
    let b: i64 = b.unwrap_or(1);

    todo!();
        /*
            auto zero_dim = empty({}, type);
      zero_dim.fill_(2);
      zero_dim.exp_();
      AT_DISPATCH_FLOATING_TYPES(zero_dim.scalar_type(), "test0", [&] {
        ASSERT(zero_dim.data_ptr<Scalar>()[0] == exp(2));
      });

      auto empty_t = empty({0}, type);
      empty_t.fill_(3);
      empty_t.exp_();

      auto a0 = empty({0}, type.options());
      auto a1 = empty({0}, type.options());
      auto a2 = empty({0}, type.options());
      auto a3 = empty({0}, type.options());
      auto a4 = empty({0}, TensorOptions(kCPU).dtype(kDouble));

      vector<Tensor> tensors({a0, a1, a2, a3, a4});
      for (usize i = 0; i < tensors.size(); i++) {
        tensors[i].resize_(shape);
        fill_tensor(i + 1, tensors[i]);
        if (a >= 0 && b >= 0) {
          tensors[i].transpose_(a, b);
        }
      }

      AT_DISPATCH_FLOATING_TYPES(a0.scalar_type(), "test1", [&] {
        CPU_tensor_apply2<Scalar, Scalar>(
            a0, a1, [](Scalar& y, const Scalar& x) { y = x * x; });
        CPU_tensor_apply2<double, Scalar>(
            a4, a1, [](double& y, Scalar x) { y = (double)(x * x); });
        for (i64 i = 0; i < a0.numel(); i++) {
          auto target = a1.data_ptr<Scalar>()[i] * a1.data_ptr<Scalar>()[i];
          ASSERT(a0.data_ptr<Scalar>()[i] == target);
          ASSERT(a4.data_ptr<double>()[i] == target);
        }
      });

      AT_DISPATCH_FLOATING_TYPES(a0.scalar_type(), "test2", [&] {
        CPU_tensor_apply3<Scalar, Scalar, Scalar>(
            a0, a1, a2, [](Scalar& y, const Scalar& x, const Scalar& z) {
              y = x * x + z;
            });
        CPU_tensor_apply3<double, Scalar, Scalar>(
            a4, a1, a2, [](double& y, const Scalar& x, const Scalar& z) {
              y = (double)(x * x + z);
            });
        for (i64 i = 0; i < a0.numel(); i++) {
          auto target = a1.data_ptr<Scalar>()[i] * a1.data_ptr<Scalar>()[i];
          target = target + a2.data_ptr<Scalar>()[i];
          ASSERT(a0.data_ptr<Scalar>()[i] == target);
          ASSERT(a4.data_ptr<double>()[i] == target);
        }
      });

      AT_DISPATCH_FLOATING_TYPES(a0.scalar_type(), "test3", [&] {
        CPU_tensor_apply4<Scalar, Scalar, Scalar, Scalar>(
            a0,
            a1,
            a2,
            a3,
            [](Scalar& y,
               const Scalar& x,
               const Scalar& z,
               const Scalar& a) { y = x * x + z * a; });
        CPU_tensor_apply4<double, Scalar, Scalar, Scalar>(
            a4,
            a1,
            a2,
            a3,
            [](double& y, const Scalar& x, const Scalar& z, const Scalar& a) {
              y = (double)(x * x + z * a);
            });
        for (i64 i = 0; i < a0.numel(); i++) {
          auto target = a1.data_ptr<Scalar>()[i] * a1.data_ptr<Scalar>()[i];
          target = target + a2.data_ptr<Scalar>()[i] * a3.data_ptr<Scalar>()[i];
          ASSERT(a0.data_ptr<Scalar>()[i] == target);
          ASSERT(a4.data_ptr<double>()[i] == target);
        }
      });
        */
}

// apply utils test 2-dim small contiguous
#[test] fn apply_utils_test_contiguous2d() {
    todo!();
    /*
    
      manual_seed(123);
      test(CPU(kDouble), {2, 1}, -1, -1);

    */
}

// apply utils test 2-dim small
#[test] fn apply_utils_test_small2d() {
    todo!();
    /*
    
      manual_seed(123);
      test(CPU(kDouble), {2, 1});

    */
}

// apply utils test 2-dim
#[test] fn apply_utils_test_2d() {
    todo!();
    /*
    
      manual_seed(123);
      test(CPU(kDouble), {20, 10});

    */
}

// apply utils test 3-dim
#[test] fn apply_utils_test_3d() {
    todo!();
    /*
    
      manual_seed(123);
      test(CPU(kDouble), {3, 4, 2});

    */
}

// apply utils test 3-dim medium
#[test] fn apply_utils_test_medium3d() {
    todo!();
    /*
    
      manual_seed(123);
      test(CPU(kDouble), {3, 40, 2});

    */
}

// apply utils test 10-dim
#[test] fn apply_utils_test_10d() {
    todo!();
    /*
    
      manual_seed(123);
      test(CPU(kDouble), {3, 4, 2, 5, 2, 1, 3, 4, 2, 3});

    */
}
