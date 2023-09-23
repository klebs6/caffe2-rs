crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/atest.cpp]

pub struct Atest {
    base:         Test,
    x_tensor:     Tensor,
    y_tensor:     Tensor,
    x_logical:    Tensor,
    y_logical:    Tensor,
    x_float:      Tensor,
    y_float:      Tensor,
    INT:          i32, // default = 1
    FLOAT:        i32, // default = 2
    INTFLOAT:     i32, // default = 3
    INTBOOL:      i32, // default = 5
    INTBOOLFLOAT: i32, // default = 7
}

impl Atest {
    
    pub fn set_up(&mut self)  {
        
        todo!();
        /*
            x_tensor = tensor({10, -1, 0, 1, -10});
        y_tensor = tensor({-10, 1, 0, -1, 10});
        x_logical = tensor({1, 1, 0, 1, 0});
        y_logical = tensor({0, 1, 0, 1, 1});
        x_float = tensor({2.0, 2.4, 5.6, 7.0, 36.0});
        y_float = tensor({1.0, 1.1, 8.7, 10.0, 24.0});
        */
    }
}

// test dtype = kInt
pub const BINARY_OPS_KERNEL_INT_MASK: i32 = 1;

// test dtype = kFloat
pub const BINARY_OPS_KERNEL_FLOAT_MASK: i32 = 2;

// test dtype = kBool
pub const BINARY_OPS_KERNEL_BOOL_MASK: i32 = 4;

pub fn unit_binary_ops_test<T, Args>(
    func:     T,
    x_tensor: &Tensor,
    y_tensor: &Tensor,
    exp:      &Tensor,
    dtype:    ScalarType,
    args:     Args)  {

    todo!();
        /*
            auto out_tensor = empty({5}, dtype);
      func(out_tensor, x_tensor.to(dtype), y_tensor.to(dtype), args...);
      ASSERT_EQ(out_tensor.dtype(), dtype);
      if (dtype == kFloat) {
        ASSERT_TRUE(exp.to(dtype).allclose(out_tensor));
      } else {
        ASSERT_TRUE(exp.to(dtype).equal(out_tensor));
      }
        */
}

/**
  | template function for running binary
  | operator test
  | 
  | - exp: expected output
  | 
  | - func: function to be tested
  | 
  | - option: 3 bits,
  | 
  |     - 1st bit: Test op over integer tensors
  | 
  |     - 2nd bit: Test op over float tensors
  | 
  |     - 3rd bit: Test op over boolean tensors
  | 
  | For example, if function should be tested
  | over integer/boolean but not for float,
  | option will be 1 * 1 + 0 * 2 + 1 * 4 = 5. If tested
  | over all the type, option should be 7.
  |
  */
pub fn run_binary_ops_test<T, Args>(
    func:     T,
    x_tensor: &Tensor,
    y_tensor: &Tensor,
    exp:      &Tensor,
    option:   i32,
    args:     Args)  {

    todo!();
        /*
            // Test op over integer tensors
      if (option & BinaryOpsKernel::IntMask) {
        unit_binary_ops_test(func, x_tensor, y_tensor, exp, kInt, args...);
      }

      // Test op over float tensors
      if (option & BinaryOpsKernel::FloatMask) {
        unit_binary_ops_test(func, x_tensor, y_tensor, exp, kFloat, args...);
      }

      // Test op over boolean tensors
      if (option & BinaryOpsKernel::BoolMask) {
        unit_binary_ops_test(func, x_tensor, y_tensor, exp, kBool, args...);
      }
        */
}

pub fn trace()  {
    
    todo!();
        /*
            Tensor foo = rand({12, 12});

      // ASSERT foo is 2-dimensional and holds floats.
      auto foo_a = foo.accessor<float, 2>();
      float trace = 0;

      for (int i = 0; i < foo_a.size(0); i++) {
        trace += foo_a[i][i];
      }

      ASSERT_FLOAT_EQ(foo.trace().item<float>(), trace);
        */
}

#[test] fn atest_operators() {
    todo!();
    /*
    
      int a = 0b10101011;
      int b = 0b01111011;

      auto a_tensor = tensor({a});
      auto b_tensor = tensor({b});

      ASSERT_TRUE(tensor({~a}).equal(~a_tensor));
      ASSERT_TRUE(tensor({a | b}).equal(a_tensor | b_tensor));
      ASSERT_TRUE(tensor({a & b}).equal(a_tensor & b_tensor));
      ASSERT_TRUE(tensor({a ^ b}).equal(a_tensor ^ b_tensor));

    */
}


#[test] fn atest_logical_and_operators() {
    todo!();
    /*
    
      auto exp_tensor = tensor({0, 1, 0, 1, 0});
      run_binary_ops_test(
          logical_and_out, x_logical, y_logical, exp_tensor, INTBOOL);

    */
}


#[test] fn atest_logical_or_operators() {
    todo!();
    /*
    
      auto exp_tensor = tensor({1, 1, 0, 1, 1});
      run_binary_ops_test(
          logical_or_out, x_logical, y_logical, exp_tensor, INTBOOL);

    */
}


#[test] fn atest_logical_xor_operators() {
    todo!();
    /*
    
      auto exp_tensor = tensor({1, 0, 0, 0, 1});
      run_binary_ops_test(
          logical_xor_out, x_logical, y_logical, exp_tensor, INTBOOL);

    */
}


#[test] fn atest_lt_operators() {
    todo!();
    /*
    
      auto exp_tensor = tensor({0, 0, 0, 0, 1});
      run_binary_ops_test<
          Tensor& (*)(Tensor&, const Tensor&, const Tensor&)>(
          lt_out, x_logical, y_logical, exp_tensor, INTBOOL);

    */
}


#[test] fn atest_le_operators() {
    todo!();
    /*
    
      auto exp_tensor = tensor({0, 1, 1, 1, 1});
      run_binary_ops_test<
          Tensor& (*)(Tensor&, const Tensor&, const Tensor&)>(
          le_out, x_logical, y_logical, exp_tensor, INTBOOL);

    */
}


#[test] fn atest_gt_operators() {
    todo!();
    /*
    
      auto exp_tensor = tensor({1, 0, 0, 0, 0});
      run_binary_ops_test<
          Tensor& (*)(Tensor&, const Tensor&, const Tensor&)>(
          gt_out, x_logical, y_logical, exp_tensor, INTBOOL);

    */
}


#[test] fn atest_ge_operators() {
    todo!();
    /*
    
      auto exp_tensor = tensor({1, 1, 1, 1, 0});
      run_binary_ops_test<
          Tensor& (*)(Tensor&, const Tensor&, const Tensor&)>(
          ge_out, x_logical, y_logical, exp_tensor, INTBOOL);

    */
}


#[test] fn atest_eq_operators() {
    todo!();
    /*
    
      auto exp_tensor = tensor({0, 1, 1, 1, 0});
      run_binary_ops_test<
          Tensor& (*)(Tensor&, const Tensor&, const Tensor&)>(
          eq_out, x_logical, y_logical, exp_tensor, INTBOOL);

    */
}


#[test] fn atest_ne_operators() {
    todo!();
    /*
    
      auto exp_tensor = tensor({1, 0, 0, 0, 1});
      run_binary_ops_test<
          Tensor& (*)(Tensor&, const Tensor&, const Tensor&)>(
          ne_out, x_logical, y_logical, exp_tensor, INTBOOL);

    */
}


#[test] fn atest_add_operators() {
    todo!();
    /*
    
      auto exp_tensor = tensor({-10, 1, 0, -1, 10});
      run_binary_ops_test(add_out, x_tensor, y_tensor, exp_tensor, INTBOOL, 2);

    */
}


#[test] fn atest_max_operators() {
    todo!();
    /*
    
      auto exp_tensor = tensor({10, 1, 0, 1, 10});
      run_binary_ops_test<
          Tensor& (*)(Tensor&, const Tensor&, const Tensor&)>(
          max_out, x_tensor, y_tensor, exp_tensor, INTBOOLFLOAT);

    */
}


#[test] fn atest_min_operators() {
    todo!();
    /*
    
      auto exp_tensor = tensor({-10, -1, 0, -1, -10});
      run_binary_ops_test<
          Tensor& (*)(Tensor&, const Tensor&, const Tensor&)>(
          min_out, x_tensor, y_tensor, exp_tensor, INTBOOLFLOAT);

    */
}


#[test] fn atest_sigmoid_backward_operator() {
    todo!();
    /*
    
      auto exp_tensor = tensor({-1100, 0, 0, -2, 900});
      // only test with type Float
      run_binary_ops_test<
          Tensor& (*)(Tensor&, const Tensor&, const Tensor&)>(
          sigmoid_backward_out, x_tensor, y_tensor, exp_tensor, FLOAT);

    */
}


#[test] fn atest_fmod_tensor_operators() {
    todo!();
    /*
    
      auto exp_tensor = tensor({0.0, 0.2, 5.6, 7.0, 12.0});
      run_binary_ops_test<
          Tensor& (*)(Tensor&, const Tensor&, const Tensor&)>(
          fmod_out, x_float, y_float, exp_tensor, INTFLOAT);

    */
}


// TEST_CASE( "atest", "[]" ) {
#[test] fn atest() {
    todo!();
    /*
    
      manual_seed(123);

      auto foo = rand({12, 6});

      ASSERT_EQ(foo.size(0), 12);
      ASSERT_EQ(foo.size(1), 6);

      foo = foo + foo * 3;
      foo -= 4;

      Scalar a = 4;
      float b = a.to<float>();
      ASSERT_EQ(b, 4);

      foo = ((foo * foo) == (foo.pow(3))).to(kByte);
      foo = 2 + (foo + 1);
      // foo = foo[3];
      auto foo_v = foo.accessor<u8, 2>();

      for (int i = 0; i < foo_v.size(0); i++) {
        for (int j = 0; j < foo_v.size(1); j++) {
          foo_v[i][j]++;
        }
      }

      ASSERT_TRUE(foo.equal(4 * ones({12, 6}, kByte)));

      trace();

      float data[] = {1, 2, 3, 4, 5, 6};

      auto f = from_blob(data, {1, 2, 3});
      auto f_a = f.accessor<float, 3>();

      ASSERT_EQ(f_a[0][0][0], 1.0);
      ASSERT_EQ(f_a[0][1][1], 5.0);

      ASSERT_EQ(f.strides()[0], 6);
      ASSERT_EQ(f.strides()[1], 3);
      ASSERT_EQ(f.strides()[2], 1);
      ASSERT_EQ(f.sizes()[0], 1);
      ASSERT_EQ(f.sizes()[1], 2);
      ASSERT_EQ(f.sizes()[2], 3);

      // TODO(ezyang): maybe do a more precise exception type.
      ASSERT_THROW(f.resize_({3, 4, 5}), exception);
      {
        int isgone = 0;
        {
          auto f2 = from_blob(data, {1, 2, 3}, [&](void*) { isgone++; });
        }
        ASSERT_EQ(isgone, 1);
      }
      {
        int isgone = 0;
        Tensor a_view;
        {
          auto f2 = from_blob(data, {1, 2, 3}, [&](void*) { isgone++; });
          a_view = f2.view({3, 2, 1});
        }
        ASSERT_EQ(isgone, 0);
        a_view.reset();
        ASSERT_EQ(isgone, 1);
      }

      if (hasCUDA()) {
        int isgone = 0;
        {
          auto base = empty({1, 2, 3}, TensorOptions(kCUDA));
          auto f2 = from_blob(base.data_ptr(), {1, 2, 3}, [&](void*) { isgone++; });
        }
        ASSERT_EQ(isgone, 1);

        // Attempt to specify the wrong device in from_blob
        auto t = empty({1, 2, 3}, TensorOptions(kCUDA, 0));
        EXPECT_ANY_THROW(from_blob(t.data_ptr(), {1, 2, 3}, Device(kCUDA, 1)));

        // Infers the correct device
        auto t_ = from_blob(t.data_ptr(), {1, 2, 3}, kCUDA);
        ASSERT_EQ(t_.device(), Device(kCUDA, 0));
      }

    */
}
