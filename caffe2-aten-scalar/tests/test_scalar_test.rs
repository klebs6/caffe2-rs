crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/scalar_test.cpp]

pub const FLOAT: Auto = ScalarType::Float;

pub struct Foo<scalar_type> {

}

impl Foo<scalar_type> {
    
    pub fn apply(
        a: Tensor,
        b: Tensor)  {
        
        todo!();
        /*
            scalar_type s = 1;
        stringstream ss;
        ss << "hello, dispatch: " << a.toString() << s << "\n";
        auto data = (scalar_type*)a.data_ptr();
        (void)data;
        */
    }
}

pub struct FooHalf {

}

impl FooHalf {
    
    pub fn apply(
        a: Tensor,
        b: Tensor)  {
        
        todo!();
        /*
        
        */
    }
}

pub fn test_overflow()  {
    
    todo!();
        /*
            auto s1 = Scalar(M_PI);
      ASSERT_EQ(s1.toFloat(), static_cast<float>(M_PI));
      s1.toHalf();

      s1 = Scalar(100000);
      ASSERT_EQ(s1.toFloat(), 100000.0);
      ASSERT_EQ(s1.toInt(), 100000);

      ASSERT_THROW(s1.toHalf(), runtime_error);

      s1 = Scalar(NAN);
      ASSERT_TRUE(isnan(s1.toFloat()));
      ASSERT_THROW(s1.toInt(), runtime_error);

      s1 = Scalar(INFINITY);
      ASSERT_TRUE(isinf(s1.toFloat()));
      ASSERT_THROW(s1.toInt(), runtime_error);
        */
}

#[test] fn test_scalar() {
    todo!();
    /*
    
      manual_seed(123);

      Scalar what = 257;
      Scalar bar = 3.0;
      Half h = bar.toHalf();
      Scalar h2 = h;
      cout << "H2: " << h2.toDouble() << " " << what.toFloat() << " "
           << bar.toDouble() << " " << what.isIntegral(false) << "\n";
      auto gen = getDefaultCPUGenerator();
      {
        // See Note [Acquire lock when using random generators]
        lock_guard<mutex> lock(gen.mutex());
        ASSERT_NO_THROW(gen.set_current_seed(random_device()()));
      }
      auto&& C = globalContext();
      if (hasCUDA()) {
        auto t2 = zeros({4, 4}, kCUDA);
        cout << &t2 << "\n";
      }
      auto t = ones({4, 4});

      auto wha2 = zeros({4, 4}).add(t).sum();
      ASSERT_EQ(wha2.item<double>(), 16.0);

      ASSERT_EQ(t.sizes()[0], 4);
      ASSERT_EQ(t.sizes()[1], 4);
      ASSERT_EQ(t.strides()[0], 4);
      ASSERT_EQ(t.strides()[1], 1);

      TensorOptions options = dtype(kFloat);
      Tensor x = randn({1, 10}, options);
      Tensor prev_h = randn({1, 20}, options);
      Tensor W_h = randn({20, 20}, options);
      Tensor W_x = randn({20, 10}, options);
      Tensor i2h = mm(W_x, x.t());
      Tensor h2h = mm(W_h, prev_h.t());
      Tensor next_h = i2h.add(h2h);
      next_h = next_h.tanh();

      ASSERT_ANY_THROW(Tensor{}.item());

      test_overflow();

      if (hasCUDA()) {
        auto r = next_h.to(Device(kCUDA), kFloat, /*non_blocking=*/ false, /*copy=*/ true);
        ASSERT_TRUE(r.to(Device(kCPU), kFloat, /*non_blocking=*/ false, /*copy=*/ true).equal(next_h));
      }
      ASSERT_NO_THROW(randn({10, 10, 2}, options));

      // check Scalar.toTensor on Scalars backed by different data types
      ASSERT_EQ(scalar_to_tensor(bar).scalar_type(), kDouble);
      ASSERT_EQ(scalar_to_tensor(what).scalar_type(), kLong);
      ASSERT_EQ(scalar_to_tensor(ones({}).item()).scalar_type(), kDouble);

      if (x.scalar_type() != ScalarType::Half) {
        AT_DISPATCH_ALL_TYPES(x.scalar_type(), "foo", [&] {
          Scalar s = 1;
          stringstream ss;
          ASSERT_NO_THROW(
              ss << "hello, dispatch" << x.toString() << s << "\n");
          auto data = (Scalar*)x.data_ptr();
          (void)data;
        });
      }

      // test direct C-scalar type conversions
      {
        auto x = ones({1, 2}, options);
        ASSERT_ANY_THROW(x.item<float>());
      }
      auto float_one = ones({}, options);
      ASSERT_EQ(float_one.item<float>(), 1);
      ASSERT_EQ(float_one.item<i32>(), 1);
      ASSERT_EQ(float_one.item<Half>(), 1);

    */
}

#[test] fn test_scalar_conj() {
    todo!();
    /*
    
      Scalar int_scalar = 257;
      Scalar float_scalar = 3.0;
      Scalar complex_scalar = complex<double>(2.3, 3.5);

      ASSERT_EQ(int_scalar.conj().toInt(), 257);
      ASSERT_EQ(float_scalar.conj().toDouble(), 3.0);
      ASSERT_EQ(complex_scalar.conj().toComplexDouble(), complex<double>(2.3, -3.5));

    */
}

#[test] fn test_scalar_equal() {
    todo!();
    /*
    
      ASSERT_FALSE(Scalar(1.0).equal(false));
      ASSERT_FALSE(Scalar(1.0).equal(true));
      ASSERT_FALSE(Scalar(true).equal(1.0));
      ASSERT_TRUE(Scalar(true).equal(true));

      ASSERT_TRUE(Scalar(complex<double>{2.0, 5.0}).equal(complex<double>{2.0, 5.0}));
      ASSERT_TRUE(Scalar(complex<double>{2.0, 0}).equal(2.0));
      ASSERT_TRUE(Scalar(complex<double>{2.0, 0}).equal(2));

      ASSERT_TRUE(Scalar(2.0).equal(complex<double>{2.0, 0.0}));
      ASSERT_FALSE(Scalar(2.0).equal(complex<double>{2.0, 4.0}));
      ASSERT_FALSE(Scalar(2.0).equal(3.0));
      ASSERT_TRUE(Scalar(2.0).equal(2));

      ASSERT_TRUE(Scalar(2).equal(complex<double>{2.0, 0}));
      ASSERT_TRUE(Scalar(2).equal(2));
      ASSERT_TRUE(Scalar(2).equal(2.0));

    */
}

#[test] fn test_scalar_formatting() {
    todo!();
    /*
    
      auto format = [] (Scalar a) {
        ostringstream str;
        str << a;
        return str.str();
      };
      ASSERT_EQ("3", format(Scalar(3)));
      ASSERT_EQ("3.1", format(Scalar(3.1)));
      ASSERT_EQ("true", format(Scalar(true)));
      ASSERT_EQ("false", format(Scalar(false)));
      ASSERT_EQ("(2,3.1)", format(Scalar(complex<double>(2.0, 3.1))));
      ASSERT_EQ("(2,3.1)", format(Scalar(complex<float>(2.0, 3.1))));

    */
}
