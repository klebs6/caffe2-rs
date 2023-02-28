crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/cpu_rng_test.cpp]

#[cfg(not(ATEN_CPU_STATIC_DISPATCH))]
mod aten_cpu_static_dispatch {

    use super::*;

    pub const K_CUSTOM_RNG: Auto = DispatchKey::CustomRNGKeyId;

    pub struct TestCPUGenerator {
        base:                      GeneratorImpl,
        value:                     u64,
        next_float_normal_sample:  Option<f32>,
        next_double_normal_sample: Option<f64>,
    }

    impl TestCPUGenerator {
        
        pub fn new(value: u64) -> Self {
        
            todo!();
            /*


                : GeneratorImpl{Device(DeviceType_CPU), DispatchKeySet(kCustomRNG)}, value_(value)
            */
        }
        
        pub fn random(&mut self) -> u32 {
            
            todo!();
            /*
                return value_;
            */
        }
        
        pub fn random64(&mut self) -> u64 {
            
            todo!();
            /*
                return value_;
            */
        }
        
        pub fn next_float_normal_sample(&mut self) -> Option<f32> {
            
            todo!();
            /*
                return next_float_normal_sample_;
            */
        }
        
        pub fn next_double_normal_sample(&mut self) -> Option<f64> {
            
            todo!();
            /*
                return next_double_normal_sample_;
            */
        }
        
        pub fn set_next_float_normal_sample(&mut self, randn: Option<f32>)  {
            
            todo!();
            /*
                next_float_normal_sample_ = randn;
            */
        }
        
        pub fn set_next_double_normal_sample(&mut self, randn: Option<f64>)  {
            
            todo!();
            /*
                next_double_normal_sample_ = randn;
            */
        }
        
        pub fn set_current_seed(&mut self, seed: u64)  {
            
            todo!();
            /*
                throw runtime_error("not implemented");
            */
        }
        
        pub fn current_seed(&self) -> u64 {
            
            todo!();
            /*
                throw runtime_error("not implemented");
            */
        }
        
        pub fn seed(&mut self) -> u64 {
            
            todo!();
            /*
                throw runtime_error("not implemented");
            */
        }
        
        pub fn set_state(&mut self, new_state: &TensorImpl)  {
            
            todo!();
            /*
                throw runtime_error("not implemented");
            */
        }
        
        pub fn get_state(&self) -> IntrusivePtr<TensorImpl> {
            
            todo!();
            /*
                throw runtime_error("not implemented");
            */
        }
        
        pub fn clone_impl(&self) -> *mut TestCPUGenerator {
            
            todo!();
            /*
                throw runtime_error("not implemented");
            */
        }
        
        pub fn device_type() -> DeviceType {
            
            todo!();
            /*
                return DeviceType_CPU;
            */
        }
    }


    // ==================================================== Random ========================================================

    pub fn random(
            self_:     &mut Tensor,
            generator: Option<Generator>) -> &mut Tensor {
        
        todo!();
            /*
                return native::templates::random_impl<native::templates::cpu::RandomKernel, TestCPUGenerator>(self, generator);
            */
    }

    pub fn random_from_to(
            self_:     &mut Tensor,
            from:      i64,
            to:        Option<i64>,
            generator: Option<Generator>) -> &mut Tensor {
        
        todo!();
            /*
                return native::templates::random_from_to_impl<native::templates::cpu::RandomFromToKernel, TestCPUGenerator>(self, from, to, generator);
            */
    }

    pub fn random_to(
            self_:     &mut Tensor,
            to:        i64,
            generator: Option<Generator>) -> &mut Tensor {
        
        todo!();
            /*
                return random_from_to(self, 0, to, generator);
            */
    }

    // ==================================================== Normal ========================================================

    pub fn normal(
            self_: &mut Tensor,
            mean:  f64,
            std:   f64,
            gen:   Option<Generator>) -> &mut Tensor {
        
        todo!();
            /*
                return native::templates::normal_impl_<native::templates::cpu::NormalKernel, TestCPUGenerator>(self, mean, std, gen);
            */
    }

    pub fn normal_tensor_float_out(
            mean:   &Tensor,
            std:    f64,
            gen:    Option<Generator>,
            output: &mut Tensor) -> &mut Tensor {
        
        todo!();
            /*
                return native::templates::normal_out_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(output, mean, std, gen);
            */
    }

    pub fn normal_float_tensor_out(
            mean:   f64,
            std:    &Tensor,
            gen:    Option<Generator>,
            output: &mut Tensor) -> &mut Tensor {
        
        todo!();
            /*
                return native::templates::normal_out_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(output, mean, std, gen);
            */
    }

    pub fn normal_tensor_tensor_out(
            mean:   &Tensor,
            std:    &Tensor,
            gen:    Option<Generator>,
            output: &mut Tensor) -> &mut Tensor {
        
        todo!();
            /*
                return native::templates::normal_out_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(output, mean, std, gen);
            */
    }


    pub fn normal_tensor_float(
            mean: &Tensor,
            std:  f64,
            gen:  Option<Generator>) -> Tensor {
        
        todo!();
            /*
                return native::templates::normal_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(mean, std, gen);
            */
    }


    pub fn normal_float_tensor(
            mean: f64,
            std:  &Tensor,
            gen:  Option<Generator>) -> Tensor {
        
        todo!();
            /*
                return native::templates::normal_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(mean, std, gen);
            */
    }


    pub fn normal_tensor_tensor(
            mean: &Tensor,
            std:  &Tensor,
            gen:  Option<Generator>) -> Tensor {
        
        todo!();
            /*
                return native::templates::normal_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(mean, std, gen);
            */
    }

    // ==================================================== Uniform =======================================================


    pub fn uniform(
            self_:     &mut Tensor,
            from:      f64,
            to:        f64,
            generator: Option<Generator>) -> &mut Tensor {
        
        todo!();
            /*
                return native::templates::uniform_impl_<native::templates::cpu::UniformKernel, TestCPUGenerator>(self, from, to, generator);
            */
    }



    // ==================================================== Cauchy ========================================================


    pub fn cauchy(
            self_:     &mut Tensor,
            median:    f64,
            sigma:     f64,
            generator: Option<Generator>) -> &mut Tensor {
        
        todo!();
            /*
                return native::templates::cauchy_impl_<native::templates::cpu::CauchyKernel, TestCPUGenerator>(self, median, sigma, generator);
            */
    }



    // ================================================== LogNormal =======================================================


    pub fn log_normal(
            self_: &mut Tensor,
            mean:  f64,
            std:   f64,
            gen:   Option<Generator>) -> &mut Tensor {
        
        todo!();
            /*
                return native::templates::log_normal_impl_<native::templates::cpu::LogNormalKernel, TestCPUGenerator>(self, mean, std, gen);
            */
    }



    // ================================================== Geometric =======================================================


    pub fn geometric(
            self_: &mut Tensor,
            p:     f64,
            gen:   Option<Generator>) -> &mut Tensor {
        
        todo!();
            /*
                return native::templates::geometric_impl_<native::templates::cpu::GeometricKernel, TestCPUGenerator>(self, p, gen);
            */
    }



    // ================================================== Exponential =====================================================


    pub fn exponential(
            self_:  &mut Tensor,
            lambda: f64,
            gen:    Option<Generator>) -> &mut Tensor {
        
        todo!();
            /*
                return native::templates::exponential_impl_<native::templates::cpu::ExponentialKernel, TestCPUGenerator>(self, lambda, gen);
            */
    }



    // ================================================== Bernoulli =======================================================


    pub fn bernoulli_tensor(
            self_: &mut Tensor,
            p:     &Tensor,
            gen:   Option<Generator>) -> &mut Tensor {
        
        todo!();
            /*
                return native::templates::bernoulli_impl_<native::templates::cpu::BernoulliKernel, TestCPUGenerator>(self, p_, gen);
            */
    }


    pub fn bernoulli_float(
            self_: &mut Tensor,
            p:     f64,
            gen:   Option<Generator>) -> &mut Tensor {
        
        todo!();
            /*
                return native::templates::bernoulli_impl_<native::templates::cpu::BernoulliKernel, TestCPUGenerator>(self, p, gen);
            */
    }


    pub fn bernoulli_out(
            self_:  &Tensor,
            gen:    Option<Generator>,
            result: &mut Tensor) -> &mut Tensor {
        
        todo!();
            /*
                return native::templates::bernoulli_out_impl<native::templates::cpu::BernoulliKernel, TestCPUGenerator>(result, self, gen);
            */
    }

    lazy_static!{
        /*
        TORCH_LIBRARY_IMPL(aten, CustomRNGKeyId, m) {
          // Random
          m.impl("random_.from",             random_from_to);
          m.impl("random_.to",               random_to);
          m.impl("random_",                  random_);
          // Normal
          m.impl("normal_",                  normal_);
          m.impl("normal.Tensor_float_out",  normal_Tensor_float_out);
          m.impl("normal.float_Tensor_out",  normal_float_Tensor_out);
          m.impl("normal.Tensor_Tensor_out", normal_Tensor_Tensor_out);
          m.impl("normal.Tensor_float",      normal_Tensor_float);
          m.impl("normal.float_Tensor",      normal_float_Tensor);
          m.impl("normal.Tensor_Tensor",     normal_Tensor_Tensor);
          m.impl("uniform_",                 uniform_);
          // Cauchy
          m.impl("cauchy_",                  cauchy_);
          // LogNormal
          m.impl("log_normal_",              log_normal_);
          // Geometric
          m.impl("geometric_",               geometric_);
          // Exponential
          m.impl("exponential_",             exponential_);
          // Bernoulli
          m.impl("bernoulli.out",            bernoulli_out);
          m.impl("bernoulli_.Tensor",        bernoulli_Tensor);
          m.impl("bernoulli_.float",         bernoulli_float);
        }
        */
    }

    pub struct RNGTest {
        base: Test,
    }

    pub const MAGIC_NUMBER: Auto = 424242424242424242ULL;

    // ==================================================== Random ========================================================

    #[test] fn rng_test_random_from_to() {
        todo!();
        /*
        
          const Device device("cpu");
          test_random_from_to<TestCPUGenerator, TorchkBool, bool>(device);
          test_random_from_to<TestCPUGenerator, TorchkUInt8, u8>(device);
          test_random_from_to<TestCPUGenerator, TorchkInt8, i8>(device);
          test_random_from_to<TestCPUGenerator, TorchkInt16, i16>(device);
          test_random_from_to<TestCPUGenerator, TorchkInt32, i32>(device);
          test_random_from_to<TestCPUGenerator, TorchkInt64, i64>(device);
          test_random_from_to<TestCPUGenerator, TorchkFloat32, float>(device);
          test_random_from_to<TestCPUGenerator, TorchkFloat64, double>(device);

        */
    }

    #[test] fn rng_test_random() {
        todo!();
        /*
        
          const Device device("cpu");
          test_random<TestCPUGenerator, TorchkBool, bool>(device);
          test_random<TestCPUGenerator, TorchkUInt8, u8>(device);
          test_random<TestCPUGenerator, TorchkInt8, i8>(device);
          test_random<TestCPUGenerator, TorchkInt16, i16>(device);
          test_random<TestCPUGenerator, TorchkInt32, i32>(device);
          test_random<TestCPUGenerator, TorchkInt64, i64>(device);
          test_random<TestCPUGenerator, TorchkFloat32, float>(device);
          test_random<TestCPUGenerator, TorchkFloat64, double>(device);

        */
    }

    /**
      | This test proves that Tensor.random_()
      | distribution is able to generate unsigned 64
      | bit max value(64 ones)
      |
      | https://github.com/pytorch/pytorch/issues/33299
      |
      */
    #[test] fn rng_test_random_64bits() {
        todo!();
        /*
        
          auto gen = make_generator<TestCPUGenerator>(u64::max);
          auto actual = Torchempty({1}, TorchkInt64);
          actual.random_(i64::min, nullopt, gen);
          ASSERT_EQ(static_cast<u64>(actual[0].item<i64>()), u64::max);

        */
    }


    // ==================================================== Normal ========================================================

    #[test] fn rng_test_normal() {
        todo!();
        /*
        
          const auto mean = 123.45;
          const auto std = 67.89;
          auto gen = make_generator<TestCPUGenerator>(MAGIC_NUMBER);

          auto actual = Torchempty({10});
          actual.normal_(mean, std, gen);

          auto expected = Torchempty_like(actual);
          native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

          ASSERT_TRUE(Torchallclose(actual, expected));

        */
    }


    #[test] fn rng_test_normal_float_tensor_out() {
        todo!();
        /*
        
          const auto mean = 123.45;
          const auto std = 67.89;
          auto gen = make_generator<TestCPUGenerator>(MAGIC_NUMBER);

          auto actual = Torchempty({10});
          normal_out(actual, mean, Torchfull({10}, std), gen);

          auto expected = Torchempty_like(actual);
          native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

          ASSERT_TRUE(Torchallclose(actual, expected));

        */
    }


    #[test] fn rng_test_normal_tensor_float_out() {
        todo!();
        /*
        
          const auto mean = 123.45;
          const auto std = 67.89;
          auto gen = make_generator<TestCPUGenerator>(MAGIC_NUMBER);

          auto actual = Torchempty({10});
          normal_out(actual, Torchfull({10}, mean), std, gen);

          auto expected = Torchempty_like(actual);
          native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

          ASSERT_TRUE(Torchallclose(actual, expected));

        */
    }


    #[test] fn rng_test_normal_tensor_out() {
        todo!();
        /*
        
          const auto mean = 123.45;
          const auto std = 67.89;
          auto gen = make_generator<TestCPUGenerator>(MAGIC_NUMBER);

          auto actual = Torchempty({10});
          normal_out(actual, Torchfull({10}, mean), Torchfull({10}, std), gen);

          auto expected = Torchempty_like(actual);
          native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

          ASSERT_TRUE(Torchallclose(actual, expected));

        */
    }


    #[test] fn rng_test_normal_float_tensor() {
        todo!();
        /*
        
          const auto mean = 123.45;
          const auto std = 67.89;
          auto gen = make_generator<TestCPUGenerator>(MAGIC_NUMBER);

          auto actual = normal(mean, Torchfull({10}, std), gen);

          auto expected = Torchempty_like(actual);
          native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

          ASSERT_TRUE(Torchallclose(actual, expected));

        */
    }


    #[test] fn rng_test_normal_tensor_float() {
        todo!();
        /*
        
          const auto mean = 123.45;
          const auto std = 67.89;
          auto gen = make_generator<TestCPUGenerator>(MAGIC_NUMBER);

          auto actual = normal(Torchfull({10}, mean), std, gen);

          auto expected = Torchempty_like(actual);
          native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

          ASSERT_TRUE(Torchallclose(actual, expected));

        */
    }


    #[test] fn rng_test_normal_tensor() {
        todo!();
        /*
        
          const auto mean = 123.45;
          const auto std = 67.89;
          auto gen = make_generator<TestCPUGenerator>(MAGIC_NUMBER);

          auto actual = normal(Torchfull({10}, mean), Torchfull({10}, std), gen);

          auto expected = Torchempty_like(actual);
          native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

          ASSERT_TRUE(Torchallclose(actual, expected));

        */
    }


    // ==================================================== Uniform =======================================================

    #[test] fn rng_test_uniform() {
        todo!();
        /*
        
          const auto from = -24.24;
          const auto to = 42.42;
          auto gen = make_generator<TestCPUGenerator>(MAGIC_NUMBER);

          auto actual = Torchempty({3, 3});
          actual.uniform_(from, to, gen);

          auto expected = Torchempty_like(actual);
          auto iter = TensorIterator::nullary_op(expected);
          native::templates::cpu::uniform_kernel(iter, from, to, check_generator<TestCPUGenerator>(gen));

          ASSERT_TRUE(Torchallclose(actual, expected));

        */
    }


    // ==================================================== Cauchy ========================================================

    #[test] fn rng_test_cauchy() {
        todo!();
        /*
        
          const auto median = 123.45;
          const auto sigma = 67.89;
          auto gen = make_generator<TestCPUGenerator>(MAGIC_NUMBER);

          auto actual = Torchempty({3, 3});
          actual.cauchy_(median, sigma, gen);

          auto expected = Torchempty_like(actual);
          auto iter = TensorIterator::nullary_op(expected);
          native::templates::cpu::cauchy_kernel(iter, median, sigma, check_generator<TestCPUGenerator>(gen));

          ASSERT_TRUE(Torchallclose(actual, expected));

        */
    }


    // ================================================== LogNormal =======================================================

    #[test] fn rng_test_log_normal() {
        todo!();
        /*
        
          const auto mean = 12.345;
          const auto std = 6.789;
          auto gen = make_generator<TestCPUGenerator>(MAGIC_NUMBER);

          auto actual = Torchempty({10});
          actual.log_normal_(mean, std, gen);

          auto expected = Torchempty_like(actual);
          auto iter = TensorIterator::nullary_op(expected);
          native::templates::cpu::log_normal_kernel(iter, mean, std, check_generator<TestCPUGenerator>(gen));

          ASSERT_TRUE(Torchallclose(actual, expected));

        */
    }


    // ================================================== Geometric =======================================================

    #[test] fn rng_test_geometric() {
        todo!();
        /*
        
          const auto p = 0.42;
          auto gen = make_generator<TestCPUGenerator>(MAGIC_NUMBER);

          auto actual = Torchempty({3, 3});
          actual.geometric_(p, gen);

          auto expected = Torchempty_like(actual);
          auto iter = TensorIterator::nullary_op(expected);
          native::templates::cpu::geometric_kernel(iter, p, check_generator<TestCPUGenerator>(gen));

          ASSERT_TRUE(Torchallclose(actual, expected));

        */
    }


    // ================================================== Exponential =====================================================

    #[test] fn rng_test_exponential() {
        todo!();
        /*
        
          const auto lambda = 42;
          auto gen = make_generator<TestCPUGenerator>(MAGIC_NUMBER);

          auto actual = Torchempty({3, 3});
          actual.exponential_(lambda, gen);

          auto expected = Torchempty_like(actual);
          auto iter = TensorIterator::nullary_op(expected);
          native::templates::cpu::exponential_kernel(iter, lambda, check_generator<TestCPUGenerator>(gen));

          ASSERT_TRUE(Torchallclose(actual, expected));

        */
    }


    // ==================================================== Bernoulli =====================================================

    #[test] fn rng_test_bernoulli_tensor() {
        todo!();
        /*
        
          const auto p = 0.42;
          auto gen = make_generator<TestCPUGenerator>(MAGIC_NUMBER);

          auto actual = Torchempty({3, 3});
          actual.bernoulli_(Torchfull({3,3}, p), gen);

          auto expected = Torchempty_like(actual);
          native::templates::cpu::bernoulli_kernel(expected, Torchfull({3,3}, p), check_generator<TestCPUGenerator>(gen));

          ASSERT_TRUE(Torchallclose(actual, expected));

        */
    }


    #[test] fn rng_test_bernoulli_scalar() {
        todo!();
        /*
        
          const auto p = 0.42;
          auto gen = make_generator<TestCPUGenerator>(MAGIC_NUMBER);

          auto actual = Torchempty({3, 3});
          actual.bernoulli_(p, gen);

          auto expected = Torchempty_like(actual);
          native::templates::cpu::bernoulli_kernel(expected, p, check_generator<TestCPUGenerator>(gen));

          ASSERT_TRUE(Torchallclose(actual, expected));

        */
    }


    #[test] fn rng_test_bernoulli() {
        todo!();
        /*
        
          const auto p = 0.42;
          auto gen = make_generator<TestCPUGenerator>(MAGIC_NUMBER);

          auto actual = bernoulli(Torchfull({3,3}, p), gen);

          auto expected = Torchempty_like(actual);
          native::templates::cpu::bernoulli_kernel(expected, Torchfull({3,3}, p), check_generator<TestCPUGenerator>(gen));

          ASSERT_TRUE(Torchallclose(actual, expected));

        */
    }


    #[test] fn rng_test_bernoulli_2() {
        todo!();
        /*
        
          const auto p = 0.42;
          auto gen = make_generator<TestCPUGenerator>(MAGIC_NUMBER);

          auto actual = Torchfull({3,3}, p).bernoulli(gen);

          auto expected = Torchempty_like(actual);
          native::templates::cpu::bernoulli_kernel(expected, Torchfull({3,3}, p), check_generator<TestCPUGenerator>(gen));

          ASSERT_TRUE(Torchallclose(actual, expected));

        */
    }


    #[test] fn rng_test_bernoulli_p() {
        todo!();
        /*
        
          const auto p = 0.42;
          auto gen = make_generator<TestCPUGenerator>(MAGIC_NUMBER);

          auto actual = bernoulli(Torchempty({3, 3}), p, gen);

          auto expected = Torchempty_like(actual);
          native::templates::cpu::bernoulli_kernel(expected, p, check_generator<TestCPUGenerator>(gen));

          ASSERT_TRUE(Torchallclose(actual, expected));

        */
    }


    #[test] fn rng_test_bernoulli_p_2() {
        todo!();
        /*
        
          const auto p = 0.42;
          auto gen = make_generator<TestCPUGenerator>(MAGIC_NUMBER);

          auto actual = Torchempty({3, 3}).bernoulli(p, gen);

          auto expected = Torchempty_like(actual);
          native::templates::cpu::bernoulli_kernel(expected, p, check_generator<TestCPUGenerator>(gen));

          ASSERT_TRUE(Torchallclose(actual, expected));

        */
    }


    #[test] fn rng_test_bernoulli_out() {
        todo!();
        /*
        
          const auto p = 0.42;
          auto gen = make_generator<TestCPUGenerator>(MAGIC_NUMBER);

          auto actual = Torchempty({3, 3});
          bernoulli_out(actual, Torchfull({3,3}, p), gen);

          auto expected = Torchempty_like(actual);
          native::templates::cpu::bernoulli_kernel(expected, Torchfull({3,3}, p), check_generator<TestCPUGenerator>(gen));

          ASSERT_TRUE(Torchallclose(actual, expected));

        */
    }

}
