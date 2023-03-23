crate::ix!();

#[test] fn math_test_gemm_no_trans_no_trans() {
    todo!();
    /*
      DeviceOption option;
      CPUContext cpu_context(option);
      Tensor X(std::vector<int>{5, 10}, CPU);
      Tensor W(std::vector<int>{10, 6}, CPU);
      Tensor Y(std::vector<int>{5, 6}, CPU);
      EXPECT_EQ(X.numel(), 50);
      EXPECT_EQ(W.numel(), 60);
      math::Set<float, CPUContext>(
          X.numel(), 1, X.mutable_data<float>(), &cpu_context);
      math::Set<float, CPUContext>(
          W.numel(), 1, W.mutable_data<float>(), &cpu_context);
      EXPECT_EQ(Y.numel(), 30);
      for (int i = 0; i < X.numel(); ++i) {
        CHECK_EQ(X.data<float>()[i], 1);
      }
      for (int i = 0; i < W.numel(); ++i) {
        CHECK_EQ(W.data<float>()[i], 1);
      }

      const float kOne = 1.0;
      const float kPointFive = 0.5;
      const float kZero = 0.0;
      math::Gemm<float, CPUContext>(
          CblasNoTrans,
          CblasNoTrans,
          5,
          6,
          10,
          kOne,
          X.data<float>(),
          W.data<float>(),
          kZero,
          Y.mutable_data<float>(),
          &cpu_context);
      EXPECT_EQ(Y.numel(), 30);
      for (int i = 0; i < Y.numel(); ++i) {
        CHECK_EQ(Y.data<float>()[i], 10) << i;
      }
      // Test Accumulate
      math::Gemm<float, CPUContext>(
          CblasNoTrans,
          CblasNoTrans,
          5,
          6,
          10,
          kOne,
          X.data<float>(),
          W.data<float>(),
          kPointFive,
          Y.mutable_data<float>(),
          &cpu_context);
      EXPECT_EQ(Y.numel(), 30);
      for (int i = 0; i < Y.numel(); ++i) {
        CHECK_EQ(Y.data<float>()[i], 15) << i;
      }
      // Test Accumulate
      math::Gemm<float, CPUContext>(
          CblasNoTrans,
          CblasNoTrans,
          5,
          6,
          10,
          kPointFive,
          X.data<float>(),
          W.data<float>(),
          kOne,
          Y.mutable_data<float>(),
          &cpu_context);
      EXPECT_EQ(Y.numel(), 30);
      for (int i = 0; i < Y.numel(); ++i) {
        CHECK_EQ(Y.data<float>()[i], 20) << i;
      }
  */
}


#[test] fn math_test_gemm_no_trans_trans() {
    todo!();
    /*
      DeviceOption option;
      CPUContext cpu_context(option);
      Tensor X(std::vector<int>{5, 10}, CPU);
      Tensor W(std::vector<int>{6, 10}, CPU);
      Tensor Y(std::vector<int>{5, 6}, CPU);
      EXPECT_EQ(X.numel(), 50);
      EXPECT_EQ(W.numel(), 60);
      math::Set<float, CPUContext>(
          X.numel(), 1, X.mutable_data<float>(), &cpu_context);
      math::Set<float, CPUContext>(
          W.numel(), 1, W.mutable_data<float>(), &cpu_context);
      EXPECT_EQ(Y.numel(), 30);
      for (int i = 0; i < X.numel(); ++i) {
        CHECK_EQ(X.data<float>()[i], 1);
      }
      for (int i = 0; i < W.numel(); ++i) {
        CHECK_EQ(W.data<float>()[i], 1);
      }

      const float kOne = 1.0;
      const float kPointFive = 0.5;
      const float kZero = 0.0;
      math::Gemm<float, CPUContext>(
          CblasNoTrans,
          CblasTrans,
          5,
          6,
          10,
          kOne,
          X.data<float>(),
          W.data<float>(),
          kZero,
          Y.mutable_data<float>(),
          &cpu_context);
      EXPECT_EQ(Y.numel(), 30);
      for (int i = 0; i < Y.numel(); ++i) {
        CHECK_EQ(Y.data<float>()[i], 10) << i;
      }
      // Test Accumulate
      math::Gemm<float, CPUContext>(
          CblasNoTrans,
          CblasTrans,
          5,
          6,
          10,
          kOne,
          X.data<float>(),
          W.data<float>(),
          kPointFive,
          Y.mutable_data<float>(),
          &cpu_context);
      EXPECT_EQ(Y.numel(), 30);
      for (int i = 0; i < Y.numel(); ++i) {
        CHECK_EQ(Y.data<float>()[i], 15) << i;
      }
      math::Gemm<float, CPUContext>(
          CblasNoTrans,
          CblasTrans,
          5,
          6,
          10,
          kPointFive,
          X.data<float>(),
          W.data<float>(),
          kOne,
          Y.mutable_data<float>(),
          &cpu_context);
      EXPECT_EQ(Y.numel(), 30);
      for (int i = 0; i < Y.numel(); ++i) {
        CHECK_EQ(Y.data<float>()[i], 20) << i;
      }
  */
}

pub const kEps: f32 = 1e-5;

pub struct GemmBatchedTest {
    /*
       : public testing::TestWithParam<testing::tuple<bool, bool>> {
       */

    option:      DeviceOption,
    cpu_context: Box<CPUContext>,
    x:           Tensor,
    w:           Tensor,
    y:           Tensor,
    trans_X:     bool,
    trans_W:     bool,
}

impl GemmBatchedTest {
    
    #[inline] pub fn set_up(&mut self)  {
        
        todo!();
        /*
            cpu_context_ = make_unique<CPUContext>(option_);
        ReinitializeTensor(
            &X_, std::vector<int64_t>{3, 5, 10}, at::dtype<float>().device(CPU));
        ReinitializeTensor(
            &W_, std::vector<int64_t>{3, 6, 10}, at::dtype<float>().device(CPU));
        ReinitializeTensor(
            &Y_, std::vector<int64_t>{3, 5, 6}, at::dtype<float>().device(CPU));
        math::Set<float, CPUContext>(
            X_.numel(), 1, X_.mutable_data<float>(), cpu_context_.get());
        math::Set<float, CPUContext>(
            W_.numel(), 1, W_.mutable_data<float>(), cpu_context_.get());
        trans_X_ = std::get<0>(GetParam());
        trans_W_ = std::get<1>(GetParam());
        */
    }
    
    #[inline] pub fn run_gemm_batched(&mut self, alpha: f32, beta: f32)  {
        
        todo!();
        /*
            const float* X_data = X_.template data<float>();
        const float* W_data = W_.template data<float>();
        float* Y_data = Y_.template mutable_data<float>();
        const int X_stride = 5 * 10;
        const int W_stride = 6 * 10;
        const int Y_stride = 5 * 6;
        std::array<const float*, 3> X_array = {
            X_data, X_data + X_stride, X_data + 2 * X_stride};
        std::array<const float*, 3> W_array = {
            W_data, W_data + W_stride, W_data + 2 * W_stride};
        std::array<float*, 3> Y_array = {
            Y_data, Y_data + Y_stride, Y_data + 2 * Y_stride};
        math::GemmBatched(
            trans_X_ ? CblasTrans : CblasNoTrans,
            trans_W_ ? CblasTrans : CblasNoTrans,
            3,
            5,
            6,
            10,
            alpha,
            X_array.data(),
            W_array.data(),
            beta,
            Y_array.data(),
            cpu_context_.get());
        */
    }
    
    #[inline] pub fn run_gemm_strided_batched(&mut self, alpha: f32, beta: f32)  {
        
        todo!();
        /*
            const float* X_data = X_.template data<float>();
        const float* W_data = W_.template data<float>();
        float* Y_data = Y_.template mutable_data<float>();
        const int X_stride = 5 * 10;
        const int W_stride = 6 * 10;
        const int Y_stride = 5 * 6;
        math::GemmStridedBatched<float, CPUContext>(
            trans_X_ ? CblasTrans : CblasNoTrans,
            trans_W_ ? CblasTrans : CblasNoTrans,
            3,
            5,
            6,
            10,
            alpha,
            X_data,
            X_stride,
            W_data,
            W_stride,
            beta,
            Y_data,
            Y_stride,
            cpu_context_.get());
        */
    }
    
    #[inline] pub fn verify_output(&self, value: f32)  {
        
        todo!();
        /*
            for (int i = 0; i < Y_.numel(); ++i) {
          EXPECT_FLOAT_EQ(value, Y_.template data<float>()[i]);
        }
        */
    }
}

#[test] fn gemm_batched_test_gemm_batched_float_test() {
    todo!();
    /*
      RunGemmBatched(1.0f, 0.0f);
      VerifyOutput(10.0f);
      RunGemmBatched(1.0f, 0.5f);
      VerifyOutput(15.0f);
      RunGemmBatched(0.5f, 1.0f);
      VerifyOutput(20.0f);
  */
}


#[test] fn gemm_batched_test_gemm_strided_batched_float_test() {
    todo!();
    /*
      RunGemmStridedBatched(1.0f, 0.0f);
      VerifyOutput(10.0f);
      RunGemmStridedBatched(1.0f, 0.5f);
      VerifyOutput(15.0f);
      RunGemmStridedBatched(0.5f, 1.0f);
      VerifyOutput(20.0f);
  */
}

instantiate_test_case_p!{
    /*
    GemmBatchedTrans,
    GemmBatchedTest,
    testing::Combine(testing::Bool(), testing::Bool())
    */
}

#[test] fn math_test_gemv_no_trans() {
    todo!();
    /*
      DeviceOption option;
      CPUContext cpu_context(option);
      Tensor A(std::vector<int>{5, 10}, CPU);
      Tensor X(std::vector<int>{10}, CPU);
      Tensor Y(std::vector<int>{5}, CPU);
      EXPECT_EQ(A.numel(), 50);
      EXPECT_EQ(X.numel(), 10);
      math::Set<float, CPUContext>(
          A.numel(), 1, A.mutable_data<float>(), &cpu_context);
      math::Set<float, CPUContext>(
          X.numel(), 1, X.mutable_data<float>(), &cpu_context);
      EXPECT_EQ(Y.numel(), 5);
      for (int i = 0; i < A.numel(); ++i) {
        CHECK_EQ(A.data<float>()[i], 1);
      }
      for (int i = 0; i < X.numel(); ++i) {
        CHECK_EQ(X.data<float>()[i], 1);
      }

      const float kOne = 1.0;
      const float kPointFive = 0.5;
      const float kZero = 0.0;
      math::Gemv<float, CPUContext>(
          CblasNoTrans,
          5,
          10,
          kOne,
          A.data<float>(),
          X.data<float>(),
          kZero,
          Y.mutable_data<float>(),
          &cpu_context);
      for (int i = 0; i < Y.numel(); ++i) {
        CHECK_EQ(Y.data<float>()[i], 10) << i;
      }
      // Test Accumulate
      math::Gemv<float, CPUContext>(
          CblasNoTrans,
          5,
          10,
          kOne,
          A.data<float>(),
          X.data<float>(),
          kPointFive,
          Y.mutable_data<float>(),
          &cpu_context);
      for (int i = 0; i < Y.numel(); ++i) {
        CHECK_EQ(Y.data<float>()[i], 15) << i;
      }
      // Test Accumulate
      math::Gemv<float, CPUContext>(
          CblasNoTrans,
          5,
          10,
          kPointFive,
          A.data<float>(),
          X.data<float>(),
          kOne,
          Y.mutable_data<float>(),
          &cpu_context);
      for (int i = 0; i < Y.numel(); ++i) {
        CHECK_EQ(Y.data<float>()[i], 20) << i;
      }
  */
}

#[test] fn math_test_gemv_trans() {
    todo!();
    /*
      DeviceOption option;
      CPUContext cpu_context(option);
      Tensor A(std::vector<int>{6, 10}, CPU);
      Tensor X(std::vector<int>{6}, CPU);
      Tensor Y(std::vector<int>{10}, CPU);
      EXPECT_EQ(A.numel(), 60);
      EXPECT_EQ(X.numel(), 6);
      math::Set<float, CPUContext>(
          A.numel(), 1, A.mutable_data<float>(), &cpu_context);
      math::Set<float, CPUContext>(
          X.numel(), 1, X.mutable_data<float>(), &cpu_context);
      EXPECT_EQ(Y.numel(), 10);
      for (int i = 0; i < A.numel(); ++i) {
        CHECK_EQ(A.data<float>()[i], 1);
      }
      for (int i = 0; i < X.numel(); ++i) {
        CHECK_EQ(X.data<float>()[i], 1);
      }

      const float kOne = 1.0;
      const float kPointFive = 0.5;
      const float kZero = 0.0;
      math::Gemv<float, CPUContext>(
          CblasTrans,
          6,
          10,
          kOne,
          A.data<float>(),
          X.data<float>(),
          kZero,
          Y.mutable_data<float>(),
          &cpu_context);
      for (int i = 0; i < Y.numel(); ++i) {
        CHECK_EQ(Y.data<float>()[i], 6) << i;
      }
      // Test Accumulate
      math::Gemv<float, CPUContext>(
          CblasTrans,
          6,
          10,
          kOne,
          A.data<float>(),
          X.data<float>(),
          kPointFive,
          Y.mutable_data<float>(),
          &cpu_context);
      for (int i = 0; i < Y.numel(); ++i) {
        CHECK_EQ(Y.data<float>()[i], 9) << i;
      }
      // Test Accumulate
      math::Gemv<float, CPUContext>(
          CblasTrans,
          6,
          10,
          kPointFive,
          A.data<float>(),
          X.data<float>(),
          kOne,
          Y.mutable_data<float>(),
          &cpu_context);
      for (int i = 0; i < Y.numel(); ++i) {
        CHECK_EQ(Y.data<float>()[i], 12) << i;
      }
  */
}


#[test] fn math_test_float_to_half_conversion() {
    todo!();
    /*
      float a = 1.0f;
      float b = 1.75f;
      float c = 128.125f;

      float converted_a = static_cast<float>(at::Half(a));
      float converted_b = static_cast<float>(at::Half(b));
      float converted_c = static_cast<float>(at::Half(c));

      CHECK_EQ(a, converted_a);
      CHECK_EQ(b, converted_b);
      CHECK_EQ(c, converted_c);
  */
}


pub struct BroadcastTest {
    /*
       : public testing::Test 
       */

    option:      DeviceOption,
    cpu_context: Box<CPUContext>,
    x:           Tensor,
    y:           Tensor,
}

impl BroadcastTest {
    
    #[inline] pub fn set_up(&mut self)  {
        
        todo!();
        /*
            cpu_context_ = make_unique<CPUContext>(option_);
        */
    }
    
    #[inline] pub fn run_broadcast_test(&mut self, 
        x_dims: &Vec<i32>,
        y_dims: &Vec<i32>,
        x_data: &Vec<f32>,
        y_data: &Vec<f32>)  {

        todo!();
        /*
            std::vector<int64_t> X_dims_64;
        std::vector<int64_t> Y_dims_64;
        std::copy(X_dims.cbegin(), X_dims.cend(), std::back_inserter(X_dims_64));
        std::copy(Y_dims.cbegin(), Y_dims.cend(), std::back_inserter(Y_dims_64));
        ReinitializeTensor(&X_, X_dims_64, at::dtype<float>().device(CPU));
        ReinitializeTensor(&Y_, Y_dims_64, at::dtype<float>().device(CPU));
        ASSERT_EQ(X_data.size(), X_.numel());
        cpu_context_->CopyFromCPU<float>(
            X_data.size(), X_data.data(), X_.mutable_data<float>());
        math::Broadcast<float, CPUContext>(
            X_dims.size(),
            X_dims.data(),
            Y_dims.size(),
            Y_dims.data(),
            1.0f,
            X_.data<float>(),
            Y_.mutable_data<float>(),
            cpu_context_.get());
        ASSERT_EQ(Y_data.size(), Y_.numel());
        for (const auto i : c10::irange(Y_data.size())) {
          EXPECT_FLOAT_EQ(Y_data[i], Y_.data<float>()[i]);
        }
        */
    }
}

#[test] fn broadcast_test_broadcast_float_test() {
    todo!();
    /*
      RunBroadcastTest({2}, {2}, {1.0f, 2.0f}, {1.0f, 2.0f});
      RunBroadcastTest({1}, {2}, {1.0f}, {1.0f, 1.0f});
      RunBroadcastTest({1}, {2, 2}, {1.0f}, {1.0f, 1.0f, 1.0f, 1.0f});
      RunBroadcastTest({2, 1}, {2, 2}, {1.0f, 2.0f}, {1.0f, 1.0f, 2.0f, 2.0f});
      RunBroadcastTest(
          {2, 1},
          {2, 2, 2},
          {1.0f, 2.0f},
          {1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f});
  */
}

///------------------------------------
pub struct RandFixedSumTest {
    option:      DeviceOption,
    cpu_context: Box<CPUContext>,
}

impl TestContext for RandFixedSumTest {

    fn setup() -> Self {
        todo!();
        /*
           cpu_context_ = make_unique<CPUContext>(option_);
         */
    }

    fn teardown(self) {
        // Perform any teardown you wish.
    }

}

#[test_context(RandFixedSumTest)]
#[test] fn test_works(ctx: &mut RandFixedSumTest) {
    //assert_eq!(ctx.value, "Hello, world!");
}

#[test] fn rand_fixed_sum_test_upper_bound() {
    todo!();
    /*
      std::vector<int> l(20);
      math::RandFixedSum<int, CPUContext>(
          20, 1, 1000, 1000, l.data(), cpu_context_.get());
  */
}

