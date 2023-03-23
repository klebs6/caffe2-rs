crate::ix!();

pub const kEps: f32 = 1e-5;

pub struct GemmBatchedGPUTest {
    /*
       : public testing::TestWithParam<testing::tuple<bool, bool>> {
       */

    ws:            Workspace,
    option:        DeviceOption,
    cuda_context:  Box<CUDAContext>,
    x:             *mut Tensor, // default = nullptr
    w:             *mut Tensor, // default = nullptr
    y:             *mut Tensor, // default = nullptr
    trans_X:       bool,
    trans_W:       bool,
}

impl GemmBatchedGPUTest {

    #[inline] pub fn set_up(&mut self)  {

        todo!();
        /*
            if (!HasCudaGPU()) {
          return;
        }
        option_.set_device_type(PROTO_CUDA);
        cuda_context_ = make_unique<CUDAContext>(option_);
        Blob* X_blob = ws_.CreateBlob("X");
        Blob* W_blob = ws_.CreateBlob("W");
        Blob* Y_blob = ws_.CreateBlob("Y");
        X_ = BlobGetMutableTensor(X_blob, CUDA);
        W_ = BlobGetMutableTensor(W_blob, CUDA);
        Y_ = BlobGetMutableTensor(Y_blob, CUDA);
        X_->Resize(std::vector<int64_t>{3, 5, 10});
        W_->Resize(std::vector<int64_t>{3, 6, 10});
        Y_->Resize(std::vector<int64_t>{3, 5, 6});
        math::Set<float, CUDAContext>(
            X_->numel(), 1.0f, X_->mutable_data<float>(), cuda_context_.get());
        math::Set<float, CUDAContext>(
            W_->numel(), 1.0f, W_->mutable_data<float>(), cuda_context_.get());
        trans_X_ = std::get<0>(GetParam());
        trans_W_ = std::get<1>(GetParam());
        */
    }
    
    #[inline] pub fn run_gemm_batched(&mut self, alpha: f32, beta: f32)  {
        
        todo!();
        /*
            const float* X_data = X_->template data<float>();
        const float* W_data = W_->template data<float>();
        float* Y_data = Y_->template mutable_data<float>();
        const int X_stride = 5 * 10;
        const int W_stride = 6 * 10;
        const int Y_stride = 5 * 6;
        std::array<const float*, 3> X_array = {
            X_data, X_data + X_stride, X_data + 2 * X_stride};
        std::array<const float*, 3> W_array = {
            W_data, W_data + W_stride, W_data + 2 * W_stride};
        std::array<float*, 3> Y_array = {
            Y_data, Y_data + Y_stride, Y_data + 2 * Y_stride};
        math::GemmBatched<float, CUDAContext>(
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
            cuda_context_.get());
        */
    }
    
    #[inline] pub fn run_gemm_strided_batched(&mut self, alpha: f32, beta: f32)  {
        
        todo!();
        /*
            const float* X_data = X_->template data<float>();
        const float* W_data = W_->template data<float>();
        float* Y_data = Y_->template mutable_data<float>();
        const int X_stride = 5 * 10;
        const int W_stride = 6 * 10;
        const int Y_stride = 5 * 6;
        math::GemmStridedBatched<float, CUDAContext>(
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
            cuda_context_.get());
        */
    }
    
    #[inline] pub fn verify_output(&self, value: f32)  {
        
        todo!();
        /*
            Tensor Y_cpu(*Y_, CPU);
        for (int i = 0; i < Y_cpu.numel(); ++i) {
          EXPECT_FLOAT_EQ(value, Y_cpu.template data<float>()[i]);
        }
        */
    }
}
