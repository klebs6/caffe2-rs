crate::ix!();

pub struct BroadcastGPUTest {
    ws:           Workspace,
    option:       DeviceOption,
    cuda_context: Box<CUDAContext>,
    x:            *mut Tensor, // default = nullptr
    y:            *mut Tensor, // default = nullptr
}

impl BroadcastGPUTest {

    #[inline] fn set_up_data(&mut self, 
        x_dims: &Vec<i32>,
        y_dims: &Vec<i32>,
        x_data: &Vec<f32>)  {
        
        todo!();
        /*
            X_->Resize(X_dims);
        Y_->Resize(Y_dims);
        ASSERT_EQ(X_data.size(), X_->numel());
        cuda_context_->CopyFromCPU<float>(
            X_data.size(), X_data.data(), X_->mutable_data<float>());
        */
    }
    
    #[inline] fn verify_result(&mut self, expected_output: &Vec<f32>)  {
        
        todo!();
        /*
            Blob* blob_y_host = ws_.CreateBlob("Y_host");
        auto* Y_host = BlobGetMutableTensor(blob_y_host, CPU);
        Y_host->CopyFrom(*Y_);
        ASSERT_EQ(expected_output.size(), Y_host->numel());
        for (std::size_t i = 0; i < expected_output.size(); ++i) {
          EXPECT_FLOAT_EQ(expected_output[i], Y_host->data<float>()[i]);
        }
        */
    }
    
    #[inline] fn run_broadcast_test(&mut self, 
        x_dims: &Vec<i32>,
        y_dims: &Vec<i32>,
        x_data: &Vec<f32>,
        y_data: &Vec<f32>)  {

        todo!();
        /*
            SetUpData(X_dims, Y_dims, X_data);
        math::Broadcast<float, CUDAContext>(
            X_dims.size(),
            X_dims.data(),
            Y_dims.size(),
            Y_dims.data(),
            1.0f,
            X_->data<float>(),
            Y_->mutable_data<float>(),
            cuda_context_.get());
        VerifyResult(Y_data);
        */
    }
}

impl TestContext for BroadcastGPUTest {

    #[inline] fn setup() -> Self {
        
        todo!();
        /*
            if (!HasCudaGPU()) {
          return;
        }
        option_.set_device_type(PROTO_CUDA);
        cuda_context_ = make_unique<CUDAContext>(option_);
        Blob* blob_x = ws_.CreateBlob("X");
        Blob* blob_y = ws_.CreateBlob("Y");
        X_ = BlobGetMutableTensor(blob_x, CUDA);
        Y_ = BlobGetMutableTensor(blob_y, CUDA);
        */
    }
    
}
