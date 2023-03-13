crate::ix!();

pub fn copy_vector_cuda_context(N: i32, x: *const bool, y: *mut bool) {
    todo!();
    /*
       CUDA_CHECK(cudaMemcpy(y, x, N * sizeof(bool), cudaMemcpyHostToDevice));
    */
}

pub fn create_operator_def_cuda_context() -> OperatorDef {
    todo!();
    /*
  caffe2::OperatorDef def;
  def.mutable_device_option()->set_device_type(caffe2::PROTO_CUDA);
  return def;
    */
}

#[test] fn elementwise_gpu_test_and() {
    todo!();
    /*
      if (!caffe2::HasCudaGPU())
        return;
      elementwiseAnd<caffe2::CUDAContext>();
  */
}


#[test] fn elementwise_gpu_test_or() {
    todo!();
    /*
      if (!caffe2::HasCudaGPU())
        return;
      elementwiseOr<caffe2::CUDAContext>();
  */
}


#[test] fn elementwise_gpu_test_xor() {
    todo!();
    /*
      if (!caffe2::HasCudaGPU())
        return;
      elementwiseXor<caffe2::CUDAContext>();
  */
}


#[test] fn elementwise_gpu_test_not() {
    todo!();
    /*
      if (!caffe2::HasCudaGPU())
        return;
      elementwiseNot<caffe2::CUDAContext>();
  */
}

