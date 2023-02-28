crate::ix!();

use crate::{
    OperatorDef
};


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

#[test] fn ElementwiseGPUTest_And() {
    todo!();
    /*
      if (!caffe2::HasCudaGPU())
        return;
      elementwiseAnd<caffe2::CUDAContext>();
  */
}


#[test] fn ElementwiseGPUTest_Or() {
    todo!();
    /*
      if (!caffe2::HasCudaGPU())
        return;
      elementwiseOr<caffe2::CUDAContext>();
  */
}


#[test] fn ElementwiseGPUTest_Xor() {
    todo!();
    /*
      if (!caffe2::HasCudaGPU())
        return;
      elementwiseXor<caffe2::CUDAContext>();
  */
}


#[test] fn ElementwiseGPUTest_Not() {
    todo!();
    /*
      if (!caffe2::HasCudaGPU())
        return;
      elementwiseNot<caffe2::CUDAContext>();
  */
}

