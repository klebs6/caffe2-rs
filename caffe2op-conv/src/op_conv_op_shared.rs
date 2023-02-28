crate::ix!();

use crate::{
    Tensor,
    Workspace
};

/**
  | Creates a mutex and shared buffer in
  | the workspace.
  | 
  | Not thread-safe, must be called from
  | the constructor.
  |
  */
pub fn create_shared_buffer_cpu_context(ws: *mut Workspace) {

    todo!();
    /*
  auto* mutexPtr = ws->CreateBlob("__CAFFE2_SHARED_CONV_BUFFER_CPU_MUTEX__")
                       ->GetMutable<std::unique_ptr<std::mutex>>();
  mutexPtr->reset(new std::mutex());
  ws->CreateBlob("__CAFFE2_SHARED_CONV_BUFFER_CPU__");
    */
}

/**
  | Thread-safe, can be invoked from RunOnDevice()
  | to serialize access to shared buffer.
  |
  */
pub fn run_with_shared_buffer_cpu_context(ws: *mut Workspace, f: fn(buffer: *mut Tensor) -> ()) {

    todo!();
    /*
      auto* mutexBlob = ws->GetBlob("__CAFFE2_SHARED_CONV_BUFFER_CPU_MUTEX__");
      CAFFE_ENFORCE(mutexBlob, "Must call createSharedBuffer() first");

      auto* mutexPtr = mutexBlob->GetMutable<std::unique_ptr<std::mutex>>();
      std::lock_guard<std::mutex> g(**mutexPtr);
      auto* buffer = BlobGetMutableTensor(
          ws->GetBlob("__CAFFE2_SHARED_CONV_BUFFER_CPU__"), CPU);
      f(buffer);
    */
}

pub fn create_shared_buffer_cuda_context(ws: *mut Workspace) {

    todo!();
    /*
      auto* mutexPtr = ws->CreateBlob("__CAFFE2_SHARED_CONV_BUFFER_CUDA_MUTEX__")
                           ->GetMutable<std::unique_ptr<std::mutex>>();
      mutexPtr->reset(new std::mutex());
      ws->CreateBlob("__CAFFE2_SHARED_CONV_BUFFER_CUDA__");
    */
}

pub fn run_with_shared_buffer_cuda_context( ws: *mut Workspace, f: fn(buffer: *mut Tensor) -> ()) {

    todo!();
    /*
      auto* mutexBlob = ws->GetBlob("__CAFFE2_SHARED_CONV_BUFFER_CUDA_MUTEX__");
      CAFFE_ENFORCE(mutexBlob, "Must call createSharedBuffer() first");

      auto* mutexPtr = mutexBlob->GetMutable<std::unique_ptr<std::mutex>>();
      std::lock_guard<std::mutex> g(**mutexPtr);
      auto* buffer = BlobGetMutableTensor(
          ws->GetBlob("__CAFFE2_SHARED_CONV_BUFFER_CUDA__"), CUDA);
      f(buffer);
    */
}
