crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/cuda_stream_test.cpp]

#[macro_export] macro_rules! assert_eq_cuda {
    ($X:ident, $Y:ident) => {
        /*
        
          {                          
            bool isTRUE = X == Y;    
            ASSERT_TRUE(isTRUE);     
          }
        */
    }
}


#[macro_export] macro_rules! assert_ne_cuda {
    ($X:ident, $Y:ident) => {
        /*
        
          {                          
            bool isFALSE = X == Y;   
            ASSERT_FALSE(isFALSE);   
          }
        */
    }
}

/* --------- Tests related to ATen streams.  --------- */

/**
  | Verifies streams are live through copying
  | and moving
  |
  */
#[test] fn test_stream_copy_and_move() {
    todo!();
    /*
    
      if (!is_available()) return;
      i32 device = -1;
      cudaStream_t cuda_stream;

      // Tests that copying works as expected and preserves the stream
      CUDAStream copyStream = getStreamFromPool();
      {
        auto s = getStreamFromPool();
        device = s.device_index();
        cuda_stream = s.stream();

        copyStream = s;

        ASSERT_EQ_CUDA(copyStream.device_index(), device);
        ASSERT_EQ_CUDA(copyStream.stream(), cuda_stream);
      }

      ASSERT_EQ_CUDA(copyStream.device_index(), device);
      ASSERT_EQ_CUDA(copyStream.stream(), cuda_stream);

      // Tests that moving works as expected and preserves the stream
      CUDAStream moveStream = getStreamFromPool();
      {
        auto s = getStreamFromPool();
        device = s.device_index();
        cuda_stream = s.stream();

        moveStream = move(s);

        ASSERT_EQ_CUDA(moveStream.device_index(), device);
        ASSERT_EQ_CUDA(moveStream.stream(), cuda_stream);
      }

      ASSERT_EQ_CUDA(moveStream.device_index(), device);
      ASSERT_EQ_CUDA(moveStream.stream(), cuda_stream);

    */
}

// Verifies streams are set properly
#[test] fn test_stream_get_and_set() {
    todo!();
    /*
    
      if (!is_available()) return;
      CUDAStream myStream = getStreamFromPool();

      // Sets and gets
      setCurrentCUDAStream(myStream);
      CUDAStream curStream = getCurrentCUDAStream();

      ASSERT_EQ_CUDA(myStream, curStream);

      // Gets, sets, and gets default stream
      CUDAStream defaultStream = getDefaultCUDAStream();
      setCurrentCUDAStream(defaultStream);
      curStream = getCurrentCUDAStream();

      ASSERT_NE_CUDA(defaultStream, myStream);
      ASSERT_EQ_CUDA(curStream, defaultStream);

    */
}

pub fn thread_fun(cur_thread_stream: &mut Option<CUDAStream>)  {
    
    todo!();
        /*
            auto new_stream = getStreamFromPool();
      setCurrentCUDAStream(new_stream);
      cur_thread_stream = {getCurrentCUDAStream()};
      ASSERT_EQ_CUDA(*cur_thread_stream, new_stream);
        */
}

// Ensures streams are thread local
#[test] fn test_stream_multithread_get_and_set() {
    todo!();
    /*
    
      if (!is_available()) return;
      optional<CUDAStream> s0, s1;

      thread t0{thread_fun, ref(s0)};
      thread t1{thread_fun, ref(s1)};
      t0.join();
      t1.join();

      CUDAStream cur_stream = getCurrentCUDAStream();
      CUDAStream default_stream = getDefaultCUDAStream();

      ASSERT_EQ_CUDA(cur_stream, default_stream);
      ASSERT_NE_CUDA(cur_stream, *s0);
      ASSERT_NE_CUDA(cur_stream, *s1);
      ASSERT_NE_CUDA(s0, s1);

    */
}

// CUDA Guard
#[test] fn test_stream_cuda_guard() {
    todo!();
    /*
    
      if (!is_available()) return;
      if (getNumGPUs() < 2) {
        return;
      }

      // -- begin setup

      ASSERT_EQ_CUDA(current_device(), 0);
      vector<CUDAStream> streams0 = {
          getDefaultCUDAStream(), getStreamFromPool()};
      ASSERT_EQ_CUDA(streams0[0].device_index(), 0);
      ASSERT_EQ_CUDA(streams0[1].device_index(), 0);
      setCurrentCUDAStream(streams0[0]);

      vector<CUDAStream> streams1;
      {
        CUDAGuard device_guard(1);
        streams1.push_back(getDefaultCUDAStream());
        streams1.push_back(getStreamFromPool());
      }
      ASSERT_EQ_CUDA(streams1[0].device_index(), 1);
      ASSERT_EQ_CUDA(streams1[1].device_index(), 1);
      setCurrentCUDAStream(streams1[0]);

      ASSERT_EQ_CUDA(current_device(), 0);

      // -- end setup

      // Setting a stream changes the current device and the stream on that device
      {
        CUDAStreamGuard guard(streams1[1]);
        ASSERT_EQ_CUDA(guard.current_device(), Device(kCUDA, 1));
        ASSERT_EQ_CUDA(current_device(), 1);
        ASSERT_EQ_CUDA(getCurrentCUDAStream(1), streams1[1]);
      }

      // Device and stream are now reset
      ASSERT_EQ_CUDA(current_device(), 0);
      ASSERT_EQ_CUDA(getCurrentCUDAStream(1), streams1[0]);

      // Setting only the device changes only the current device and not the stream
      {
        CUDAGuard guard(/*device=*/1);
        ASSERT_EQ_CUDA(guard.current_device(), Device(kCUDA, 1));
        ASSERT_EQ_CUDA(current_device(), 1);
        ASSERT_EQ_CUDA(getCurrentCUDAStream(1), streams1[0]);
      }

      ASSERT_EQ_CUDA(current_device(), 0);
      ASSERT_EQ_CUDA(getCurrentCUDAStream(0), streams0[0]);

    */
}

// Streampool Round Robin
#[test] fn test_stream_pool() {
    todo!();
    /*
    
      if (!is_available()) return;
      vector<CUDAStream> streams{};
      for (const auto i : irange(200)) {
        streams.emplace_back(getStreamFromPool());
      }

      unordered_set<cudaStream_t> stream_set{};
      bool hasDuplicates = false;
      for (const auto i: irange(streams.size())) {
        cudaStream_t cuda_stream = streams[i];
        auto result_pair = stream_set.insert(cuda_stream);
        if (!result_pair.second)
          hasDuplicates = true;
      }

      ASSERT_TRUE(hasDuplicates);

    */
}

// Multi-GPU
#[test] fn test_stream_multi_gpu() {
    todo!();
    /*
    
      if (!is_available()) return;
      if (getNumGPUs() < 2)
        return;

      CUDAStream s0 = getStreamFromPool(true, 0);
      CUDAStream s1 = getStreamFromPool(false, 1);

      setCurrentCUDAStream(s0);
      setCurrentCUDAStream(s1);

      ASSERT_EQ_CUDA(s0, getCurrentCUDAStream());

      CUDAGuard device_guard{1};
      ASSERT_EQ_CUDA(s1, getCurrentCUDAStream());

    */
}

// CudaEvent Syncs
#[test] fn test_stream_cuda_event_sync() {
    todo!();
    /*
    
      if (!is_available()) return;
      const auto stream = getStreamFromPool();
      CudaEvent event;

      ASSERT_TRUE(event.query());

      event.recordOnce(stream);

      const auto wait_stream0 = getStreamFromPool();
      const auto wait_stream1 = getStreamFromPool();

      event.block(wait_stream0);
      event.block(wait_stream1);

      cudaStreamSynchronize(wait_stream0);
      ASSERT_TRUE(event.query());

    */
}

// Cross-Device Events
#[test] fn test_stream_cross_device() {
    todo!();
    /*
    
      if (!is_available()) return;
      if (getNumGPUs() < 2)
        return;

      const auto stream0 = getStreamFromPool();
      CudaEvent event0;

      set_device(1);
      const auto stream1 = getStreamFromPool();
      CudaEvent event1;

      event0.record(stream0);
      event1.record(stream1);

      event0 = move(event1);

      ASSERT_EQ_CUDA(event0.device(), Device(kCUDA, 1));

      event0.block(stream0);

      cudaStreamSynchronize(stream0);
      ASSERT_TRUE(event0.query());

    */
}

// Generic Events
#[test] fn test_stream_generic_inline_cuda_event() {
    todo!();
    /*
    
      if (!is_available()) return;

      InlineEvent<CUDAGuardImpl> event{DeviceType_CUDA};
      Stream stream = getStreamFromPool();

      event.record(stream);

      const Stream wait_stream0 = getStreamFromPool();
      const Stream wait_stream1 = getStreamFromPool();

      event.block(wait_stream0);
      event.block(wait_stream1);

      const CUDAStream cuda_stream{wait_stream0};
      cudaStreamSynchronize(cuda_stream);

      ASSERT_TRUE(event.query());

    */
}

#[test] fn test_stream_generic_virtual_cuda_event() {
    todo!();
    /*
    
      if (!is_available()) return;

      Event event{DeviceType_CUDA};
      Stream stream = getStreamFromPool();

      event.recordOnce(stream);

      const Stream wait_stream0 = getStreamFromPool();
      const Stream wait_stream1 = getStreamFromPool();

      wait_stream0.wait(event);
      wait_stream1.wait(event);

      const CUDAStream cuda_stream{wait_stream0};
      cudaStreamSynchronize(cuda_stream);

      ASSERT_TRUE(event.query());
      ASSERT_TRUE(event.flag() == EventFlag::PYTORCH_DEFAULT);

    */
}
