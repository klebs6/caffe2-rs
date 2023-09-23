crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/cpu_caching_allocator_test.cpp]

#[test] fn cpu_caching_allocator_test_check_alloc_free() {
    todo!();
    /*
    
      CPUCachingAllocator caching_allocator;
      WithCPUCachingAllocatorGuard cachine_allocator_guard(
          &caching_allocator);
      Tensor a = rand({23, 23});
      float* data_ptr = a.data_ptr<float>();
      a.reset();
      a = rand({23, 23});
      ASSERT_TRUE(data_ptr == a.data_ptr<float>());

    */
}

/// This should just free the pointer correctly.
#[test] fn cpu_caching_allocator_test_check_alloc_outside_free_inside() {
    todo!();
    /*
    
      CPUCachingAllocator caching_allocator;
      Tensor a = rand({23, 23});
      {
        WithCPUCachingAllocatorGuard cachine_allocator_guard(
            &caching_allocator);
        float* data_ptr = a.data_ptr<float>();
        a.reset();
        a = rand({23, 23});
      }

    */
}

#[test] fn cpu_caching_allocator_test_check_alloc_inside_free_outside() {
    todo!();
    /*
    
      CPUCachingAllocator caching_allocator;
      Tensor a;
      {
        WithCPUCachingAllocatorGuard cachine_allocator_guard(
            &caching_allocator);
        a = rand({23, 23});
      }
      a.reset();

    */
}

pub fn main(
        argc: i32,
        argv: &[*mut u8]) -> i32 {
    
    todo!();
        /*
            // At the moment caching allocator is only exposed to mobile cpu allocator.
    #ifdef C10_MOBILE
      ::testing::InitGoogleTest(&argc, argv);
      manual_seed(42);
      return RUN_ALL_TESTS();
    #endif /* C10_Mobile */
        */
}
