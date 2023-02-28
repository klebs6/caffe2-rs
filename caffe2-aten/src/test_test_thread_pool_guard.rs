crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/test_thread_pool_guard.cpp]

#[test] fn test_thread_pool_guard() {
    todo!();
    /*
    
      auto threadpool_ptr = pthreadpool_();

      ASSERT_NE(threadpool_ptr, nullptr);
      {
        _NoPThreadPoolGuard g1;
        auto threadpool_ptr1 = pthreadpool_();
        ASSERT_EQ(threadpool_ptr1, nullptr);

        {
          _NoPThreadPoolGuard g2;
          auto threadpool_ptr2 = pthreadpool_();
          ASSERT_EQ(threadpool_ptr2, nullptr);
        }

        // Guard should restore prev value (nullptr)
        auto threadpool_ptr3 = pthreadpool_();
        ASSERT_EQ(threadpool_ptr3, nullptr);
      }

      // Guard should restore prev value (pthreadpool_)
      auto threadpool_ptr4 = pthreadpool_();
      ASSERT_NE(threadpool_ptr4, nullptr);
      ASSERT_EQ(threadpool_ptr4, threadpool_ptr);

    */
}

#[test] fn test_thread_pool_guard_run_with() {
    todo!();
    /*
    
      const vector<i64> array = {1, 2, 3};

      auto pool = pthreadpool();
      i64 inner = 0;
      {
        // Run on same thread
        _NoPThreadPoolGuard g1;
        auto fn = [&array, &inner](const usize task_id) {
          inner += array[task_id];
        };
        pool->run(fn, 3);

        // confirm the guard is on
        auto threadpool_ptr = pthreadpool_();
        ASSERT_EQ(threadpool_ptr, nullptr);
      }
      ASSERT_EQ(inner, 6);

    */
}
