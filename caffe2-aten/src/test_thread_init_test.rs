crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/thread_init_test.cpp]

/**
  | This checks whether threads can see the global
  | numbers of threads set and also whether the
  | scheduler will throw an exception when multiple
  | threads call their first parallel construct.
  |
  */
pub fn test(given_num_threads: i32)  {
    
    todo!();
        /*
            auto t = ones({1000 * 1000}, CPU(kFloat));
      ASSERT_TRUE(given_num_threads >= 0);
      ASSERT_EQ(get_num_threads(), given_num_threads);
      auto t_sum = t.sum();
      for (int i = 0; i < 1000; ++i) {
        t_sum = t_sum + t.sum();
      }
        */
}

pub fn main() -> i32 {
    
    todo!();
        /*
            init_num_threads();

      set_num_threads(4);
      test(4);
      thread t1([](){
        init_num_threads();
        test(4);
      });
      t1.join();

      #if !AT_PARALLEL_NATIVE
      set_num_threads(5);
      ASSERT_TRUE(get_num_threads() == 5);
      #endif

      // test inter-op settings
      set_num_interop_threads(5);
      ASSERT_EQ(get_num_interop_threads(), 5);
      ASSERT_ANY_THROW(set_num_interop_threads(6));

      return 0;
        */
}
