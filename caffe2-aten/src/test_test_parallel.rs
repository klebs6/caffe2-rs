crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/test_parallel.cpp]

#[test] fn test_parallel() {
    todo!();
    /*
    
      manual_seed(123);
      set_num_threads(1);

      Tensor a = rand({1, 3});
      a[0][0] = 1;
      a[0][1] = 0;
      a[0][2] = 0;
      Tensor as = rand({3});
      as[0] = 1;
      as[1] = 0;
      as[2] = 0;
      ASSERT_TRUE(a.sum(0).equal(as));

    */
}

#[test] fn test_parallel_nested() {
    todo!();
    /*
    
      Tensor a = ones({1024, 1024});
      auto expected = a.sum();
      // check that calling sum() from within a parallel block computes the same result
      parallel_for(0, 10, 1, [&](i64 begin, i64 end) {
        if (begin == 0) {
          ASSERT_TRUE(a.sum().equal(expected));
        }
      });

    */
}

#[test] fn test_parallel_exceptions() {
    todo!();
    /*
    
      // parallel case
      ASSERT_THROW(
        parallel_for(0, 10, 1, [&](i64 begin, i64 end) {
          throw runtime_error("exception");
        }),
        runtime_error);

      // non-parallel case
      ASSERT_THROW(
        parallel_for(0, 1, 1000, [&](i64 begin, i64 end) {
          throw runtime_error("exception");
        }),
        runtime_error);

    */
}

#[test] fn test_parallel_intra_op_launch_future() {
    todo!();
    /*
    
      int v1 = 0;
      int v2 = 0;

      auto fut1 = intraop_launch_future([&v1](){
        v1 = 1;
      });

      auto fut2 = intraop_launch_future([&v2](){
        v2 = 2;
      });

      fut1->wait();
      fut2->wait();

      ASSERT_TRUE(v1 == 1 && v2 == 2);

    */
}
