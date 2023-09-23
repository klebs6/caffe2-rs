// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/memory_format_test.cpp]

lazy_static!{
    /*
    vector<vector<i64>> sizes = {{4, 4, 4, 4}, {4, 4, 1, 1}, {4, 1, 4, 4}, {4, 1, 4, 1}, {4, 1, 1, 4}, {1, 4, 1, 4}, {1, 4, 4, 1}};
    */
}

#[test] fn memory_format_test_set() {
    todo!();
    /*
      for (auto size : sizes) {
        Tensor t = rand(size);
        for (auto memory_format : {MemoryFormat::ChannelsLast, MemoryFormat::Contiguous}) {
          t.resize_(size, memory_format);
          EXPECT_TRUE(t.suggest_memory_format() == memory_format);
        }
      }

      Tensor t = rand({4, 1, 1, 1});
      EXPECT_TRUE(t.suggest_memory_format() == MemoryFormat::Contiguous);
      t.resize_({4, 1, 1, 1}, MemoryFormat::ChannelsLast);
      // TODO: Should be able to handle this after accumulated permutation is implemented;
      // Ambiguous case where we fallback to Contiguous;
      // This should be `EXPECT_TRUE(t.suggest_memory_format() == MemoryFormat::ChannelsLast);`
      EXPECT_TRUE(t.suggest_memory_format() == MemoryFormat::Contiguous);

    */
}

#[test] fn memory_format_test_transpose() {
    todo!();
    /*
    
      Tensor t = rand({2, 3, 4, 5});
      EXPECT_TRUE(t.suggest_memory_format() == MemoryFormat::Contiguous);
      t.transpose_(1, 3);
      EXPECT_TRUE(t.suggest_memory_format() != MemoryFormat::ChannelsLast);
      t.transpose_(2, 3);
      EXPECT_TRUE(t.suggest_memory_format() == MemoryFormat::ChannelsLast);
      t = rand({2, 3, 4, 5});
      t.transpose_(1, 2);
      EXPECT_TRUE(t.suggest_memory_format() != MemoryFormat::ChannelsLast);
      t = rand({2, 3, 4, 5});
      t.transpose_(2, 3);
      EXPECT_TRUE(t.suggest_memory_format() != MemoryFormat::ChannelsLast);

      // corner cases:
      t = rand({1, 4, 1, 4});
      t.transpose_(1, 3);
      EXPECT_TRUE(t.suggest_memory_format() != MemoryFormat::ChannelsLast);
      t = rand({1, 4, 1, 4});
      t.transpose_(1, 2);
      EXPECT_TRUE(t.suggest_memory_format() != MemoryFormat::ChannelsLast);
      t = rand({1, 4, 1, 4});
      t.transpose_(2, 3);
      EXPECT_TRUE(t.suggest_memory_format() != MemoryFormat::ChannelsLast);
      t = rand({1, 4, 1, 4});
      t.transpose_(2, 3);
      t.transpose_(1, 2);
      EXPECT_TRUE(t.suggest_memory_format() == MemoryFormat::ChannelsLast);

      t = rand({1, 4, 4, 1});
      t.transpose_(1, 3);
      EXPECT_TRUE(t.suggest_memory_format() != MemoryFormat::ChannelsLast);
      t = rand({1, 4, 4, 1});
      t.transpose_(1, 2);
      EXPECT_TRUE(t.suggest_memory_format() != MemoryFormat::ChannelsLast);
      t = rand({1, 4, 4, 1});
      t.transpose_(2, 3);
      EXPECT_TRUE(t.suggest_memory_format() != MemoryFormat::ChannelsLast);
      t = rand({1, 4, 4, 1});
      t.transpose_(2, 3);
      t.transpose_(1, 2);
      EXPECT_TRUE(t.suggest_memory_format() == MemoryFormat::ChannelsLast);

    */
}

#[inline] pub fn slice_step_two(
        t:      &mut Tensor,
        dim:    i32,
        format: MemoryFormat)  {
    
    todo!();
        /*
            t = t.slice(dim, 0, 3, 2);
      EXPECT_TRUE(t.suggest_memory_format() == format);
      t = t.slice(dim, 0, 3, 2);
      EXPECT_TRUE(t.suggest_memory_format() == format);
        */
}

#[test] fn memory_format_test_slice_step_two() {
    todo!();
    /*
    
      Tensor t = rand({4, 4, 4, 4});
      sliceStepTwo(t, 1, MemoryFormat::Contiguous);
      sliceStepTwo(t, 2, MemoryFormat::Contiguous);
      sliceStepTwo(t, 3, MemoryFormat::Contiguous);

      t = rand({4, 4, 4, 4});
      sliceStepTwo(t, 2, MemoryFormat::Contiguous);
      sliceStepTwo(t, 3, MemoryFormat::Contiguous);
      sliceStepTwo(t, 1, MemoryFormat::Contiguous);

      t = rand({4, 4, 4, 4});
      t.resize_({4, 4, 4, 4}, MemoryFormat::ChannelsLast);
      sliceStepTwo(t, 1, MemoryFormat::ChannelsLast);
      sliceStepTwo(t, 2, MemoryFormat::ChannelsLast);
      sliceStepTwo(t, 3, MemoryFormat::ChannelsLast);

      t = rand({4, 4, 4, 4});
      t.resize_({4, 4, 4, 4}, MemoryFormat::ChannelsLast);
      sliceStepTwo(t, 2, MemoryFormat::ChannelsLast);
      sliceStepTwo(t, 3, MemoryFormat::ChannelsLast);
      sliceStepTwo(t, 1, MemoryFormat::ChannelsLast);

      t = rand({4, 4, 1, 1});
      sliceStepTwo(t, 1, MemoryFormat::Contiguous);
      t = rand({4, 4, 1, 1});
      t.resize_({4, 4, 1, 1}, MemoryFormat::ChannelsLast);
      t = t.slice(1, 0, 3, 2);
      EXPECT_TRUE(t.suggest_memory_format() == MemoryFormat::ChannelsLast);
      t = t.slice(1, 0, 3, 2);
      // TODO: Should be able to handle this after accumulated permutation is implemented;
      // won't be able to tell how we ended up here
      // [4, 1, 1, 4]@[4, 4, 4, 1] slice twice at dim3
      // [4, 4, 1, 1]@[4, 1, 4, 4] slice twice at dim1
      // EXPECT_TRUE(t.suggest_memory_format() == MemoryFormat::ChannelsLast);
      EXPECT_TRUE(t.suggest_memory_format() == MemoryFormat::Contiguous);

      t = rand({4, 1, 4, 4});
      sliceStepTwo(t, 2, MemoryFormat::Contiguous);
      sliceStepTwo(t, 3, MemoryFormat::Contiguous);
      t = rand({4, 1, 4, 4});
      t.resize_({4, 1, 4, 4}, MemoryFormat::ChannelsLast);
      sliceStepTwo(t, 2, MemoryFormat::ChannelsLast);
      sliceStepTwo(t, 3, MemoryFormat::ChannelsLast);

      t = rand({4, 1, 1, 4});
      sliceStepTwo(t, 3, MemoryFormat::Contiguous);
      t = rand({4, 1, 1, 4});
      t.resize_({4, 1, 1, 4}, MemoryFormat::ChannelsLast);
      sliceStepTwo(t, 3, MemoryFormat::ChannelsLast);

      t = rand({4, 1, 4, 1});
      sliceStepTwo(t, 2, MemoryFormat::Contiguous);
      t = rand({4, 1, 4, 1});
      t.resize_({4, 1, 4, 1}, MemoryFormat::ChannelsLast);
      sliceStepTwo(t, 2, MemoryFormat::ChannelsLast);

    */
}

#[inline] pub fn slice_first(
        t:      &mut Tensor,
        dim:    i32,
        format: MemoryFormat)  {
    
    todo!();
        /*
            t = t.slice(dim, 0, 1, 1);
      EXPECT_TRUE(t.suggest_memory_format() == format);
        */
}

#[test] fn memory_format_test_slice_first() {
    todo!();
    /*
    
      Tensor t = rand({4, 4, 4, 4});
      sliceFirst(t, 1, MemoryFormat::Contiguous);
      sliceFirst(t, 2, MemoryFormat::Contiguous);
      sliceFirst(t, 3, MemoryFormat::Contiguous);

      t = rand({4, 4, 4, 4});
      sliceFirst(t, 2, MemoryFormat::Contiguous);
      sliceFirst(t, 3, MemoryFormat::Contiguous);
      sliceFirst(t, 1, MemoryFormat::Contiguous);

      t = rand({4, 4, 4, 4});
      t.resize_({4, 4, 4, 4}, MemoryFormat::ChannelsLast);
      sliceFirst(t, 1, MemoryFormat::ChannelsLast);
      sliceFirst(t, 2, MemoryFormat::ChannelsLast);
      sliceFirst(t, 3, MemoryFormat::ChannelsLast);

      t = rand({4, 4, 4, 4});
      t.resize_({4, 4, 4, 4}, MemoryFormat::ChannelsLast);
      sliceFirst(t, 2, MemoryFormat::ChannelsLast);
      sliceFirst(t, 3, MemoryFormat::ChannelsLast);
      sliceFirst(t, 1, MemoryFormat::ChannelsLast);

      t = rand({4, 4, 1, 1});
      sliceFirst(t, 1, MemoryFormat::Contiguous);
      t = rand({4, 4, 1, 1});
      t.resize_({4, 4, 1, 1}, MemoryFormat::ChannelsLast);
      sliceFirst(t, 1, MemoryFormat::ChannelsLast);

      t = rand({4, 1, 4, 4});
      sliceFirst(t, 2, MemoryFormat::Contiguous);
      sliceFirst(t, 3, MemoryFormat::Contiguous);
      t = rand({4, 1, 4, 4});
      t.resize_({4, 1, 4, 4}, MemoryFormat::ChannelsLast);
      sliceFirst(t, 2, MemoryFormat::ChannelsLast);
      sliceFirst(t, 3, MemoryFormat::ChannelsLast);

      t = rand({4, 1, 1, 4});
      sliceFirst(t, 3, MemoryFormat::Contiguous);
      t = rand({4, 1, 1, 4});
      t.resize_({4, 1, 1, 4}, MemoryFormat::ChannelsLast);
      sliceFirst(t, 3, MemoryFormat::ChannelsLast);

      t = rand({4, 1, 4, 1});
      sliceFirst(t, 2, MemoryFormat::Contiguous);
      t = rand({4, 1, 4, 1});
      t.resize_({4, 1, 4, 1}, MemoryFormat::ChannelsLast);
      // TODO: Should be able to handle this after accumulated permutation is implemented;
      // [4, 1, 4, 1]@[4, 1, 1, 1] after slice becomes [4, 1, 1, 1]@[4, 1, 1, 1]
      // sliceFirst(t, 2, MemoryFormat::ChannelsLast);
      sliceFirst(t, 2, MemoryFormat::Contiguous);

    */
}
