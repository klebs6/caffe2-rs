crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/memory_overlapping_test.cpp]
lazy_static!{
    /*
    vector<vector<i64>> sizes = {{1, 2, 3}, {1, 3, 2}, {2, 1, 3}, {3, 1, 2}, {3, 2, 1}, {2, 3, 1}};
    */
}

#[test] fn memory_overlap_test_tensor_expanded() {
    todo!();
    /*
    
      for (auto size : sizes) {
        Tensor t = ones({1}).expand(size);
        EXPECT_FALSE(t.is_contiguous());
        EXPECT_FALSE(t.is_non_overlapping_and_dense());
      }

    */
}

#[test] fn memory_overlap_test_scalar_expanded() {
    todo!();
    /*
    
      for (auto size : sizes) {
        Tensor t = tensor(1).expand(size);
        EXPECT_FALSE(t.is_contiguous());
        EXPECT_FALSE(t.is_non_overlapping_and_dense());
      }

    */
}

#[test] fn memory_overlap_test_non_contiguous_tensor() {
    todo!();
    /*
    
      for (auto size : sizes) {
        Tensor t = rand(size).transpose(1, 2).transpose(0, 2);
        if (!t.is_contiguous()) {
          EXPECT_TRUE(t.is_non_overlapping_and_dense());
        }
      }

    */
}

#[test] fn memory_overlap_test_non_contiguous_expanded_tensor() {
    todo!();
    /*
    
      for (auto size : sizes) {
        Tensor t = rand(size).transpose(1, 2).transpose(0, 2);
        if (!t.is_contiguous()) {
          for (auto size_to_add : {1, 2, 3, 4}) {
            auto transpose_size = t.sizes().vec();
            vector<i64> expanded_size(transpose_size);
            expanded_size.insert(expanded_size.begin(), size_to_add);
            auto expanded = t.expand(expanded_size);
            EXPECT_FALSE(t.is_contiguous());
            if (size_to_add == 1) {
              EXPECT_TRUE(expanded.is_non_overlapping_and_dense());
            } else {
              EXPECT_FALSE(expanded.is_non_overlapping_and_dense());
            }
          }
        }
      }

    */
}

#[test] fn memory_overlap_test_contiguous_tensor() {
    todo!();
    /*
    
      for (auto size : sizes) {
        Tensor t = rand(size);
        EXPECT_TRUE(t.is_contiguous());
        EXPECT_TRUE(t.is_non_overlapping_and_dense());
      }

    */
}

#[test] fn memory_overlap_test_contiguous_expanded_tensor() {
    todo!();
    /*
    
      for (auto size : sizes) {
        Tensor t = rand(size);
        for (auto size_to_add : {1, 2, 3, 4}) {
          vector<i64> expanded_size(size);
          expanded_size.insert(expanded_size.begin(), size_to_add);
          auto expanded = t.expand(expanded_size);
          EXPECT_TRUE(t.is_contiguous());
          EXPECT_TRUE(t.is_non_overlapping_and_dense());
        }
      }

    */
}
