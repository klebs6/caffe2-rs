crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/reduce_ops_test.cpp]

#[test] fn reduce_ops_test_max_values_and_min() {
    todo!();
    /*
    
      const int W = 10;
      const int H = 10;
      if (hasCUDA()) {
        for (const auto dtype : {kHalf, kFloat, kDouble, kShort, kInt, kLong}) {
          auto a = rand({H, W}, TensorOptions(kCUDA).dtype(kHalf));
          ASSERT_FLOAT_EQ(
            a.amax(IntArrayRef{0, 1}).item<double>(),
            a.max().item<double>()
          );
          ASSERT_FLOAT_EQ(
            a.amin(IntArrayRef{0, 1}).item<double>(),
            a.min().item<double>()
          );
        }
      }

    */
}
