crate::ix!();

#[test] fn cpu_context_test_alloc_alignment() {
    todo!();
    /*
  for (int i = 1; i < 10; ++i) {
    auto data = CPUContext::New(i);
    EXPECT_EQ((reinterpret_cast<size_t>(data.get()) % gAlignment), 0);
    // data is freed when out of scope
  }
  */
}

#[test] fn cpu_context_test_alloc_dealloc() {
    todo!();
    /*
  auto data_ptr = CPUContext::New(10 * sizeof(float));
  float* data = static_cast<float*>(data_ptr.get());
  EXPECT_NE(data, nullptr);
  auto dst_data_ptr = CPUContext::New(10 * sizeof(float));
  float* dst_data = static_cast<float*>(dst_data_ptr.get());
  EXPECT_NE(dst_data, nullptr);
  for (int i = 0; i < 10; ++i) {
    data[i] = i;
  }
  DeviceOption option;
  CPUContext context(option);
  context.CopyToCPU<float>(10, data, dst_data);
  for (int i = 0; i < 10; ++i) {
    EXPECT_FLOAT_EQ(dst_data[i], i);
  }
  // data_ptr is freed when out of scope
  */
}

