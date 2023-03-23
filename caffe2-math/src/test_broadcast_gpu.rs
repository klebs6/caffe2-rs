crate::ix!();

instantiate_test_case_p!{
    /*
    GemmBatchedGPUTrans,
    GemmBatchedGPUTest,
    testing::Combine(testing::Bool(), testing::Bool())
    */
}

#[test] fn broadcast_gpu_test_broadcast_gpu_float_test() {
    todo!();
    /*
      if (!HasCudaGPU()) {
        return;
      }
      RunBroadcastTest({2}, {2}, {1.0f, 2.0f}, {1.0f, 2.0f});
      RunBroadcastTest({1}, {2}, {1.0f}, {1.0f, 1.0f});
      RunBroadcastTest({1}, {2, 2}, {1.0f}, {1.0f, 1.0f, 1.0f, 1.0f});
      RunBroadcastTest({2, 1}, {2, 2}, {1.0f, 2.0f}, {1.0f, 1.0f, 2.0f, 2.0f});
      RunBroadcastTest(
          {2, 1},
          {2, 2, 2},
          {1.0f, 2.0f},
          {1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f});
  */
}
