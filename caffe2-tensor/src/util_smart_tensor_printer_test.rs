crate::ix!();

#[inline] pub fn my_to_string<T>(value: &T) -> String {
    todo!();
    /*
        return to_string(value);
    */
}

#[inline] pub fn expect_stderr_contains<T>(values: &Vec<T>) {
    todo!();
    /*
        std::string captured_stderr = testing::internal::GetCapturedStderr();
      for (const auto& value : values) {
        std::string stringValue = my_to_string(value);
        EXPECT_TRUE(captured_stderr.find(stringValue) != std::string::npos);
      }
    */
}

#[inline] pub fn print_tensor_and_check<T>(values: &Vec<T>) {
    todo!();
    /*
        testing::internal::CaptureStderr();

      Tensor tensor =
          TensorCPUFromValues<T>({static_cast<int64_t>(values.size())}, values);

      SmartTensorPrinter::PrintTensor(tensor);
      expect_stderr_contains(values);
    */
}

// We need real glog for this test to pass
#[cfg(caffe2_use_google_glog)]
#[cfg(not(target_os = "osx"))]
#[test] fn SmartTensorPrinterTest_SimpleTest() {
    todo!();
    /*
      printTensorAndCheck(std::vector<int>{1, 2, 3, 4, 5});
      printTensorAndCheck(std::vector<std::string>{"bob", "alice", "facebook"});
  */
}
