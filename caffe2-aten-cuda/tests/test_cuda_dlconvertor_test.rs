crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/cuda_dlconvertor_test.cpp]

#[test] fn test_dlconvertor_dlconvertorcuda() {
    todo!();
    /*
    
      manual_seed(123);

      Tensor a = rand({3, 4}, kCUDA);
      DLManagedTensor* dlMTensor = toDLPack(a);

      Tensor b = fromDLPack(dlMTensor);

      ASSERT_TRUE(a.equal(b));

    */
}

#[test] fn test_dlconvertor_no_stridescuda() {
    todo!();
    /*
    
      manual_seed(123);

      Tensor a = rand({3, 4}, kCUDA);
      DLManagedTensor* dlMTensor = toDLPack(a);
      dlMTensor->dl_tensor.strides = nullptr;

      Tensor b = fromDLPack(dlMTensor);

      ASSERT_TRUE(a.equal(b));

    */
}

#[test] fn test_dlconvertor_dlconvertorcudahip() {
    todo!();
    /*
    
      if (!is_available())
        return;
      manual_seed(123);

      Tensor a = rand({3, 4}, kCUDA);
      DLManagedTensor* dlMTensor = toDLPack(a);

    #if AT_ROCM_ENABLED()
      ASSERT_TRUE(dlMTensor->dl_tensor.device.device_type == DLDeviceType::kDLROCM);
    #else
      ASSERT_TRUE(dlMTensor->dl_tensor.device.device_type == DLDeviceType::kDLGPU);
    #endif

      Tensor b = fromDLPack(dlMTensor);

      ASSERT_TRUE(a.equal(b));

    */
}
