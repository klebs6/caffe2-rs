crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/dlconvertor_test.cpp]

#[test] fn test_dlconvertor() {
    todo!();
    /*
    
      manual_seed(123);

      Tensor a = rand({3, 4});
      DLManagedTensor* dlMTensor = toDLPack(a);

      Tensor b = fromDLPack(dlMTensor);

      ASSERT_TRUE(a.equal(b));

    */
}

#[test] fn test_dlconvertor_no_strides() {
    todo!();
    /*
    
      manual_seed(123);

      Tensor a = rand({3, 4});
      DLManagedTensor* dlMTensor = toDLPack(a);
      dlMTensor->dl_tensor.strides = nullptr;

      Tensor b = fromDLPack(dlMTensor);

      ASSERT_TRUE(a.equal(b));

    */
}
