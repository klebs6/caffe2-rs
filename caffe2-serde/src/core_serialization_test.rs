crate::ix!();

/**
  | NOLINTNEXTLINE: cppcoreguidelines-avoid-c-arrays
  |
  */
define_bool!{
    caffe2_test_generate_unknown_dtype_blob,
    false,
    "Recompute and log the serialized blob data
     for the TensorSerialization.TestUnknownDType
     test" 
}

/**
  | This data was computed by serializing
  | a 10-element int32_t tensor, but with the
  | data_type field set to 4567.
  |
  | This allows us to test the behavior of the code
  | when deserializing data from a future version
  | of the code that has new data types that our
  | code does not understand.
  */
pub const kFutureDtypeBlob: [u8; 61] = hex!{
    "0a 09 74 65 73 74 5f 62 6c 6f 62 12 06 54 65 6e
     73 6f 72 1a 28 08 0a 08 01 10 d7 23 22 0a 00 01
     02 03 04 05 06 07 08 09 3a 09 74 65 73 74 5f 62
     6c 6f 62 42 02 08 00 5a 04 08 00 10 0a"
};

/**
  | The same tensor with the data_type actually
  | set to TensorProto_DataType_INT32
  |
  */
pub const kInt32DtypeBlob: [u8; 60] = hex!{
    "0a 09 74 65 73 74 5f 62 6c 6f 62 12 06 54 65 6e
     73 6f 72 1a 27 08 0a 08 01 10 02 22 0a 00 01 02
     03 04 05 06 07 08 09 3a 09 74 65 73 74 5f 62 6c
     6f 62 42 02 08 00 5a 04 08 00 10 0a"
}; 

#[inline] pub fn log_blob(data: &str)  {
    
    todo!();
    /*
        constexpr size_t kBytesPerLine = 16;
      constexpr size_t kCharsPerEncodedByte = 4;
      std::vector<char> hexStr;
      hexStr.resize((kBytesPerLine * kCharsPerEncodedByte) + 1);
      hexStr[kBytesPerLine * kCharsPerEncodedByte] = '\0';
      size_t lineIdx = 0;
      for (char c : data) {
        snprintf(
            hexStr.data() + (kCharsPerEncodedByte * lineIdx),
            kCharsPerEncodedByte + 1,
            "\\x%02x",
            static_cast<unsigned int>(c));
        ++lineIdx;
        if (lineIdx >= kBytesPerLine) {
          LOG(INFO) << "    \"" << hexStr.data() << "\"";
          lineIdx = 0;
        }
      }
      if (lineIdx > 0) {
        hexStr[lineIdx * kCharsPerEncodedByte] = '\0';
        LOG(INFO) << "    \"" << hexStr.data() << "\"";
      }
    */
}

#[test] fn tensor_serialization_test_unknown_dtype() {
    todo!();
    /*
      // This code was used to generate the blob data listed above.
      constexpr size_t kTestTensorSize = 10;
      if (FLAGS_caffe2_test_generate_unknown_dtype_blob) {
        Blob blob;
        auto* blobTensor = BlobGetMutableTensor(&blob, CPU);
        blobTensor->Resize(kTestTensorSize, 1);
        auto *tensorData = blobTensor->mutable_data<int32_t>();
        for (int n = 0; n < kTestTensorSize; ++n) {
          tensorData[n] = n;
        }
        auto data = SerializeBlob(blob, "test_blob");
        LOG(INFO) << "test blob: size=" << data.size();
        logBlob(data);
      }

      // Test deserializing the normal INT32 data,
      // just to santity check that deserialization works
      Blob i32Blob;
      DeserializeBlob(std::string(kInt32DtypeBlob), &i32Blob);
      const auto& tensor = BlobGetTensor(i32Blob, c10::DeviceType::CPU);
      EXPECT_EQ(kTestTensorSize, tensor.numel());
      EXPECT_EQ(TypeMeta::Make<int32_t>(), tensor.dtype());
      const auto* tensor_data = tensor.template data<int32_t>();
      for (int i = 0; i < kTestTensorSize; ++i) {
        EXPECT_EQ(static_cast<float>(i), tensor_data[i]);
      }

      // Now test deserializing our blob with an unknown data type
      Blob futureDtypeBlob;
      try {
        DeserializeBlob(std::string(kFutureDtypeBlob), &futureDtypeBlob);
        FAIL() << "DeserializeBlob() should have failed";
      } catch (const std::exception& ex) {
        EXPECT_STREQ(
            "Cannot deserialize tensor: unrecognized data type", ex.what());
      }
  */
}
