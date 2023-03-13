crate::ix!();

pub struct BlobTestFoo {
    val: i32,
}

pub struct BlobTestBar {}

pub struct BlobTestNonDefaultConstructible {
    val: i32,
}

impl BlobTestNonDefaultConstructible {
    
    pub fn new(x: i32) -> Self {
        todo!();
        /*
            : val(x)
        */
    }
}

caffe_known_type![BlobTestFoo];
caffe_known_type![BlobTestBar];
caffe_known_type![BlobTestNonDefaultConstructible];

pub struct BlobTestFooSerializer {
    base: dyn BlobSerializerBase,
}

impl BlobSerializerBase for BlobTestFooSerializer {
    
    /**
      | Serializes a Blob. Note that this blob
      | has to contain Tensor, otherwise this
      | function produces a fatal error.
      |
      */
    #[inline] fn serialize(
        &mut self, 
        pointer:   *const c_void,
        type_meta: TypeMeta,
        name:      &String,
        acceptor:  SerializationAcceptor,
        options:   Option<&BlobSerializationOptions>)  
    {
        todo!();
        /*
            CAFFE_ENFORCE(typeMeta.Match<BlobTestFoo>());

        BlobProto blob_proto;
        blob_proto.set_name(name);
        blob_proto.set_type("BlobTestFoo");
        // For simplicity we will just serialize the 4-byte content as a string.
        blob_proto.set_content(std::string(
            reinterpret_cast<const char*>(
                &static_cast<const BlobTestFoo*>(pointer)->val),
            sizeof(int32_t)));
        acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
        */
    }
}

pub struct BlobTestFooDeserializer {
    base: dyn BlobDeserializerBase,
}

impl BlobTestFooDeserializer {
    
    #[inline] pub fn deserialize(&mut self, 
        proto: &BlobProto, 
        blob:  *mut Blob)  {
        
        todo!();
        /*
            blob->GetMutable<BlobTestFoo>()->val =
            reinterpret_cast<const int32_t*>(proto.content().c_str())[0];
        */
    }
}

register_blob_serializer![
    /*(TypeMeta::Id<BlobTestFoo>()), 
      BlobTestFooSerializer*/
];

register_blob_deserializer![
    /*
    BlobTestFoo,                 
    BlobTestFooDeserializer
    */
];

#[test] fn blob_test_blob() {
    todo!();
    /*
      Blob blob;

      int* int_unused CAFFE2_UNUSED = blob.GetMutable<int>();
      EXPECT_TRUE(blob.IsType<int>());
      EXPECT_FALSE(blob.IsType<BlobTestFoo>());
      EXPECT_FALSE(BlobIsTensorType(blob, CPU));

      BlobTestFoo* foo_unused CAFFE2_UNUSED = blob.GetMutable<BlobTestFoo>();
      EXPECT_TRUE(blob.IsType<BlobTestFoo>());
      EXPECT_FALSE(blob.IsType<int>());
      EXPECT_FALSE(BlobIsTensorType(blob, CPU));

      Tensor* tensor_unused CAFFE2_UNUSED = BlobGetMutableTensor(&blob, CPU);
      EXPECT_TRUE(BlobIsTensorType(blob, CPU));
      EXPECT_FALSE(blob.IsType<BlobTestFoo>());
      EXPECT_FALSE(blob.IsType<int>());
      */
}

#[test] fn blob_test_blob_uninitialized() {
    todo!();
    /*
      Blob blob;
      ASSERT_THROW(blob.Get<int>(), EnforceNotMet);
      */
}

#[test] fn blob_test_blob_wrong_type() {
    todo!();
    /*
      Blob blob;
      BlobTestFoo* foo_unused CAFFE2_UNUSED = blob.GetMutable<BlobTestFoo>();
      EXPECT_TRUE(blob.IsType<BlobTestFoo>());
      EXPECT_FALSE(blob.IsType<int>());
      // When not null, we should only call with the right type.
      EXPECT_NE(&blob.Get<BlobTestFoo>(), nullptr);
      ASSERT_THROW(blob.Get<int>(), EnforceNotMet);
      */
}

#[test] fn blob_test_blob_reset() {
    todo!();
    /*
      Blob blob;
      std::unique_ptr<BlobTestFoo> foo(new BlobTestFoo());
      EXPECT_TRUE(blob.Reset(foo.release()) != nullptr);
      // Also test that Reset works.
      blob.Reset();
      */
}

#[test] fn blob_test_blob_move() {
    todo!();
    /*
      Blob blob1;
      std::unique_ptr<BlobTestFoo> foo(new BlobTestFoo());
      auto* fooPtr = foo.get();
      EXPECT_TRUE(blob1.Reset(foo.release()) != nullptr);
      Blob blob2;
      blob2 = std::move(blob1);
      ASSERT_THROW(blob1.Get<BlobTestFoo>(), EnforceNotMet);
      EXPECT_EQ(&blob2.Get<BlobTestFoo>(), fooPtr);
      Blob blob3{std::move(blob2)};
      EXPECT_EQ(&blob3.Get<BlobTestFoo>(), fooPtr);
      */
}

#[test] fn blob_test_blob_non_constructible() {
    todo!();
    /*
      Blob blob;
      ASSERT_THROW(blob.Get<BlobTestNonDefaultConstructible>(), EnforceNotMet);
      // won't work because it's not default constructible
      // blob.GetMutable<BlobTestNonDefaultConstructible>();
      EXPECT_FALSE(
          blob.GetMutableOrNull<BlobTestNonDefaultConstructible>() != nullptr);
      EXPECT_TRUE(blob.Reset(new BlobTestNonDefaultConstructible(42)) != nullptr);
      ASSERT_NO_THROW(blob.Get<BlobTestNonDefaultConstructible>());
      ASSERT_TRUE(
          blob.GetMutableOrNull<BlobTestNonDefaultConstructible>() != nullptr);
      EXPECT_EQ(blob.Get<BlobTestNonDefaultConstructible>().val, 42);
      blob.GetMutableOrNull<BlobTestNonDefaultConstructible>()->val = 37;
      EXPECT_EQ(blob.Get<BlobTestNonDefaultConstructible>().val, 37);
      */
}

#[test] fn blob_test_blob_share_external_pointer() {
    todo!();
    /*
      Blob blob;
      std::unique_ptr<BlobTestFoo> foo(new BlobTestFoo());
      EXPECT_EQ(blob.ShareExternal<BlobTestFoo>(foo.get()), foo.get());
      EXPECT_TRUE(blob.IsType<BlobTestFoo>());
      // Also test that Reset works.
      blob.Reset();
      */
}

#[test] fn blob_test_blob_share_external_object() {
    todo!();
    /*
      Blob blob;
      BlobTestFoo foo;
      EXPECT_EQ(blob.ShareExternal<BlobTestFoo>(&foo), &foo);
      EXPECT_TRUE(blob.IsType<BlobTestFoo>());
      // Also test that Reset works.
      blob.Reset();
      */
}

#[test] fn blob_test_string_serialization() {
    todo!();
    /*
      const std::string kTestString = "Hello world?";
      Blob blob;
      *blob.GetMutable<std::string>() = kTestString;

      string serialized = SerializeBlob(blob, "test");
      BlobProto proto;
      CHECK(proto.ParseFromString(serialized));
      EXPECT_EQ(proto.name(), "test");
      EXPECT_EQ(proto.type(), "std::string");
      EXPECT_FALSE(proto.has_tensor());
      EXPECT_EQ(proto.content(), kTestString);
      */
}

#[test] fn tensor_non_typed_test_tensor_change_type() {
    todo!();
    /*
      vector<int> dims(3);
      dims[0] = 2;
      dims[1] = 3;
      dims[2] = 5;
      Tensor tensor(dims, CPU);

      auto* ptr = tensor.mutable_data<int>();
      EXPECT_TRUE(ptr != nullptr);
      EXPECT_TRUE(tensor.data<int>() != nullptr);
      EXPECT_TRUE(tensor.dtype().Match<int>());

      // int and float are same size, so should retain the pointer
      // NB: this is only true when the use_count of the underlying Storage is 1, if
      // the underlying Storage is shared between multiple Tensors We'll create a
      // new Storage when the data type changes
      EXPECT_TRUE(tensor.mutable_data<float>() == (float*)ptr);
      EXPECT_TRUE(tensor.data<float>() == (const float*)ptr);
      EXPECT_TRUE(tensor.dtype().Match<float>());

      // at::Half is smaller, so still should share buffer
      EXPECT_TRUE(tensor.mutable_data<at::Half>() == (at::Half*)ptr);
      EXPECT_TRUE(tensor.data<at::Half>() == (const at::Half*)ptr);
      EXPECT_TRUE(tensor.dtype().Match<at::Half>());

      // share the data with other tensor so that the pointer won't be reused
      // when we reallocate
      Tensor other_tensor = tensor.Alias();
      // but double is bigger, so it should allocate a new one
      auto* doubleptr = tensor.mutable_data<double>();
      EXPECT_TRUE(doubleptr != (double*)ptr);
      EXPECT_TRUE(doubleptr != nullptr);
      EXPECT_TRUE(tensor.data<double>() != nullptr);
      EXPECT_TRUE(tensor.dtype().Match<double>());
      */
}

#[test] fn tensor_non_typed_test_non_default_constructible() {
    todo!();
    /*
      vector<int> dims(3);
      dims[0] = 2;
      dims[1] = 3;
      dims[2] = 5;
      Tensor tensor(dims, CPU);

      // this doesn't compile - good!
      // auto* ptr = tensor.mutable_data<BlobTestNonDefaultConstructible>();
      EXPECT_THROW(
          tensor.raw_mutable_data(
              TypeMeta::Make<BlobTestNonDefaultConstructible>()),
          EnforceNotMet);
      */
}

type TensorCPUTest<T>      = Test::<T>;
type TensorCPUDeathTest<T> = Test::<T>;
type TensorTypes           = Types<(u8, i32, f32)>;

typed_test_case![TensorCPUTest,      TensorTypes];
typed_test_case![TensorCPUDeathTest, TensorTypes];

#[test] fn tensor_cpu_test_tensor_initialized_empty() {
    todo!();
    /*
      Tensor tensor(CPU);
      EXPECT_EQ(tensor.dim(), 1);
      EXPECT_EQ(tensor.numel(), 0);
      vector<int> dims(3);
      dims[0] = 2;
      dims[1] = 3;
      dims[2] = 5;
      tensor.Resize(dims);
      EXPECT_EQ(tensor.dim(), 3);
      EXPECT_EQ(tensor.dim32(0), 2);
      EXPECT_EQ(tensor.dim32(1), 3);
      EXPECT_EQ(tensor.dim32(2), 5);
      EXPECT_EQ(tensor.numel(), 2 * 3 * 5);
      EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
      EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
      */
}

#[test] fn tensor_cpu_test_tensor_initialized_non_empty() {
    todo!();
    /*
      vector<int> dims(3);
      dims[0] = 2;
      dims[1] = 3;
      dims[2] = 5;
      Tensor tensor(dims, CPU);
      EXPECT_EQ(tensor.dim(), 3);
      EXPECT_EQ(tensor.dim32(0), 2);
      EXPECT_EQ(tensor.dim32(1), 3);
      EXPECT_EQ(tensor.dim32(2), 5);
      EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
      EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
      dims[0] = 7;
      dims[1] = 11;
      dims[2] = 13;
      dims.push_back(17);
      tensor.Resize(dims);
      EXPECT_EQ(tensor.dim(), 4);
      EXPECT_EQ(tensor.dim32(0), 7);
      EXPECT_EQ(tensor.dim32(1), 11);
      EXPECT_EQ(tensor.dim32(2), 13);
      EXPECT_EQ(tensor.dim32(3), 17);
      EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
      EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
      */
}

#[test] fn tensor_cpu_test_tensor_initialized_zero_dim() {
    todo!();
    /*
      vector<int> dims(3);
      dims[0] = 2;
      dims[1] = 0;
      dims[2] = 5;
      Tensor tensor(dims, CPU);
      EXPECT_EQ(tensor.dim(), 3);
      EXPECT_EQ(tensor.dim32(0), 2);
      EXPECT_EQ(tensor.dim32(1), 0);
      EXPECT_EQ(tensor.dim32(2), 5);
      EXPECT_TRUE(tensor.mutable_data<TypeParam>() == nullptr);
      EXPECT_TRUE(tensor.data<TypeParam>() == nullptr);
      */
}

#[test] fn tensor_cpu_test_tensor_resize_zero_dim() {
    todo!();
    /*
      vector<int> dims(3);
      dims[0] = 2;
      dims[1] = 3;
      dims[2] = 5;
      Tensor tensor(dims, CPU);
      EXPECT_EQ(tensor.dim(), 3);
      EXPECT_EQ(tensor.dim32(0), 2);
      EXPECT_EQ(tensor.dim32(1), 3);
      EXPECT_EQ(tensor.dim32(2), 5);
      EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
      EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);

      dims[0] = 7;
      dims[1] = 0;
      dims[2] = 13;
      tensor.Resize(dims);
      EXPECT_EQ(tensor.numel(), 0);
      EXPECT_EQ(tensor.dim(), 3);
      EXPECT_EQ(tensor.dim32(0), 7);
      EXPECT_EQ(tensor.dim32(1), 0);
      EXPECT_EQ(tensor.dim32(2), 13);
      // output value can be arbitrary, but the call to data() shouldn't crash
      tensor.mutable_data<TypeParam>();
      tensor.data<TypeParam>();
      */
}

#[test] fn tensor_cpu_test_tensor_initialized_scalar() {
    todo!();
    /*
      vector<int> dims;
      Tensor tensor(dims, CPU);
      EXPECT_EQ(tensor.dim(), 0);
      EXPECT_EQ(tensor.numel(), 1);
      EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
      EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
      */
}

#[test] fn tensor_cpu_test_tensor_alias() {
    todo!();
    /*
      vector<int> dims(3);
      dims[0] = 2;
      dims[1] = 3;
      dims[2] = 5;
      Tensor tensor(dims, CPU);
      EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
      Tensor other_tensor = tensor.Alias();
      EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
      EXPECT_TRUE(other_tensor.data<TypeParam>() != nullptr);
      EXPECT_EQ(tensor.data<TypeParam>(), other_tensor.data<TypeParam>());
      // Set one value, check the other
      for (int i = 0; i < tensor.numel(); ++i) {
        tensor.mutable_data<TypeParam>()[i] = i;
        EXPECT_EQ(other_tensor.data<TypeParam>()[i], i);
      }
      */
}

#[test] fn tensor_cpu_test_tensor_share_data_raw_pointer() {
    todo!();
    /*
      vector<int> dims(3);
      dims[0] = 2;
      dims[1] = 3;
      dims[2] = 5;
      std::unique_ptr<TypeParam[]> raw_buffer(new TypeParam[2 * 3 * 5]);
      Tensor tensor(dims, CPU);
      tensor.ShareExternalPointer(raw_buffer.get());
      EXPECT_EQ(tensor.mutable_data<TypeParam>(), raw_buffer.get());
      EXPECT_EQ(tensor.data<TypeParam>(), raw_buffer.get());
      // Set one value, check the other
      for (int i = 0; i < tensor.numel(); ++i) {
        raw_buffer.get()[i] = i;
        EXPECT_EQ(tensor.data<TypeParam>()[i], i);
      }
      */
}

#[test] fn tensor_cpu_test_tensor_share_data_raw_pointer_with_meta() {
    todo!();
    /*
      vector<int> dims(3);
      dims[0] = 2;
      dims[1] = 3;
      dims[2] = 5;
      std::unique_ptr<TypeParam[]> raw_buffer(new TypeParam[2 * 3 * 5]);
      Tensor tensor(dims, CPU);
      TypeMeta meta = TypeMeta::Make<TypeParam>();
      tensor.ShareExternalPointer(raw_buffer.get(), meta);
      EXPECT_EQ(tensor.mutable_data<TypeParam>(), raw_buffer.get());
      EXPECT_EQ(tensor.data<TypeParam>(), raw_buffer.get());
      // Set one value, check the other
      for (int i = 0; i < tensor.numel(); ++i) {
        raw_buffer.get()[i] = i;
        EXPECT_EQ(tensor.data<TypeParam>()[i], i);
      }
      */
}

#[test] fn tensor_cpu_test_tensor_alias_can_use_different_shapes() {
    todo!();
    /*
      vector<int> dims(3);
      dims[0] = 2;
      dims[1] = 3;
      dims[2] = 5;
      vector<int> alternate_dims(1);
      alternate_dims[0] = 2 * 3 * 5;
      Tensor tensor(dims, CPU);
      EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
      Tensor other_tensor = tensor.Alias();
      other_tensor.Resize(alternate_dims);
      EXPECT_EQ(other_tensor.dim(), 1);
      EXPECT_EQ(other_tensor.dim32(0), alternate_dims[0]);
      EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
      EXPECT_TRUE(other_tensor.data<TypeParam>() != nullptr);
      EXPECT_EQ(tensor.data<TypeParam>(), other_tensor.data<TypeParam>());
      // Set one value, check the other
      for (int i = 0; i < tensor.numel(); ++i) {
        tensor.mutable_data<TypeParam>()[i] = i;
        EXPECT_EQ(other_tensor.data<TypeParam>()[i], i);
      }
      */
}

#[test] fn tensor_cpu_test_no_longer_aliass_after_numel_changes() {
    todo!();
    /*
      vector<int> dims(3);
      dims[0] = 2;
      dims[1] = 3;
      dims[2] = 5;
      Tensor tensor(dims, CPU);
      EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
      Tensor other_tensor = tensor.Alias();
      EXPECT_EQ(tensor.data<TypeParam>(), other_tensor.data<TypeParam>());
      auto* old_pointer = other_tensor.data<TypeParam>();

      dims[0] = 7;
      tensor.Resize(dims);
      EXPECT_EQ(old_pointer, other_tensor.data<TypeParam>());
      EXPECT_NE(old_pointer, tensor.mutable_data<TypeParam>());
      */
}

#[test] fn tensor_cpu_test_no_longer_alias_after_free_memory() {
    todo!();
    /*
      vector<int> dims(3);
      dims[0] = 2;
      dims[1] = 3;
      dims[2] = 5;
      Tensor tensor(dims, CPU);
      EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
      Tensor other_tensor = tensor.Alias();
      EXPECT_EQ(tensor.data<TypeParam>(), other_tensor.data<TypeParam>());
      auto* old_pointer = other_tensor.data<TypeParam>();

      tensor.FreeMemory();
      EXPECT_EQ(old_pointer, other_tensor.data<TypeParam>());
      EXPECT_NE(old_pointer, tensor.mutable_data<TypeParam>());
      */
}

#[test] fn tensor_cpu_test_keep_on_shrink() {
    todo!();
    /*
      // Set flags (defaults)
      FLAGS_caffe2_keep_on_shrink = true;
      FLAGS_caffe2_max_keep_on_shrink_memory = LLONG_MAX;

      vector<int> dims{2, 3, 5};
      Tensor tensor(dims, CPU);
      TypeParam* ptr = tensor.mutable_data<TypeParam>();
      EXPECT_TRUE(ptr != nullptr);
      // Expanding - will reallocate
      tensor.Resize(3, 4, 6);
      TypeParam* larger_ptr = tensor.mutable_data<TypeParam>();
      EXPECT_TRUE(larger_ptr != nullptr);

      // This check can fail when malloc() returns the same recently freed address
      // EXPECT_NE(ptr, larger_ptr);

      // Shrinking - will not reallocate
      tensor.Resize(1, 2, 4);
      TypeParam* smaller_ptr = tensor.mutable_data<TypeParam>();
      EXPECT_TRUE(smaller_ptr != nullptr);
      EXPECT_EQ(larger_ptr, smaller_ptr);
      // resize to 0 in the meantime;
      tensor.Resize(3, 0, 6);
      // Expanding but still under capacity - will not reallocate
      tensor.Resize(2, 3, 5);
      TypeParam* new_ptr = tensor.mutable_data<TypeParam>();
      EXPECT_TRUE(new_ptr != nullptr);
      EXPECT_EQ(larger_ptr, new_ptr);
      */
}

#[test] fn tensor_cpu_test_max_keep_on_shrink() {
    todo!();
    /*
      // Set flags
      FLAGS_caffe2_keep_on_shrink = true;
      FLAGS_caffe2_max_keep_on_shrink_memory = 8 * 4 * sizeof(TypeParam);

      vector<int> dims{1, 8, 8};
      Tensor tensor(dims, CPU);
      TypeParam* ptr = tensor.mutable_data<TypeParam>();
      EXPECT_TRUE(ptr != nullptr);
      // Shrinking - will not reallocate
      tensor.Resize(1, 7, 8);
      TypeParam* smaller_ptr = tensor.mutable_data<TypeParam>();
      EXPECT_TRUE(smaller_ptr != nullptr);
      EXPECT_EQ(ptr, smaller_ptr);
      // Resize to more than maximum shrink, should reallocate
      tensor.Resize(1, 1, 8);
      TypeParam* new_ptr = tensor.mutable_data<TypeParam>();
      EXPECT_TRUE(new_ptr != nullptr);

      // This check can fail when malloc() returns the same recently freed address
      // EXPECT_NE(ptr, new_ptr);

      // Restore default flags
      FLAGS_caffe2_max_keep_on_shrink_memory = LLONG_MAX;
      */
}

#[test] fn tensor_cpu_death_test_cannot_access_raw_data_when_empty() {
    todo!();
    /*
      Tensor tensor(CPU);
      EXPECT_EQ(tensor.dim(), 1);
      EXPECT_EQ(tensor.numel(), 0);
      ASSERT_ANY_THROW(tensor.raw_data());
      */
}

#[test] fn tensor_cpu_death_test_cannot_access_data_when_empty() {
    todo!();
    /*
      Tensor tensor(CPU);
      EXPECT_EQ(tensor.dim(), 1);
      EXPECT_EQ(tensor.numel(), 0);
      ASSERT_ANY_THROW(tensor.data<TypeParam>());
      */
}

#[test] fn tensor_test_tensor_non_fundamental_type() {
    todo!();
    /*
      Tensor tensor(vector<int>{2, 3, 4}, CPU);
      EXPECT_TRUE(tensor.mutable_data<std::string>() != nullptr);
      const std::string* ptr = tensor.data<std::string>();
      for (int i = 0; i < tensor.numel(); ++i) {
        EXPECT_TRUE(ptr[i] == "");
      }
      */
}

#[test] fn tensor_test_tensor_non_fundamental_type_clone() {
    todo!();
    /*
      Tensor tensor(vector<int>{2, 3, 4}, CPU);
      std::string* ptr = tensor.mutable_data<std::string>();
      EXPECT_TRUE(ptr != nullptr);
      for (int i = 0; i < tensor.numel(); ++i) {
        EXPECT_TRUE(ptr[i] == "");
        ptr[i] = "filled";
      }
      Tensor dst_tensor = tensor.Clone();
      const std::string* dst_ptr = dst_tensor.data<std::string>();
      for (int i = 0; i < dst_tensor.numel(); ++i) {
        EXPECT_TRUE(dst_ptr[i] == "filled");
      }
      // Change the original tensor
      for (int i = 0; i < tensor.numel(); ++i) {
        EXPECT_TRUE(ptr[i] == "filled");
        ptr[i] = "changed";
      }
      // Confirm that the cloned tensor is not affect
      for (int i = 0; i < dst_tensor.numel(); ++i) {
        EXPECT_TRUE(dst_ptr[i] == "filled");
      }
      */
}

#[test] fn tensor_test_tensor_64bit_dimension() {
    todo!();
    /*
      // Initialize a large tensor.
      int64_t large_number =
          static_cast<int64_t>(int::max) + 1;
      Tensor tensor(vector<int64_t>{large_number}, CPU);
      EXPECT_EQ(tensor.dim(), 1);
      EXPECT_EQ(tensor.size(0), large_number);
      EXPECT_EQ(tensor.numel(), large_number);
      try {
        EXPECT_TRUE(tensor.mutable_data<char>() != nullptr);
      } catch (const EnforceNotMet& e) {
        string msg = e.what();
        size_t found = msg.find("posix_memalign");
        if (found != string::npos) {
          msg = msg.substr(0, msg.find('\n'));
          LOG(WARNING) << msg;
          LOG(WARNING) << "Out of memory issue with posix_memalign;\n";
          return;
        } else {
          throw e;
        }
      }
      EXPECT_EQ(tensor.nbytes(), large_number * sizeof(char));
      EXPECT_EQ(tensor.itemsize(), sizeof(char));
      // Try to go even larger, but this time we will not do mutable_data because we
      // do not have a large enough memory.
      tensor.Resize(large_number, 100);
      EXPECT_EQ(tensor.dim(), 2);
      EXPECT_EQ(tensor.size(0), large_number);
      EXPECT_EQ(tensor.size(1), 100);
      EXPECT_EQ(tensor.numel(), large_number * 100);
      */
}

#[test] fn tensor_test_undefined_tensor() {
    todo!();
    /*
      Tensor x;
      EXPECT_FALSE(x.defined());
      */
}

#[test] fn tensor_test_copy_and_assignment() {
    todo!();
    /*
      Tensor x(CPU);
      x.Resize(16, 17);
      testing::randomFill(x.template mutable_data<float>(), 16 * 17);
      EXPECT_TRUE(x.defined());

      Tensor y(x);
      Tensor z = x;
      testing::assertTensorEquals(x, y);
      testing::assertTensorEquals(x, z);
      */
}

#[test] fn tensor_death_test_cannot_cast_down_large_dims() {
    todo!();
    /*
      int64_t large_number =
          static_cast<int64_t>(int::max) + 1;
      Tensor tensor(vector<int64_t>{large_number}, CPU);
      EXPECT_EQ(tensor.dim(), 1);
      EXPECT_EQ(tensor.size(0), large_number);
      ASSERT_THROW(tensor.dim32(0), EnforceNotMet);
      */
}

#[macro_export] macro_rules! test_serialization_with_type {
    ($TypeParam:ty, $field_name:tt) => {
        /*
        TEST(TensorTest, TensorSerialization_##TypeParam) {                     
            Blob blob;                                                            
            Tensor* tensor = BlobGetMutableTensor(&blob, CPU);                    
            tensor->Resize(2, 3);                                                 
            for (int i = 0; i < 6; ++i) {                                         
                tensor->mutable_data<TypeParam>()[i] = static_cast<TypeParam>(i);   
            }                                                                     
            string serialized = SerializeBlob(blob, "test");                      
            BlobProto proto;                                                      
            CHECK(proto.ParseFromString(serialized));                             
            EXPECT_EQ(proto.name(), "test");                                      
            EXPECT_EQ(proto.type(), "Tensor");                                    
            EXPECT_TRUE(proto.has_tensor());                                      
            const TensorProto& tensor_proto = proto.tensor();                     
            EXPECT_EQ(                                                            
                tensor_proto.data_type(),                                         
                TypeMetaToDataType(TypeMeta::Make<TypeParam>()));                 
            EXPECT_EQ(tensor_proto.field_name##_size(), 6);                       
            for (int i = 0; i < 6; ++i) {                                         
                EXPECT_EQ(tensor_proto.field_name(i), static_cast<TypeParam>(i));   
            }                                                                     
            Blob new_blob;                                                        
            EXPECT_NO_THROW(DeserializeBlob(serialized, &new_blob));              
            EXPECT_TRUE(BlobIsTensorType(new_blob, CPU));                         
            const TensorCPU& new_tensor = blob.Get<TensorCPU>();                  
            EXPECT_EQ(new_tensor.dim(), 2);                                       
            EXPECT_EQ(new_tensor.size(0), 2);                                     
            EXPECT_EQ(new_tensor.size(1), 3);                                     
            for (int i = 0; i < 6; ++i) {                                         
                EXPECT_EQ(                                                          
                    tensor->data<TypeParam>()[i], new_tensor.data<TypeParam>()[i]); 
            }                                                                     
        }                                                                       

        TEST(EmptyTensorTest, TensorSerialization_##TypeParam) {                
            Blob blob;                                                            
            TensorCPU* tensor = BlobGetMutableTensor(&blob, CPU);                 
            tensor->Resize(0, 3);                                                 
            tensor->mutable_data<TypeParam>();                                    
            string serialized = SerializeBlob(blob, "test");                      
            BlobProto proto;                                                      
            CHECK(proto.ParseFromString(serialized));                             
            EXPECT_EQ(proto.name(), "test");                                      
            EXPECT_EQ(proto.type(), "Tensor");                                    
            EXPECT_TRUE(proto.has_tensor());                                      
            const TensorProto& tensor_proto = proto.tensor();                     
            EXPECT_EQ(                                                            
                tensor_proto.data_type(),                                         
                TypeMetaToDataType(TypeMeta::Make<TypeParam>()));                 
            EXPECT_EQ(tensor_proto.field_name##_size(), 0);                       
            Blob new_blob;                                                        
            EXPECT_NO_THROW(DeserializeBlob(serialized, &new_blob));              
            EXPECT_TRUE(BlobIsTensorType(new_blob, CPU));                         
            const TensorCPU& new_tensor = blob.Get<TensorCPU>();                  
            EXPECT_EQ(new_tensor.dim(), 2);                                       
            EXPECT_EQ(new_tensor.size(0), 0);                                     
            EXPECT_EQ(new_tensor.size(1), 3);                                     
        }
        */
    }
}

test_serialization_with_type!{bool,   int32_data}
test_serialization_with_type!{f64,    double_data}
test_serialization_with_type!{f32,    float_data}
test_serialization_with_type!{i32,    int32_data}
test_serialization_with_type!{i8,     int32_data}
test_serialization_with_type!{i16,    int32_data}
test_serialization_with_type!{u8,     int32_data}
test_serialization_with_type!{u16,    int32_data}
test_serialization_with_type!{i64,    int64_data}

#[test] fn tensor_test_tensor_serialization_custom_type() {
    todo!();
    /*
      Blob blob;
      TensorCPU* tensor = BlobGetMutableTensor(&blob, CPU);
      tensor->Resize(2, 3);
      for (int i = 0; i < 6; ++i) {
        tensor->mutable_data<BlobTestFoo>()[i].val = i;
      }
      string serialized = SerializeBlob(blob, "test");
      BlobProto proto;
      CHECK(proto.ParseFromString(serialized));
      EXPECT_EQ(proto.name(), "test");
      EXPECT_EQ(proto.type(), "Tensor");
      Blob new_blob;
      EXPECT_NO_THROW(DeserializeBlob(serialized, &new_blob));
      EXPECT_TRUE(BlobIsTensorType(new_blob, CPU));
      const TensorCPU& new_tensor = blob.Get<TensorCPU>();
      EXPECT_EQ(new_tensor.dim(), 2);
      EXPECT_EQ(new_tensor.size(0), 2);
      EXPECT_EQ(new_tensor.size(1), 3);
      for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(
            new_tensor.data<BlobTestFoo>()[i].val,
            tensor->data<BlobTestFoo>()[i].val);
      }
      */
}

#[test] fn tensor_test_half() {
    todo!();
    /*
      const int64_t kSize = 3000000;
      Blob blob;
      TensorCPU* tensor = BlobGetMutableTensor(&blob, CPU);
      tensor->Resize(kSize);
      for (int i = 0; i < tensor->numel(); ++i) {
        tensor->mutable_data<at::Half>()[i].x = i % 10000;
      }
      string serialized = SerializeBlob(blob, "test");
      BlobProto proto;
      CHECK(proto.ParseFromString(serialized));
      EXPECT_EQ(proto.name(), "test");
      EXPECT_EQ(proto.type(), "Tensor");
      EXPECT_TRUE(proto.has_tensor());
      const TensorProto& tensor_proto = proto.tensor();
      EXPECT_EQ(
          tensor_proto.data_type(), TypeMetaToDataType(TypeMeta::Make<at::Half>()));
      if (FLAGS_caffe2_serialize_fp16_as_bytes) {
        EXPECT_EQ(tensor_proto.byte_data().size(), 2 * kSize);
        for (int i = 0; i < kSize; ++i) {
          auto value = tensor->mutable_data<at::Half>()[i].x;
          auto low_bits = static_cast<char>(value & 0xff);
          auto high_bits = static_cast<char>(value >> 8);
          EXPECT_EQ(tensor_proto.byte_data()[2 * i], low_bits);
          EXPECT_EQ(tensor_proto.byte_data()[2 * i + 1], high_bits);
        }
      } else {
        EXPECT_EQ(tensor_proto.int32_data().size(), kSize);
      }
      Blob new_blob;
      EXPECT_NO_THROW(DeserializeBlob(serialized, &new_blob));
      EXPECT_TRUE(BlobIsTensorType(new_blob, CPU));
      const TensorCPU& new_tensor = blob.Get<TensorCPU>();
      EXPECT_EQ(new_tensor.dim(), 1);
      EXPECT_EQ(new_tensor.size(0), kSize);
      for (int i = 0; i < kSize; ++i) {
        EXPECT_EQ(new_tensor.data<at::Half>()[i].x, i % 10000);
      }
      */
}

#[test] fn tensor_test_tensor_factory() {
    todo!();
    /*
      Tensor a = empty({1, 2, 3}, at::device(CPU).dtype<float>());
      EXPECT_NE(a.data<float>(), nullptr);
      a.mutable_data<float>()[0] = 3.0;
      Tensor b = empty({1, 2, 3}, at::device(CPU).dtype<int>());
      EXPECT_NE(b.data<int>(), nullptr);
      b.mutable_data<int>()[0] = 3;
      */
}

#[test] fn q_tensor_test_q_tensor_serialization() {
    todo!();
    /*
      Blob blob;
      QTensor<CPUContext>* qtensor = blob.GetMutable<QTensor<CPUContext>>();
      qtensor->SetPrecision(5);
      qtensor->SetSigned(false);
      qtensor->SetScale(1.337);
      qtensor->SetBias(-1.337);
      qtensor->Resize(std::vector<int>{2, 3});
      // "Randomly" set bits.
      srand(0);
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 5; ++j) {
          qtensor->SetBitAtIndex(j, i, rand() % 2);
        }
      }

      string serialized = SerializeBlob(blob, "test");
      BlobProto proto;
      CHECK(proto.ParseFromString(serialized));
      EXPECT_EQ(proto.name(), "test");
      EXPECT_EQ(proto.type(), "QTensor");
      EXPECT_TRUE(proto.has_qtensor());
      const QTensorProto& qtensor_proto = proto.qtensor();

      EXPECT_EQ(qtensor_proto.precision(), qtensor->precision());
      EXPECT_EQ(qtensor_proto.scale(), qtensor->scale());
      EXPECT_EQ(qtensor_proto.bias(), qtensor->bias());
      EXPECT_EQ(qtensor_proto.is_signed(), qtensor->is_signed());

      Blob new_blob;
      DeserializeBlob(serialized, &new_blob);
      EXPECT_TRUE(new_blob.IsType<QTensor<CPUContext>>());
      const QTensor<CPUContext>& new_qtensor = blob.Get<QTensor<CPUContext>>();
      EXPECT_EQ(new_qtensor.ndim(), 2);
      EXPECT_EQ(new_qtensor.dim32(0), 2);
      EXPECT_EQ(new_qtensor.dim32(1), 3);
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 5; ++j) {
          EXPECT_EQ(qtensor->GetBitAtIndex(j, i), new_qtensor.GetBitAtIndex(j, i));
        }
      }
      */
}

type StringMap = Vec<(String,String)>;

pub struct VectorCursor {
    data: *mut StringMap, // default = nullptr
    pos:  usize, // default = 0
}

impl VectorCursor {

    pub fn new(data: *mut StringMap) -> Self {
        todo!();
        /*
            : data_(data) 
        pos_ = 0;
        */
    }
}

impl Cursor for VectorCursor {
    
    #[inline] fn seek(&mut self, unused: &String)  {
        
        todo!();
        /*
        
        */
    }
    
    #[inline] fn seek_to_first(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    #[inline] fn next(&mut self)  {
        
        todo!();
        /*
            ++pos_;
        */
    }
    
    #[inline] fn key(&mut self) -> String {
        
        todo!();
        /*
            return (*data_)[pos_].first;
        */
    }
    
    #[inline] fn value(&mut self) -> String {
        
        todo!();
        /*
            return (*data_)[pos_].second;
        */
    }
    
    #[inline] fn valid(&mut self) -> bool {
        
        todo!();
        /*
            return pos_ < data_->size();
        */
    }
}

pub struct VectorDB {
    base: Database,
    name: String,

    /*TODO uncomment
      static std::mutex dataRegistryMutex_;
      static std::map<string, StringMap> data_;
      */
}

impl Drop for VectorDB {
    fn drop(&mut self) {
        todo!();
        /* 
        data_.erase(name_);
       */
    }
}

impl VectorDB {
    
    pub fn new(source: &String, mode: DatabaseMode) -> Self {
        todo!();
        /*
            : DB(source, mode), name_(source)
        */
    }
    
    #[inline] pub fn close(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn new_cursor(&mut self) -> Box<dyn Cursor> {
        
        todo!();
        /*
            return make_unique<VectorCursor>(getData());
        */
    }
    
    #[inline] pub fn new_transaction(&mut self) -> Box<dyn Transaction> {
        
        todo!();
        /*
            CAFFE_THROW("Not implemented");
        */
    }
    
    #[inline] pub fn register_data(name: &String, data: StringMap)  {
        
        todo!();
        /*
            std::lock_guard<std::mutex> guard(dataRegistryMutex_);
        data_[name] = std::move(data);
        */
    }
    
    #[inline] pub fn get_data(&mut self) -> *mut StringMap {
        
        todo!();
        /*
            auto it = data_.find(name_);
        CAFFE_ENFORCE(it != data_.end(), "Can't find ", name_);
        return &(it->second);
        */
    }
}

register_caffe2_db![vector_db, VectorDB];

type TypedTensorTest<TypeParam> = Test<TypeParam>;

type TensorDataTypes = (f32, bool, f64, i32, i8, i16, u8, u16, i64);

typed_test_case![TypedTensorTest, TensorDataTypes];

#[test] fn typed_tensor_test_big_tensor_serialization() {
    todo!();
    /*
      int64_t d1 = 2;
      int64_t d2 = FLAGS_caffe2_test_big_tensor_size
          ? FLAGS_caffe2_test_big_tensor_size / d1
          : static_cast<int64_t>(int::max) + 1;
      int64_t size = d1 * d2;
      string db_source = (string)std::tmpnam(nullptr);
      VLOG(1) << "db_source: " << db_source;

      {
        VLOG(1) << "Test begin";
        Blob blob;
        Tensor* tensor = BlobGetMutableTensor(&blob, CPU);
        VLOG(1) << "Allocating blob";
        tensor->Resize(d1, d2);
        auto mutableData = tensor->mutable_data<TypeParam>();
        VLOG(1) << "Filling out the blob";
        for (int64_t i = 0; i < size; ++i) {
          mutableData[i] = static_cast<TypeParam>(i);
        }
        StringMap data;
        std::mutex mutex;
        /*auto db = CreateDB("minidb", db_source, WRITE);*/
        auto acceptor = [&](const std::string& key, const std::string& value) {
          std::lock_guard<std::mutex> guard(mutex);
          /*db->NewTransaction()->Put(key, value);*/
          data.emplace_back(key, value);
        };
        SerializeBlob(blob, "test", acceptor);
        VectorDB::registerData(db_source, std::move(data));
        VLOG(1) << "finished writing to DB";
      }

      {
        DeviceOption option;
        option.set_device_type(PROTO_CPU);
        Argument db_type_arg = MakeArgument<string>("db_type", "vector_db");
        Argument absolute_path_arg = MakeArgument<bool>("absolute_path", true);
        Argument db_source_arg = MakeArgument<string>("db", db_source);
        auto op_def = CreateOperatorDef(
            "Load",
            "",
            std::vector<string>{},
            std::vector<string>({"test"}),
            std::vector<Argument>{db_type_arg, db_source_arg, absolute_path_arg},
            option,
            "DUMMY_ENGINE");
        Workspace ws;
        auto load_op = CreateOperator(op_def, &ws);
        EXPECT_TRUE(load_op != nullptr);
        VLOG(1) << "Running operator";

        load_op->Run();
        VLOG(1) << "Reading blob from workspace";
        auto new_blob = ws.GetBlob("test");
        EXPECT_TRUE(BlobIsTensorType(*new_blob, CPU));
        const auto& new_tensor = new_blob->Get<TensorCPU>();

        EXPECT_EQ(new_tensor.dim(), d1);
        EXPECT_EQ(new_tensor.size(0), d1);
        EXPECT_EQ(new_tensor.size(1), d2);
        for (int64_t i = 0; i < size; ++i) {
          EXPECT_EQ(static_cast<TypeParam>(i), new_tensor.data<TypeParam>()[i]);
        }
      }
      */
}

/**
  | This struct is used to test serialization
  | and deserialization of huge blobs,
  | that are not tensors.
  |
  */
pub struct DummyType {
    n_chunks: i32,
}

impl DummyType {
    
    pub fn new(n_chunks_init: i32) -> Self {
        todo!();
        /*
            : n_chunks(n_chunks_init)
        */
    }
    
    #[inline] pub fn serialize(&self, name: &String, chunk_id: i32) -> String {
        
        todo!();
        /*
            BlobProto blobProto;
        blobProto.set_name(name);
        blobProto.set_type("DummyType");
        std::string content("");
        blobProto.set_content(content);
        blobProto.set_content_num_chunks(n_chunks);
        blobProto.set_content_chunk_id(chunk_id);
        return blobProto.SerializeAsString();
        */
    }
    
    #[inline] pub fn deserialize(&mut self, unused: &BlobProto)  {
        
        todo!();
        /*
            ++n_chunks;
        */
    }
}

///------------------------------------------
pub struct DummyTypeSerializer {
    base: BlobSerializationOptions,
}

impl BlobSerializerBase for DummyTypeSerializer {
    
    #[inline] fn serialize(
        &mut self, 
        pointer:    *const c_void,
        type_meta:  TypeMeta,
        name:       &String,
        acceptor:   SerializationAcceptor,
        options:   Option<&BlobSerializationOptions>)  
    {
        todo!();
        /*
            CAFFE_ENFORCE(typeMeta.Match<DummyType>());
        const auto& container = *static_cast<const DummyType*>(pointer);
        for (int k = 0; k < container.n_chunks; ++k) {
          std::string serialized_chunk = container.serialize(name, k);
          acceptor(c10::str(name, kChunkIdSeparator, k), serialized_chunk);
        }
        */
    }
}

///------------------------------------------
pub struct DummyTypeDeserializer {
    base: dyn BlobDeserializerBase,
}

impl DummyTypeDeserializer {
    
    #[inline] pub fn deserialize(&mut self, proto: &BlobProto, blob: *mut Blob)  {
        
        todo!();
        /*
            auto* container = blob->GetMutable<DummyType>();
        container->deserialize(proto);
        */
    }
}

///------------------------------------------
caffe_known_type![DummyType];

register_blob_serializer![
    /*
    (TypeMeta::Id<DummyType>()), 
    DummyTypeSerializer
    */
];

register_typed_class!{
    BlobDeserializerRegistry, 
    "DummyType", 
    DummyTypeDeserializer
}

#[test] fn content_chunks_serialization() {
    todo!();
    /*

  string db_source = (string)std::tmpnam(nullptr);
  VLOG(1) << "db_source: " << db_source;

  {
    VLOG(1) << "Test begin";
    Blob blob;
    DummyType* container = blob.GetMutable<DummyType>();
    VLOG(1) << "Allocating blob";
    container->n_chunks = 10;
    VLOG(1) << "Filling out the blob";
    StringMap data;
    std::mutex mutex;
    auto acceptor = [&](const std::string& key, const std::string& value) {
      std::lock_guard<std::mutex> guard(mutex);
      data.emplace_back(key, value);
    };
    SerializeBlob(blob, "test", acceptor);
    VectorDB::registerData(db_source, std::move(data));
    VLOG(1) << "finished writing to DB";
  }

  {
    DeviceOption option;
    option.set_device_type(PROTO_CPU);
    Argument db_type_arg = MakeArgument<string>("db_type", "vector_db");
    Argument absolute_path_arg = MakeArgument<bool>("absolute_path", true);
    Argument db_source_arg = MakeArgument<string>("db", db_source);
    auto op_def = CreateOperatorDef(
        "Load",
        "",
        std::vector<string>{},
        std::vector<string>({"test"}),
        std::vector<Argument>{db_type_arg, db_source_arg, absolute_path_arg},
        option,
        "DUMMY_ENGINE");
    Workspace ws;
    auto load_op = CreateOperator(op_def, &ws);
    EXPECT_TRUE(load_op != nullptr);
    VLOG(1) << "Running operator";

    load_op->Run();
    VLOG(1) << "Reading blob from workspace";
    auto new_blob = ws.GetBlob("test");
    EXPECT_TRUE(new_blob->IsType<DummyType>());
    const auto& container = new_blob->Get<DummyType>();
    EXPECT_EQ(container.n_chunks, 10);
  }
  */
}

#[test] fn custom_chunk_size_big_tensor_serialization() {
    todo!();
    /*

  int64_t d1 = 2;
  int64_t d2 = FLAGS_caffe2_test_big_tensor_size
      ? FLAGS_caffe2_test_big_tensor_size / d1
      : static_cast<int64_t>(int::max) + 1;
  BlobSerializationOptions options;

  Blob blob;
  TensorCPU* tensor = BlobGetMutableTensor(&blob, CPU);
  tensor->Resize(d1, d2);
  tensor->mutable_data<float>();
  std::mutex mutex;
  int counter = 0;
  auto acceptor = [&](const std::string& /*key*/,
                      const std::string& /*value*/) {
    std::lock_guard<std::mutex> guard(mutex);
    counter++;
  };
  options.set_chunk_size(d1 * d2);
  SerializeBlob(blob, "test", acceptor, options);
  EXPECT_EQ(counter, 1);

  counter = 0;
  options.set_chunk_size((d1 * d2) / 2 + 1);
  SerializeBlob(blob, "test", acceptor, options);
  EXPECT_EQ(counter, 2);

  counter = 0;
  options.set_chunk_size(-1);
  SerializeBlob(blob, "test", acceptor, options);
  EXPECT_EQ(counter, 1);
  */
}

#[test] fn q_tensor_q_tensor_sizing_test() {
    todo!();
    /*
      vector<int> dims(3);
      dims[0] = 2;
      dims[1] = 3;
      dims[2] = 5;
      QTensor<CPUContext> qtensor(dims, 3);
      EXPECT_TRUE(qtensor.mutable_data() != nullptr);
      EXPECT_EQ(qtensor.nbytes(), 12);
      EXPECT_EQ(qtensor.size(), 30);
      */
}

#[test] fn blob_test_casting_message() {
    todo!();
    /*
      Blob b;
      b.GetMutable<BlobTestFoo>();
      b.Get<BlobTestFoo>();
      try {
        b.Get<BlobTestBar>();
        FAIL() << "Should have thrown";
      } catch (const EnforceNotMet& e) {
        string msg = e.what_without_backtrace();
        LOG(INFO) << msg;
        EXPECT_NE(msg.find("BlobTestFoo"), std::string::npos) << msg;
        EXPECT_NE(msg.find("BlobTestBar"), std::string::npos) << msg;
      }
      */
}

#[test] fn tensor_construction_uninitialized_copy_test() {
    todo!();
    /*
      Tensor x(CPU);
      Tensor y(x, CPU);
      Tensor z = x.Clone();
      EXPECT_FALSE(x.dtype_initialized());
      EXPECT_FALSE(y.dtype_initialized());
      LOG(INFO) << "z.size()" << z.numel();
      EXPECT_FALSE(z.dtype_initialized());
      */
}

#[test] fn tensor_construction_copy_constructor_test() {
    todo!();
    /*
      Tensor x(CPU);
      x.Resize(5);
      x.mutable_data<float>()[0] = 1;
      Tensor y = x.Clone();
      Tensor z(x, CPU);

      EXPECT_EQ(*x.data<float>(), 1);
      EXPECT_EQ(*y.data<float>(), 1);
      EXPECT_EQ(*z.data<float>(), 1);
      x.mutable_data<float>()[0] = 5;
      EXPECT_EQ(*x.data<float>(), 5);
      EXPECT_EQ(*y.data<float>(), 1);
      EXPECT_EQ(*z.data<float>(), 1);
      */
}

#[test] fn tensor_construction_move_assignment_op_test() {
    todo!();
    /*
      Tensor x(CPU);
      x.Resize(5);
      x.mutable_data<float>()[0] = 1;
      Tensor y(CPU);
      y = std::move(x);

      EXPECT_EQ(*y.data<float>(), 1);
      */
}

#[test] fn tensor_serialization_mistakenly_serializing_dtype_uninitialized_tensor() {
    todo!();
    /*
      // This test preserves a legacy behavior that dtype-unitialized tensors can
      // go through serialization. We want to kill this behavior - when it's done,
      // remove this test
      Blob blob;
      Tensor* x = BlobGetMutableTensor(&blob, CPU);
      x->Resize(0);
      string output;
      SerializeBlob(
          blob,
          "foo",
          [&output](const string& /*blobName*/, const std::string& data) {
            output = data;
          });
      BlobProto b;
      CHECK(b.ParseFromString(output));
      LOG(INFO) << "serialized proto: " << b.DebugString();

      Blob new_blob;
      // Deserializing an empty Tensor gives a {0}-dim, float CPU Tensor
      DeserializeBlob(output, &new_blob);
      const Tensor& new_tensor = new_blob.Get<Tensor>();
      LOG(INFO) << "tensor " << new_tensor.DebugString();
      EXPECT_TRUE(new_tensor.dtype_initialized());
      LOG(INFO) << "dtype:" << new_tensor.dtype();
      EXPECT_EQ(0, new_tensor.numel());
      EXPECT_EQ(1, new_tensor.dim());
      */
}

#[inline] pub fn create_proto_with_int_32data(
    data_type:  &TensorProto_DataType,
    num_el:     usize,
    use_cached: Option<bool>) -> BlobProto 
{
    let use_cached: bool = use_cached.unwrap_or(true);

    todo!();
    /*
        static std::map<caffe2::TensorProto::DataType, caffe2::BlobProto> protos;
      if (useCached && protos.count(dataType)) {
        return protos[dataType];
      }
      caffe2::BlobProto proto;
      proto.set_type("Tensor");
      auto tensor = proto.mutable_tensor();
      tensor->add_dims(numEl);
      tensor->add_dims(1);
      tensor->set_data_type(dataType);
      tensor->set_name("test_feature");
      tensor->mutable_device_detail()->set_device_type(0);
      tensor->mutable_segment()->set_begin(0);
      tensor->mutable_segment()->set_end(numEl);
      for (size_t i = 0; i < numEl; ++i) {
        int32_t data = 0;
        switch (dataType) {
          case caffe2::TensorProto_DataType_INT32:
            data = static_cast<int32_t>(rand() % 0xffffffff);
            break;
          case caffe2::TensorProto_DataType_BOOL:
            data = static_cast<uint8_t>(rand() % 0x00000001);
            break;
          case caffe2::TensorProto_DataType_UINT8:
            data = static_cast<uint8_t>(rand() % 0x000000ff);
            break;
          case caffe2::TensorProto_DataType_INT8:
            data = static_cast<int8_t>(rand() % 0x000000ff);
            break;
          case caffe2::TensorProto_DataType_UINT16:
            data = static_cast<uint16_t>(rand() % 0x0000ffff);
            break;
          case caffe2::TensorProto_DataType_INT16:
            data = static_cast<int16_t>(rand() % 0x0000ffff);
            break;
          case caffe2::TensorProto_DataType_FLOAT16:
            data = static_cast<uint16_t>(rand() % 0x0000ffff);
            break;
          default:
            continue;
        }
        tensor->add_int32_data(data);
      }
      protos[dataType] = proto;
      return proto;
    */
}

#[inline] pub fn test_data_type(
    data_type: &TensorProto_DataType,
    data_type_name: String)  
{
    todo!();
    /*
        LOG(INFO) << dataTypeName;
      FLAGS_caffe2_serialize_using_bytes_as_holder = true;
      size_t numEl = 1000;
      // Proto with int32
      auto protoInt32 = CreateProtoWithInt32Data(dataType, numEl, false);
      caffe2::Blob blobInt32;
      DeserializeBlob(protoInt32, &blobInt32);
      auto serializedStr = SerializeBlob(blobInt32, protoInt32.name());
      caffe2::BlobProto protoBytes;
      // Proto with bytes
      protoBytes.ParseFromString(serializedStr);
      caffe2::Blob blobBytes;
      DeserializeBlob(protoBytes, &blobBytes);
      FLAGS_caffe2_serialize_using_bytes_as_holder = false;
      // Proto with int32 from proto with bytes
      protoBytes.ParseFromString(SerializeBlob(blobBytes, protoBytes.name()));
      EXPECT_EQ(numEl, protoInt32.tensor().int32_data_size());
      EXPECT_EQ(numEl, protoBytes.tensor().int32_data_size());
      for (int i = 0; i < numEl; ++i) {
        EXPECT_EQ(
            protoInt32.tensor().int32_data(i), protoBytes.tensor().int32_data(i));
      }
    */
}

#[test] fn tensor_serialization_test_correctness() {
    todo!();
    /*
      FLAGS_caffe2_serialize_using_bytes_as_holder = true;
      TestDataType(
          caffe2::TensorProto_DataType_INT32, "TensorProto_DataType_INT32");
      TestDataType(caffe2::TensorProto_DataType_BOOL, "TensorProto_DataType_BOOL");
      TestDataType(
          caffe2::TensorProto_DataType_UINT8, "TensorProto_DataType_UINT8");
      TestDataType(caffe2::TensorProto_DataType_INT8, "TensorProto_DataType_INT8");
      TestDataType(
          caffe2::TensorProto_DataType_UINT16, "TensorProto_DataType_UINT16");
      TestDataType(
          caffe2::TensorProto_DataType_INT16, "TensorProto_DataType_INT16");
      TestDataType(
          caffe2::TensorProto_DataType_FLOAT16, "TensorProto_DataType_FLOAT16");
      */
}
