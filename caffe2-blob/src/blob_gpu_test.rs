crate::ix!();

type TensorTypes = (u8, i32, f32);

#[cfg(test)]
pub mod tensor_gpu_test {

    use super::*;

    #[inline] fn test_data_points_to_something<T>(tensor: &Tensor) {
        assert!(tensor.mutable_data::<T>() != std::ptr::null_mut());
        assert!(tensor.data::<T>()         != std::ptr::null_mut());
    }

    #[inline] fn test_data_points_to_something_for_all_supported_types(tensor: &Tensor) {
        test_data_points_to_something::<u8>(&tensor);
        test_data_points_to_something::<i32>(&tensor);
        test_data_points_to_something::<f32>(&tensor);
    }

    #[test] 
    fn tensor_initialized_empty() {

        if !has_cuda_gpu() {
            return;
        }

        let tensor: Tensor = Tensor::from(DeviceType::Cuda);

        assert_eq!(tensor.numel(), 0);
        assert_eq!(tensor.dim(),   1);

        let dims: Vec::<i32> = vec![2,3,5];

        tensor.resize(dims);

        assert_eq!(tensor.dim(),    3);
        assert_eq!(tensor.dim32(0), 2);
        assert_eq!(tensor.dim32(1), 3);
        assert_eq!(tensor.dim32(2), 5);

        test_data_points_to_something_for_all_supported_types(&tensor);
    }

    #[test] 
    fn tensor_initialized_nonempty() {

        if !has_cuda_gpu() {
            return;
        }

        let mut dims: Vec::<i32> = vec![2,3,5];

        let tensor: Tensor 
        = Tensor::new_with_dimension_and_type(
            &dims, 
            DeviceType::Cuda
        );

        assert_eq!(tensor.dim(), 3);
        assert_eq!(tensor.dim32(0), 2);
        assert_eq!(tensor.dim32(1), 3);
        assert_eq!(tensor.dim32(2), 5);

        test_data_points_to_something_for_all_supported_types(&tensor);

        dims[0] = 7;
        dims[1] = 11;
        dims[2] = 13;

        dims.push(17);

        tensor.resize(dims);

        assert_eq!(tensor.dim(), 4);
        assert_eq!(tensor.dim32(0), 7);
        assert_eq!(tensor.dim32(1), 11);
        assert_eq!(tensor.dim32(2), 13);
        assert_eq!(tensor.dim32(3), 17);

        test_data_points_to_something_for_all_supported_types(&tensor);
    }

    #[test] 
    fn tensor_alias() {

        if !has_cuda_gpu() {
            return;
        }

        let mut dims: Vec::<i32> = Vec::<i32>::with_capacity(3);

        dims[0] = 2;
        dims[1] = 3;
        dims[2] = 5;

        let tensor: Tensor = Tensor::new_with_dimension_and_type(&*dims, DeviceType::Cuda);

        macro_rules! tensor_alias_test {
            ($($TypeParam:ty),*) => {
                $(
                    assert!(
                        tensor.mutable_data::<$TypeParam>()  
                        != std::ptr::null_mut()
                    );

                    let other_tensor: Tensor = tensor.alias();

                    assert!(
                        tensor.data::<$TypeParam>() 
                        != std::ptr::null_mut()
                    );

                    assert!(
                        other_tensor.data::<$TypeParam>() 
                        != std::ptr::null_mut()
                    );

                    assert_eq!(
                        tensor.data::<$TypeParam>(),
                        other_tensor.data::<$TypeParam>()
                    );
                )*
            }
        }

        tensor_alias_test![u8,i32,f32];
    }

    #[test] 
    fn tensor_alias_can_use_different_shapes() {

        if !has_cuda_gpu() {
            return;
        }

        let mut dims: Vec::<i32> = Vec::<i32>::with_capacity(3);

        dims[0] = 2;
        dims[1] = 3;
        dims[2] = 5;

        let mut alternate_dims: Vec::<i32> = Vec::<i32>::with_capacity(1);

        alternate_dims[0] = 2 * 3 * 5;

        let tensor: Tensor = Tensor::new_with_dimension_and_type(&*dims, DeviceType::Cuda);

        macro_rules! tensor_alias_shape_test {
            ($($TypeParam:ty),*) => {
                $(
                    assert!(
                        tensor.mutable_data::<$TypeParam>() 
                        != std::ptr::null_mut()
                    );

                    let other_tensor: Tensor = tensor.alias();

                    other_tensor.resize(&alternate_dims);

                    assert_eq!(
                        other_tensor.dim(), 
                        1
                    );

                    assert_eq!(
                        other_tensor.dim32(0), 
                        alternate_dims[0]
                    );

                    assert!(
                        tensor.data::<$TypeParam>() 
                        != std::ptr::null_mut()
                    );

                    assert!(
                        other_tensor.data::<$TypeParam>() 
                        != std::ptr::null_mut()
                    );

                    assert_eq!(
                        tensor.data::<$TypeParam>(), 
                        other_tensor.data::<$TypeParam>()
                    );

                )*
            }
        }

        tensor_alias_shape_test![u8,i32,f32];
    }

    #[test] 
    fn no_longer_alias_after_numel_changes() {

        if !has_cuda_gpu() {
            return;
        }

        let mut dims: Vec::<i32> = vec![3];

        dims[0] = 2;
        dims[1] = 3;
        dims[2] = 5;

        let tensor: Tensor = Tensor::new_with_dimension_and_type(&dims, DeviceType::Cuda);

        macro_rules! no_longer_alias_test {
            ($($TypeParam:ty),*) => {
                $(
                    assert!(
                        tensor.mutable_data::<$TypeParam>() 
                        != std::ptr::null_mut()
                    );

                    let other_tensor: Tensor = tensor.alias();

                    assert_eq!(
                        tensor.data::<$TypeParam>(), 
                        other_tensor.data::<$TypeParam>()
                    );

                    assert!(
                        tensor.mutable_data::<$TypeParam>() 
                        != std::ptr::null_mut()
                    );

                    let other_tensor: Tensor = tensor.alias();

                    assert_eq!(
                        tensor.data::<$TypeParam>(), 
                        other_tensor.data::<$TypeParam>()
                    );

                    let old_pointer 
                    = other_tensor.data::<$TypeParam>();

                    dims[0] = 7;

                    tensor.resize(&dims);

                    assert_eq!(
                        old_pointer, 
                        other_tensor.data::<$TypeParam>()
                    );

                    assert_ne!(
                        old_pointer, 
                        tensor.mutable_data::<$TypeParam>()
                    );
                )*
            }
        }

        no_longer_alias_test![u8,i32,f32];
    }

    macro_rules! test_serialization_gpu_with_type {
        ($TypeParam:ty, $field_name:tt) => {
            paste!{
                #[test] fn [<TensorSerialization_ $TypeParam >]() {                   

                    if !has_cuda_gpu() {
                        return;
                    }

                    let mut blob: Blob;

                    let cpu_tensor: Tensor = Tensor::from(DeviceType::CPU);

                    cpu_tensor.resize((2, 3));

                    todo!();
                    /*
                    for (int i = 0; i < 6; ++i) {                                          
                        cpu_tensor.mutable_data<TypeParam>()[i] = static_cast<TypeParam>(i); 
                    }                                                                      
                    BlobGetMutableTensor(&blob, CUDA)->CopyFrom(cpu_tensor);               
                    string serialized = SerializeBlob(blob, "test");                       
                    BlobProto proto;                                                       
                    CAFFE_ENFORCE(proto.ParseFromString(serialized));                      
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
                    EXPECT_TRUE(BlobIsTensorType(new_blob, CUDA));                         
                    Tensor new_cpu_tensor(blob.Get<Tensor>(), CPU);                        
                    EXPECT_EQ(new_cpu_tensor.dim(), 2);                                    
                    EXPECT_EQ(new_cpu_tensor.size(0), 2);                                  
                    EXPECT_EQ(new_cpu_tensor.size(1), 3);                                  
                    for (int i = 0; i < 6; ++i) {                                          
                        EXPECT_EQ(                                                           
                            cpu_tensor.data<TypeParam>()[i],                                 
                            new_cpu_tensor.data<TypeParam>()[i]);                            
                    }                                                                      
                    */
                }
            }
      }
    }

    test_serialization_gpu_with_type![bool,      int32_data];
    test_serialization_gpu_with_type![double,    double_data];
    test_serialization_gpu_with_type![float,     float_data];
    test_serialization_gpu_with_type![int,       int32_data];
    test_serialization_gpu_with_type![int8_t,    int32_data];
    test_serialization_gpu_with_type![int16_t,   int32_data];
    test_serialization_gpu_with_type![uint8_t,   int32_data];
    test_serialization_gpu_with_type![uint16_t,  int32_data];
    test_serialization_gpu_with_type![int64_t,   int64_data];
}

macro_rules! expect_throw {
    ($($tt:tt)*) => {
        //nothing to see here folks!
    }
}

#[cfg(test)]
pub mod tensor_gpu_death_test {

    use super::*;

    #[test] 
    fn cannot_access_data_when_empty() {

        if !has_cuda_gpu() {
            return;
        }

        let FLAGS_gtest_death_test_style = "threadsafe";

        let tensor: Tensor = Tensor::from(DeviceType::Cuda);

        assert_eq!(tensor.dim(), 1);
        assert_eq!(tensor.numel(), 0);

        expect_throw!(tensor.data::<u8>(),  EnforceNotMet);
        expect_throw!(tensor.data::<i32>(), EnforceNotMet);
        expect_throw!(tensor.data::<f32>(), EnforceNotMet);
    }
}

#[cfg(test)]
pub mod tensor_construction {

    use super::*;

    #[test] fn reinitialize_tensor_test() {

        if !has_cuda_gpu() {
            return;
        }

        todo!();
        /*
      Tensor x = caffe2::empty({1}, at::dtype<float>().device(CUDA, 0));
      auto* data_before = x.template mutable_data<float>();
      // We'll only compare device_type in ReinitializeTensor,
      // so no tensor reallocation will happen here
      ReinitializeTensor(&x, {1}, at::dtype<float>().device(CUDA));
      auto* data_after = x.template mutable_data<float>();
      EXPECT_EQ(data_before, data_after);
      */
    }
}

#[cfg(test)]
pub mod tensor_test {

    use super::*;

    #[test] 
    fn tensor_serialization_multiple_devices() {

        todo!();

        /*
        let mut blob: Blob;
        let tensor: Tensor = Tensor::from(DeviceType::CPU);

        tensor.resize((2, 3));

        for i in 0..6 {
            *tensor.mutable_data::<f32>().add(i as usize) = i as f32;
        }
        */

        /*

      for (int gpu_id = 0; gpu_id < NumCudaDevices(); ++gpu_id) {
        CUDAGuard guard(gpu_id);
        CUDAContext context(gpu_id); // switch to the current gpu
        blob.Reset(new Tensor(tensor, CUDA));
        string serialized = SerializeBlob(blob, "test");
        BlobProto proto;
        CAFFE_ENFORCE(proto.ParseFromString(serialized));
        EXPECT_EQ(proto.name(), "test");
        EXPECT_TRUE(proto.has_tensor());
        const TensorProto& tensor_proto = proto.tensor();
        EXPECT_EQ(tensor_proto.data_type(), TensorProto::FLOAT);
        EXPECT_EQ(tensor_proto.float_data_size(), 6);
        for (int i = 0; i < 6; ++i) {
          EXPECT_EQ(tensor_proto.float_data(i), i);
        }
        EXPECT_TRUE(tensor_proto.has_device_detail());
        EXPECT_EQ(tensor_proto.device_detail().device_type(), PROTO_CUDA);
        EXPECT_EQ(tensor_proto.device_detail().device_id(), gpu_id);
        // Test if the restored blob is still of the same device.
        blob.Reset();
        EXPECT_NO_THROW(DeserializeBlob(serialized, &blob));
        EXPECT_TRUE(BlobIsTensorType(blob, CUDA));
        EXPECT_EQ(GetGPUIDForPointer(blob.Get<TensorCUDA>().data<float>()),
                  gpu_id);
        // Test if we force the restored blob on a different device, we
        // can still get so.
        blob.Reset();
        proto.mutable_tensor()->mutable_device_detail()->set_device_id(0);
        EXPECT_NO_THROW(DeserializeBlob(proto.SerializeAsString(), &blob));
        EXPECT_TRUE(BlobIsTensorType(blob, CUDA));
        EXPECT_EQ(GetGPUIDForPointer(blob.Get<TensorCUDA>().data<float>()), 0);
      }
      */
    }
}
