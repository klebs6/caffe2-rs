crate::ix!();

/**
  | Storage orders that are often used in
  | the image applications.
  |
  */
#[derive(PartialEq,Eq)]
pub enum StorageOrder {
    UNKNOWN = 0,
    NHWC    = 1,
    NCHW    = 2,
}

#[inline] pub fn string_to_storage_order(str: &String) -> StorageOrder {
    
    todo!();
    /*
        if (str == "NHWC" || str == "nhwc") {
        return StorageOrder::NHWC;
      } else if (str == "NCHW" || str == "nchw") {
        return StorageOrder::NCHW;
      } else {
        LOG(ERROR) << "Unknown storage order string: " << str;
        return StorageOrder::UNKNOWN;
      }
    */
}

#[inline] pub fn get_dim_from_order_string(str: &String) -> i32 {
    
    todo!();
    /*
        auto order = StringToStorageOrder(str);
      switch (order) {
        case StorageOrder::NHWC:
          return 3;
        case StorageOrder::NCHW:
          return 1;
        default:
          CAFFE_THROW("Unsupported storage order: ", str);
          return -1;
      }
    */
}

#[inline] pub fn name_scope_separator() -> u8 {
    
    todo!();
    /*
        return '/';
    */
}

/**
  | at::Half is defined in
  | c10/util/Half.h. Currently half float operators
  | are mainly on CUDA gpus.
  |
  | The reason we do not directly use the cuda
  | __half data type is because that requires
  | compilation with nvcc. The float16 data type
  | should be compatible with the cuda __half data
  | type, but will allow us to refer to the data
  | type without the need of cuda.
  */

// Helpers to avoid using typeinfo with -rtti
pub trait Fp16Type {
    fn fp16_type() -> bool;
}

default impl<T> Fp16Type for T {
    fn fp16_type() -> bool {
        false
    }
}

impl Fp16Type for f16 {
    fn fp16_type() -> bool {
        true
    }
}

/**
  | From TypeMeta to caffe2::DataType
  | protobuffer enum.
  |
  */
#[inline] pub fn type_meta_to_data_type(meta: TypeMeta) 
-> TensorProto_DataType {
    
    todo!();
    /*
        static_assert(
          sizeof(int) == 4, "int in this compiler does not equal to 4 bytes.");
      static std::map<TypeIdentifier, TensorProto::DataType> data_type_map{
          {TypeMeta::Id<float>(), TensorProto_DataType_FLOAT},
          {TypeMeta::Id<int>(), TensorProto_DataType_INT32},
          // BYTE does not have a type meta to proto mapping: we should
          // always use uint8_t when serializing. BYTE is kept for backward
          // compatibility.
          // {TypeMeta::Id<>(), TensorProto_DataType_BYTE},
          {TypeMeta::Id<string>(), TensorProto_DataType_STRING},
          {TypeMeta::Id<bool>(), TensorProto_DataType_BOOL},
          {TypeMeta::Id<uint8_t>(), TensorProto_DataType_UINT8},
          {TypeMeta::Id<int8_t>(), TensorProto_DataType_INT8},
          {TypeMeta::Id<uint16_t>(), TensorProto_DataType_UINT16},
          {TypeMeta::Id<int16_t>(), TensorProto_DataType_INT16},
          {TypeMeta::Id<int64_t>(), TensorProto_DataType_INT64},
          {TypeMeta::Id<at::Half>(), TensorProto_DataType_FLOAT16},
          {TypeMeta::Id<double>(), TensorProto_DataType_DOUBLE},
          {TypeMeta::Id<c10::qint8>(), TensorProto_DataType_INT8},
          {TypeMeta::Id<c10::quint8>(), TensorProto_DataType_UINT8},
          {TypeMeta::Id<c10::qint32>(), TensorProto_DataType_INT32},
      };
      const auto it = data_type_map.find(meta.id());
      return (
          it == data_type_map.end() ? TensorProto_DataType_UNDEFINED : it->second);
    */
}

/**
  | From caffe2::DataType protobuffer
  | enum to TypeMeta
  |
  */
#[inline] pub fn data_type_to_type_meta(dt: &TensorProto_DataType) -> TypeMeta {
    
    todo!();
    /*
        static std::map<TensorProto::DataType, TypeMeta> type_meta_map{
          {TensorProto_DataType_FLOAT, TypeMeta::Make<float>()},
          {TensorProto_DataType_INT32, TypeMeta::Make<int>()},
          {TensorProto_DataType_BYTE, TypeMeta::Make<uint8_t>()},
          {TensorProto_DataType_STRING, TypeMeta::Make<std::string>()},
          {TensorProto_DataType_BOOL, TypeMeta::Make<bool>()},
          {TensorProto_DataType_UINT8, TypeMeta::Make<uint8_t>()},
          {TensorProto_DataType_INT8, TypeMeta::Make<int8_t>()},
          {TensorProto_DataType_UINT16, TypeMeta::Make<uint16_t>()},
          {TensorProto_DataType_INT16, TypeMeta::Make<int16_t>()},
          {TensorProto_DataType_INT64, TypeMeta::Make<int64_t>()},
          {TensorProto_DataType_FLOAT16, TypeMeta::Make<at::Half>()},
          {TensorProto_DataType_DOUBLE, TypeMeta::Make<double>()},
      };
      const auto it = type_meta_map.find(dt);
      if (it == type_meta_map.end()) {
        throw std::runtime_error("Unknown data type.");
      }
      return it->second;
    */
}
