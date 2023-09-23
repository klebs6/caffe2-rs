crate::ix!();

#[macro_export] macro_rules! serialize_type_case {
    ($proto_type:ident, $type:ident) => {
        todo!();
        /*
        
          case TensorProto_DataType_##proto_type: {                     
            SerializeTensorData(SerializeParams<type>(                  
                GetTensorDataRange<type>(input, chunkBegin, chunkSize), 
                proto,                                                  
                *context,                                               
                options));                                              
            return;                                                     
          }
        */
    }
}

/**
  | DeserializeTensorData() is specialized
  | for each supported combination of
  | SerializationFormat and output type.
  | 
  | The default implementation throws
  | an exception, but this function can
  | be specialized to support different
  | combinations.
  |
  */
lazy_static!{
    /*
    pub fn deserialize_tensor_data<Context: BaseContext, const Format: TensorProto_SerializationFormat, T>(
        params: &DeserializeParams<T,Context>)
    {

        todo!();
        /*
          CAFFE_ENFORCE(
              false,
              "unsupported serialization format ",
              static_cast<int>(params.tensor_proto.data_format()),
              " when deserializing float data");
        */
    }
    */
}


#[macro_export] macro_rules! deserialize_format_case {
    ($format:ident) => {
        todo!();
        /*
        
          case TensorProto_SerializationFormat_##format: {                      
            DeserializeTensorData<TensorProto_SerializationFormat_##format, T>( 
                params);                                                        
            return;                                                             
          }
        */
    }
}


#[macro_export] macro_rules! deserialize_type_case {
    ($proto_type:ident, $type:ident) => {
        todo!();
        /*
        
          case TensorProto_DataType_##proto_type: {                              
            DeserializeTensorBody(                                               
                format,                                                          
                GetMutableTensorDataRange<type>(*tensor, chunkBegin, chunkSize), 
                tensor_proto,                                                    
                context);                                                        
            return;                                                              
          }
        */
    }
}


pub const kTensorBlobType: &'static str = "Tensor";

/**
  | String used to separate chunk id from
  | the blob name when storing in DB
  |
  */
pub const kChunkIdSeparator: &'static str = "#%";

/**
  | Constants for use in the
  | BlobSerializationOptions chunk_size field.
  |
  | These should ideally be defined in
  | caffe2.proto so they can be exposed across
  | languages, but protobuf does not appear to
  | allow defining constants.
  */
pub const kDefaultChunkSize: i32 = 0;
pub const kNoChunking:       i32 = -1;

pub type SerializationAcceptor = fn(blobName: &String, data: &String) -> ();

/**
  | @brief
  | 
  | BlobSerializerBase is an abstract
  | class that serializes a blob to a string.
  | 
  | This class exists purely for the purpose
  | of registering type-specific serialization
  | code. If you need to serialize a specific
  | type, you should write your own Serializer
  | class, and then register it using
  | 
  | REGISTER_BLOB_SERIALIZER. For a detailed
  | example, see TensorSerializer for
  | details.
  |
  */
pub trait BlobSerializerBase {

    /**
      | -----------
      | @brief
      | 
      | The virtual function that returns a
      | serialized string for the input blob.
      | 
      | -----------
      | @param blob
      | 
      | the input blob to be serialized.
      | ----------
      | @param name
      | 
      | the blob name to be used in the serialization
      | implementation. It is up to the implementation
      | whether this name field is going to be
      | used or not.
      | ----------
      | @param acceptor
      | 
      | a lambda which accepts key value pairs
      | to save them to storage. serailizer
      | can use it to save blob in several chunks
      | acceptor should be thread-safe
      |
      */
    fn serialize(
        &mut self,
        pointer:   *const c_void,
        type_meta: TypeMeta,
        name:      &String,
        acceptor:  SerializationAcceptor,
        options:   Option<&BlobSerializationOptions>);

}

/**
  | @brief
  | 
  | BlobDeserializerBase is an abstract
  | class that deserializes a blob from
  | a BlobProto or a TensorProto.
  |
  */
pub trait BlobDeserializerBase {

    /// Deserializes from a BlobProto object.
    fn deserialize(&mut self, 
        proto: &BlobProto, 
        blob: *mut Blob);
}

/**
  | Serializes the given blob, if possible.
  | 
  | -----------
  | @note
  | 
  | this serialization uses the registration
  | mechanism and one has to implement specific
  | serialization approaches for specific
  | classes.
  | 
  | Acceptor should take care of writing
  | data to the actual storage.
  |
  */
#[inline] pub fn serialize_blob_from_registration_mechanism(
    blob:     &Blob,
    name:     &String,
    acceptor: SerializationAcceptor,
    options:  Option<BlobSerializationOptions>)  
{

    todo!();
    /*
        SerializeBlob(blob.GetRaw(), blob.meta(), name, std::move(acceptor), options);
    */
}

/**
  | @brief
  | 
  | Convenience function to serialize
  | a blob to a string.
  | 
  | This is a convenience function to serialize
  | small Blobs that produce manageable
  | serialized strings. To serialize big
  | blobs such as large sparse tensors,
  | use the fully-functional interface
  | in blob_serializer_base.h.
  | 
  | -----------
  | @note
  | 
  | this function doesn't do chunking and
  | might break with big tensors.
  |
  */
#[inline] pub fn serialize_blob_convenience(blob: &Blob, name: &String) -> String {
    
    todo!();
    /*
        return SerializeBlob(blob.GetRaw(), blob.meta(), name);
    */
}

#[inline] pub fn serialize_blob(
    pointer: *const c_void,
    type_meta: TypeMeta,
    name: &String,
    acceptor: SerializationAcceptor,
    options: Option<&BlobSerializationOptions>)  {

    match options {
        Some(options) => {
            todo!();
            /*
                std::unique_ptr<BlobSerializerBase> serializer(
                  CreateSerializer(typeMeta.id()));
              CAFFE_ENFORCE(serializer, "No known serializer for ", typeMeta.name());
              serializer->SerializeWithOptions(pointer, typeMeta, name, std::move(acceptor), options);
            */

        }
        None => {

            todo!();
            /*
                std::string data;
              BlobSerializerBase::SerializationAcceptor acceptor =
                  [&data](const std::string&, const std::string& blob_str) {
                    DCHECK(data.empty()); // should be called once with kNoChunking
                    data = blob_str;
                  };
              BlobSerializationOptions options;
              options.set_chunk_size(kNoChunking);
              SerializeBlob(pointer, typeMeta, name, acceptor, options);
              return data;
            */

        }
    }
}

/**
  | Deserializes from a string containing
  | either BlobProto or TensorProto. If the deserialization
  | fails, the content in the blob should
  | no longer be trusted.
  |
  */
#[inline] pub fn deserialize_blob_convenience(
    content: &String, 
    result: *mut Blob) 
{
    todo!();
    /*
        BlobProto blob_proto;
      CAFFE_ENFORCE(
          blob_proto.ParseFromString(content),
          "Cannot parse content into a BlobProto.");
      DeserializeBlob(blob_proto, result);
    */
}

#[inline] pub fn deserialize_blob(
    blob_proto: &BlobProto, 
    result: *mut Blob)
{
    
    todo!();
    /*
        if (blob_proto.type() == kTensorBlobType) {
        // This is a tensor object. Depending on the device type, we will
        // use the corresponding TensorDeserializer.
        auto deserializer = CreateDeserializer(
            "Tensor" +
            DeviceTypeName(blob_proto.tensor().device_detail().device_type()));
        // Tensor's deserializer should always be registered, but we will double
        // check if it is not null anyway.
        CAFFE_ENFORCE(deserializer.get());
        deserializer->Deserialize(blob_proto, result);
      } else {
        auto deserializer = CreateDeserializer(blob_proto.type());
        CAFFE_ENFORCE(
            deserializer.get(),
            "No registered deserializer for type ",
            blob_proto.type());
        deserializer->Deserialize(blob_proto, result);
      }
    */
}

/// Get dimensions from Tensor proto
#[inline] pub fn dims_from_tensor_proto(
    proto: &TensorProto) -> Vec<i64> {
    
    todo!();
    /*
        std::vector<int64_t> dims;
      dims.reserve(proto.dims().size());
      for (const int64_t d : proto.dims()) {
        dims.push_back(d);
      }
      return dims;
    */
}

/// Get number of elements from Tensor proto
#[inline] pub fn numel_from_tensor_proto(
    tensor_proto: &TensorProto) -> i64 {
    
    todo!();
    /*
        int64_t numel = 1;
      for (const int64_t d : tensor_proto.dims()) {
        numel *= d;
      }
      return numel;
    */
}

/// Get data type from Tensor proto
#[inline] pub fn get_data_type(
    tensor_proto: &TensorProto) -> TypeMeta {
    
    todo!();
    /*
        TypeMeta dtype;
      if (tensor_proto.data_type() != TensorProto_DataType_UNDEFINED) {
        dtype = DataTypeToTypeMeta(tensor_proto.data_type());
      } else {
        Blob temp_blob;
        DeserializeBlob(tensor_proto.string_data(0), &temp_blob);
        dtype = temp_blob.meta();
      }
      return dtype;
    */
}

/**
  | Get TensorOptions from Tensor proto
  | Assumes TensorProto is not empty
  |
  */
#[inline] pub fn tensor_options_from_proto(
    tensor_proto: &TensorProto) -> TensorOptions {
    
    todo!();
    /*
        return at::dtype(GetDataType(tensor_proto))
          .device(OptionToDevice(tensor_proto.device_detail()));
    */
}

#[inline] pub fn context_from_proto<Context: BaseContext>(
    tensor_proto: &TensorProto) -> Box<Context> {
    
    todo!();
    /*
        auto device = OptionToDevice(tensor_proto.device_detail());
      return CreateContext(device);
    */
}

/**
  | Get an empty Tensor from the TensorProto
  | given the meta data in proto (data type
  | and size of the Tensor) without actually
  | filling in the data.
  | 
  | We need this function because we want
  | to construct a fully initialized Tensor
  | in the beginning instead of keeping
  | partially initialized Tensor around
  | the process. Consider the case when
  | we have a Tensor that is split into multiple
  | protos during serialization, in deserialization,
  | we have to fill the Tensor in multiple
  | calls to Deserialize, therefore we
  | need to create a new Tensor with the correct
  | size and data type before the call to
  | 
  | Deserialize, because otherwise we
  | will have to check whether the function
  | call is the first call to initialize
  | the underlying Tensor, which makes
  | the function stateful and complicated.
  | 
  | The legacy code get away with this problem
  | by passing in a partially initialized
  | Tensor and use Resize and mutable_data
  | to set the correct size, data type and
  | allocate memory for the
  | 
  | Tensor, so the state is encoded in these
  | function calls. e.g. mutable_data
  | will allocate memory on the first call
  | and it will return a pointer to the allocated
  | memory on later calls.
  |
  */
#[inline] pub fn empty_tensor_from_proto(tensor_proto: &TensorProto) -> Tensor {
    
    todo!();
    /*
        auto context = ContextFromProto(tensor_proto);
      context->SwitchToDevice();
      if (NumelFromTensorProto(tensor_proto) == 0 &&
          tensor_proto.data_type() == TensorProto_DataType_UNDEFINED) {
        // TODO: remove when serialization of dtype uninitialized tensor is removed
        return caffe2::empty(
            {0},
            at::dtype<float>().device(
                OptionToDevice(tensor_proto.device_detail())));
      } else {
        return caffe2::empty(
            DimsFromTensorProto(tensor_proto),
            TensorOptionsFromProto(tensor_proto));
      }
    */
}

#[inline] pub fn enable_byte_encoding<T>() -> bool {
    todo!();
    /*
        // if typeSize == 1, endianness does not matter. Else check for endianness.
      if (sizeof(T) > 1 && !kIsLittleEndian) {
        return false;
      }
      return FLAGS_caffe2_serialize_using_bytes_as_holder;
    */
}

#[inline] pub fn enable_byte_encoding_float16() -> bool {
    
    todo!();
    /*
        if (!kIsLittleEndian) {
        return false;
      }
      // Check if special casing for float is enabled if
      // caffe2_serialize_using_bytes_as_holder is not enabled.
      return FLAGS_caffe2_serialize_using_bytes_as_holder ||
          FLAGS_caffe2_serialize_fp16_as_bytes;
    */
}


#[inline] pub fn serialize_using_bytes_or_int32<T, S, Context: BaseContext>(
    enable_byte_encoding: bool,
    input: &[S],
    context: &mut Context,
    proto: &mut TensorProto) 
{
    todo!();
    /*
        if (enableByteEncoding) {
        const auto bufSize = sizeof(T) * input.size();
        auto* byteData = reinterpret_cast<const uint8_t*>(input.data());
        unique_ptr<uint8_t[]> buffer(new uint8_t[bufSize]);
        context.template CopyToCPU<uint8_t>(bufSize, byteData, buffer.get());
        context.FinishDeviceComputation();
        proto.set_byte_data(buffer.get(), bufSize);
      } else {
        detail::CopyToProtoWithCast(
            input.size(),
            reinterpret_cast<const T*>(input.data()),
            proto.mutable_int32_data(),
            &context);
      }
    */
}

#[inline] pub fn deserialize_tensor<Context: BaseContext>(
    tensor_proto: &TensorProto,
    tensor:       *mut Tensor,
    context:      &mut Context)  
{
    todo!();
    /*
        int64_t chunkBegin = 0;
      auto chunkEnd = tensor->numel();
      if (tensor_proto.has_segment()) {
        chunkBegin = tensor_proto.segment().begin();
        chunkEnd = tensor_proto.segment().end();
      }
      CAFFE_ENFORCE(
          0 <= chunkBegin && chunkBegin <= chunkEnd && chunkEnd <= tensor->numel(),
          "Invalid chunk ",
          chunkBegin,
          ' ',
          chunkEnd,
          " with total tensor size ",
          tensor->numel());
      auto chunkSize = chunkEnd - chunkBegin;

      if (!tensor_proto.has_data_type()) {
        // If the data_type field is not set, this either means it was not present
        // in the serialized data, or it was set to an enum value that we don't know
        // about.  This likely means that the serialized data was written by a
        // different version of the software using a new data type value that we
        // don't understand.
        throw std::runtime_error(
            "Cannot deserialize tensor: unrecognized data type");
      }

      // If the data_format field is not present this is an older buffer
      // serialized with the FMT_PROTOBUF format.
      auto format = tensor_proto.has_data_format()
          ? static_cast<TensorProto::SerializationFormat>(
                tensor_proto.data_format())
          : TensorProto_SerializationFormat_FMT_PROTOBUF;

      switch (tensor_proto.data_type()) {
        DESERIALIZE_TYPE_CASE(FLOAT, float);
        DESERIALIZE_TYPE_CASE(INT32, int32_t);
        DESERIALIZE_TYPE_CASE(STRING, std::string);
        DESERIALIZE_TYPE_CASE(BOOL, bool);
        DESERIALIZE_TYPE_CASE(UINT8, uint8_t);
        DESERIALIZE_TYPE_CASE(INT8, int8_t);
        DESERIALIZE_TYPE_CASE(UINT16, uint16_t);
        DESERIALIZE_TYPE_CASE(INT16, int16_t);
        DESERIALIZE_TYPE_CASE(INT64, int64_t);
        DESERIALIZE_TYPE_CASE(FLOAT16, at::Half);
        DESERIALIZE_TYPE_CASE(DOUBLE, double);
        case TensorProto_DataType_BYTE:
          // BYTE is special, since it is a legacy data type value that effectively
          // means the same thing as UINT8, except that it used to be serialized in
          // a different format.  Recent code always writes out byte data with the
          // UINT8 type, never BYTE, but let's leave legacy deserialization code in
          // place for now just in case we ever encounter an old blob using this
          // format.
          DeserializeLegacyByteData(
              format,
              DeserializeParams<uint8_t>{
                  GetMutableTensorDataRange<uint8_t>(
                      *tensor, chunkBegin, chunkSize),
                  tensor_proto,
                  context});
          return;
        case TensorProto_DataType_UNDEFINED: {
          Blob temp_blob;
          void* raw_ptr = nullptr;
          for (const auto i : c10::irange(chunkSize)) {
            DeserializeBlob(tensor_proto.string_data(i), &temp_blob);
            if (i == 0) {
              raw_ptr = tensor->raw_mutable_data(temp_blob.meta());
            }
            temp_blob.meta().copy()(
                temp_blob.GetRaw(),
                static_cast<char*>(raw_ptr) +
                    (i + chunkBegin) * temp_blob.meta().itemsize(),
                1);
          }
        } return;
        case TensorProto_DataType_ZERO_COLLISION_HASH:
          CAFFE_ENFORCE(
              false,
              "Deserialization for zero collision hash type is supported by "
              "specialized deserializer ZeroCollisionIdHashDeserializer");
          return;
        case TensorProto_DataType_REBATCHING_BUFFER:
          CAFFE_ENFORCE(
              false,
              "Deserialization for REBATCHING_BUFFER type is supported by "
              "specialized serializer RebatchingBufferDeserialier");
          return;
          // Note: we intentially do not provide "default:" so if any new data types
      }

      // We should never reach here unless there is a bug and protobuf somehow
      // returns an unexpected value.  protobuf should filter out all unknown enum
      // values, and the has_data_type() check above will catch that case.
      CAFFE_ENFORCE(
          false,
          "Deserialization for REBATCHING_BUFFER type is supported by "
          "specialized serializer RebatchingBufferDeserialier");
    */
}

/**
  | Return a mutable Range pointing to a
  | portion of the tensor's data field.
  | 
  | Returns a Range pointing to the elements
  | starting at the specified start index,
  | and including the specified number
  | of elements.
  |
  */
pub fn get_mutable_tensor_data_range<T>(
    tensor:       &mut Tensor, 
    start:        usize, 
    num_elements: usize) -> Range<*mut T> 
{
    todo!();
    /*
      CAFFE_ENFORCE(
          start + numElements <= tensor.numel(),
          "Requested invalid mutable tensor range [",
          start,
          ", ",
          start + numElements,
          ") with total tensor size ",
          tensor.numel());
      return Range<T*>(tensor.template mutable_data<T>() + start, numElements);
    */
}

pub fn get_tensor_data_range<T>(
    tensor: &Tensor,
    start:  usize,
    num_elements: usize) -> &[T] 
{
    todo!();
    /*
      CAFFE_ENFORCE(
          start + numElements <= tensor.numel(),
          "Requested invalid tensor range [",
          start,
          ", ",
          start + numElements,
          ") with total tensor size ",
          tensor.numel());
      return c10::ArrayRef<T>(tensor.template data<T>() + start, numElements);
    */
}

/**
  | Converts MessageLite to string while also
  | checking that SerializeAsString succeeds. Pass
  | description of class/function of the call if
  | you'd like it appended to the error message.
  */
#[inline] pub fn serialize_as_string_enforce_check(
    msg:            &MessageLite,
    error_location: *const u8) -> String {
    
    todo!();
    /*
        std::string serialize_output;
      bool result = msg.SerializeToString(&serialize_output);
      if (!error_location) {
        CAFFE_ENFORCE(result, "protobuf::SerializeToString failed");
      } else {
        CAFFE_ENFORCE(
            result, "protobuf::SerializeToString failed for ", error_location);
      }
      return serialize_output;
    */
}

#[inline] pub fn copy_to_proto_with_cast<Context: BaseContext, SrcType, DstType>(
    size:    usize,
    src:     *const SrcType,
    field:   *mut RepeatedField<DstType>,
    context: *mut Context) 
{
    todo!();
    /*
        // TODO: we are having one unnecessary copy here if the context is already
      // CPUContext. Remove it if it is performance critical.
      unique_ptr<SrcType[]> buffer(new SrcType[size]);
      context->template CopyToCPU<SrcType>(size, src, buffer.get());
      context->FinishDeviceComputation();
      field->Reserve(size);
      for (size_t i = 0; i < size; ++i) {
        field->Add(static_cast<DstType>(buffer[i]));
      }
    */
}

#[inline] pub fn copy_from_proto_as_is<Context: BaseContext, SrcType, DstType>(
    size:    usize,
    field:   &RepeatedField<SrcType>,
    dst:     *mut DstType,
    context: *mut Context)
{
    todo!();
    /*
        static_assert(
          sizeof(SrcType) == sizeof(DstType),
          "The source type and dest type cannot be copied as-is. Did "
          "you mean CopyFromProtoWithCast?");
      CAFFE_ENFORCE_EQ(size, field.size(), "Incorrect proto field size.");
      context->template CopyFromCPU<DstType>(
          size, reinterpret_cast<const DstType*>(field.data()), dst);
    */
}

#[inline] pub fn copy_from_proto_with_cast<Context: BaseContext, SrcType, DstType>(
    size:    usize,
    field:   &RepeatedField<SrcType>,
    dst:     *mut DstType,
    context: *mut Context)
{
    todo!();
    /*
        CAFFE_ENFORCE_EQ(size, field.size(), "Incorrect proto field size.");
      // TODO: we are having one unnecessary copy here if the context is already
      // CPUContext. Remove it if it is performance critical.
      unique_ptr<DstType[]> buffer(new DstType[size]);
      const SrcType* src = field.data();
      for (size_t i = 0; i < size; ++i) {
        buffer[i] = static_cast<DstType>(src[i]);
      }
      context->template CopyFromCPU<DstType>(size, buffer.get(), dst);
    */
}

/**
  | Convert BlobProto to string with success
  | checks.
  |
  */
#[inline] pub fn serialize_blob_proto_as_string_enforce_check(blob: &BlobProto) -> String {
    
    todo!();
    /*
        return SerializeAsString_EnforceCheck(blob, blob.name().c_str());
    */
}

/**
  | Make space for new elements to be copied to
  | the end of the repeated field.
  |
  | The new space is not guaranteed to be
  | initialized.
  */
#[inline] pub fn extend_repeated_field<T>(
    field: *mut RepeatedField<T>,
    size: usize) 
{
    todo!();
    /*
        field->Reserve(field->size() + size);
    #if GOOGLE_PROTOBUF_VERSION >= 3000000
      field->AddNAlreadyReserved(size);
    #else
      // We unfortunately do still need to support old protobuf versions in some
      // build configurations.
      for (size_t i = 0; i < size; ++i) {
        field->Add(0);
      }
    #endif
    */
}

#[inline] pub fn copy_to_proto_as_is<Context: BaseContext, SrcType, DstType>(
    size:    usize,
    src:     *const SrcType,
    field:   *mut RepeatedField<DstType>,
    context: *mut Context) {
    todo!();
    /*
        static_assert(
          sizeof(SrcType) == sizeof(DstType),
          "The source type and dest type cannot be copied as-is. Did "
          "you mean CopyToProtoWithCast?");
      ExtendRepeatedField(field, size);
      context->template CopyToCPU<SrcType>(
          size, src, reinterpret_cast<SrcType*>(field->mutable_data()));
      // Make sure that we finish the copy into the protobuf.
      context->FinishDeviceComputation();
    */
}

#[inline] pub fn deserialize_tensor_body<Context: BaseContext, T>(
    format:       SerializationFormat,
    dest:         Range<*mut T>,
    tensor_proto: &TensorProto,
    context:      &mut Context) 
{
    todo!();
    /*
        DeserializeParams<T> params(dest, tensor_proto, context);
      switch (format) {
        DESERIALIZE_FORMAT_CASE(FMT_PROTOBUF);
      }

      // This can happen if the blob was serialized by a newer version of the code
      // using some new format value that we don't understand.
      CAFFE_ENFORCE(
          false,
          "unsupported serialization format " + c10::str(static_cast<int>(format)));
    */
}

#[inline] pub fn serialize_tensor_data_f32<Context: BaseContext>(
    params: &SerializeParams<f32,Context>)
{
    todo!();
    /*
        params.CopyToRepeatedField(params.tensor_proto.mutable_float_data());
    */
}

#[inline] pub fn serialize_tensor_data_f64<Context: BaseContext>(
    params: &SerializeParams<f64, Context>)
{
    todo!();
    /*
        params.CopyToRepeatedField(params.tensor_proto.mutable_double_data());
    */
}

#[inline] pub fn serialize_tensor_data_string<Context: BaseContext>(
    params: &SerializeParams<String, Context>) 
{
    todo!();
    /*
        params.tensor_proto.mutable_string_data()->Reserve(params.input.size());
      for (const std::string& element : params.input) {
        params.tensor_proto.add_string_data(element);
      }
    */
}

#[inline] pub fn serialize_tensor_data_f16<Context: BaseContext>(
    params: &SerializeParams<f16,Context>)
{
    todo!();
    /*
        SerializeUsingBytesOrInt32<uint16_t>(
          EnableByteEncodingFloat16(),
          params.input,
          params.context,
          params.tensor_proto);
    */
}

#[inline] pub fn serialize_tensor_data_i64<Context: BaseContext>(
    params: &SerializeParams<i64, Context>) 
{
    todo!();
    /*
        params.CopyToRepeatedField(params.tensor_proto.mutable_int64_data());
    */
}

#[inline] pub fn serialize_tensor_data_i32<Context: BaseContext>(
    params: &SerializeParams<i32, Context>)
{
    
    todo!();
    /*
        params.CopyToRepeatedField(params.tensor_proto.mutable_int32_data());
    */
}

/**
  | SerializeParams is just a helper class
  | to consolidate the parameters required
  | for serializing tensor data so they
  | can be passed around more easily.
  | 
  | It also contains some helper functions
  | to perform some operations on the parameters
  | that are shared by multiple serialization
  | functions.
  |
  */
pub struct SerializeParams<'a, T, Context: BaseContext> {
  input:        &'a mut [T],
  tensor_proto: &'a mut TensorProto,
  context:      &'a mut Context,
  options:      &'a BlobSerializationOptions,
}

impl<'a, T, Context: BaseContext> SerializeParams<'a,T, Context> {
    
    #[inline] pub fn set_data_format(&self, format: SerializationFormat)  {
        
        todo!();
        /*
            tensor_proto.set_data_format(format);
        */
    }
    
    #[inline] pub fn copy_to_repeated_field(&self, field: *mut RepeatedField<T>)  {
        
        todo!();
        /*
            detail::CopyToProtoAsIs(input.size(), input.data(), field, &context);
        */
    }
}

/**
  | DeserializeParams is just a helper
  | class to consolidate the parameters
  | required for deserializing tensor
  | data so they can be passed around more
  | easily.
  | 
  | It also contains some helper functions
  | to perform some operations on the parameters
  | that are shared by multiple deserialization
  | functions.
  |
  */
pub struct DeserializeParams<'a, T, Context: BaseContext> {
  dest:         Range<*mut T>,
  tensor_proto: &'a TensorProto,
  context:      &'a mut Context,
}

impl<'a,T,Context: BaseContext> DeserializeParams<'a,T,Context> {
    
    #[inline] pub fn copy_from_repeated_field(&self, field: &RepeatedField<T>)  {
        
        todo!();
        /*
            detail::CopyFromProtoAsIs(dest.size(), field, dest.data(), &context);
        */
    }
    
    #[inline] pub fn literal_copy(&self, src: &str)  {
        
        todo!();
        /*
            // Simply copy the data as-is from src to dest
        CAFFE_ENFORCE_EQ(
            dest.size() * sizeof(T),
            src.size(),
            "incorrect data size when deserializing blob: ",
            dest.size(),
            " * ",
            sizeof(T),
            " != ",
            src.size());
        context.CopyBytesFromCPU(src.size(), src.data(), dest.data());
        */
    }
    
    #[inline] pub fn copy_from_bytes_or_int32(&self)  {
        
        todo!();
        /*
            DeserializeFromBytesOrInt32<T>(tensor_proto, dest, context);
        */
    }
}

/**
  | @brief
  | 
  | StringSerializer is the serializer
  | for
  | 
  | String.
  | 
  | StringSerializer takes in a blob that
  | contains a String, and serializes it
  | into a BlobProto protocol buffer.
  |
  */
pub struct StringSerializer {

}

impl BlobSerializerBase for StringSerializer {

    /**
      | Serializes a Blob. Note that this blob
      | has to contain Tensor, otherwise this
      | function produces a fatal error.
      |
      */
    #[inline] fn serialize(
        &mut self, 
        pointer: *const c_void,
        type_meta: TypeMeta,
        name: &String,
        acceptor: SerializationAcceptor,
        options:   Option<&BlobSerializationOptions>)
    {

        todo!();
        /*
            CAFFE_ENFORCE(typeMeta.Match<std::string>());

            BlobProto blob_proto;
            blob_proto.set_name(name);
            blob_proto.set_type("std::string");
            blob_proto.set_content(*static_cast<const std::string*>(pointer));
            acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
        */
    }
}

/**
  | @brief
  | 
  | StringDeserializer is the deserializer
  | for Strings.
  |
  */
pub struct StringDeserializer {

}

impl BlobDeserializerBase for StringDeserializer {

    #[inline] fn deserialize(&mut self, proto: &BlobProto, blob: *mut Blob)  {
        
        todo!();
        /*
            *blob->GetMutable<std::string>() = proto.content();
        */
    }
}

#[inline] pub fn get_gpuid_for_pointer(ptr: *const c_void) -> i32 {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn deserialize_legacy_byte_data<Context: BaseContext>(
    format: SerializationFormat,
    params: &DeserializeParams<u8,Context>)
{
    todo!();
    /*
        // The BYTE format should only be used for very old blobs that don't
      // have a data_format field in the first place.  Let's log this case but
      // continue attempting deserialization anyway.
      CAFFE_ENFORCE_EQ(
          format,
          TensorProto_SerializationFormat_FMT_PROTOBUF,
          "found serialized blob with BYTE data type but unexpected data format ",
          static_cast<int>(format));

      params.LiteralCopy(params.tensor_proto.byte_data());
    */
}

#[macro_export] macro_rules! deserialize_impl {
    ($type:ident, $data_type:ident, $body:expr) => {
        /*
        paste!{
            pub fn deserialize_tensor_data< [<TensorProto_SerializationFormat_ $data_type>], $type >(
                params: &DeserializeParams<$type>) 
            {
                $body
            }
        }
        */
    }
}

deserialize_impl!{i64, FMT_PROTOBUF, {
    todo!();
    /*
       params.CopyFromRepeatedField(params.tensor_proto.int64_data());
       */
}}


deserialize_impl!{i32, FMT_PROTOBUF, {
    todo!();
    /*
       params.CopyFromRepeatedField(params.tensor_proto.int32_data());
       */
}}

deserialize_impl!{u16, FMT_PROTOBUF, {
    todo!();
    /*
       params.CopyFromBytesOrInt32();
       */
}}

deserialize_impl!{i16, FMT_PROTOBUF, {
    todo!();
    /*
       params.CopyFromBytesOrInt32();
       */
}}

deserialize_impl!{u8, FMT_PROTOBUF, {
    todo!();
    /*
       params.CopyFromBytesOrInt32();
       */
}}

deserialize_impl!{i8, FMT_PROTOBUF, {
    todo!();
    /*
       params.CopyFromBytesOrInt32();
       */
}}

deserialize_impl!{bool, FMT_PROTOBUF, {
    todo!();
    /*
       params.CopyFromBytesOrInt32();
       */
}}

deserialize_impl!{f16, FMT_PROTOBUF, {
    todo!();
    /*
       DeserializeFromBytesOrInt32<uint16_t, at::Half>(
       params.tensor_proto, params.dest, params.context);
       */
}}

deserialize_impl!{f32, FMT_PROTOBUF, {
    todo!();
    /*
       params.CopyFromRepeatedField(params.tensor_proto.float_data());
       */
}}

deserialize_impl!{f64, FMT_PROTOBUF, {
    todo!();
    /*
       params.CopyFromRepeatedField(params.tensor_proto.double_data());
       */
}}

deserialize_impl!{String, FMT_PROTOBUF, {
    todo!();
    /*
  CAFFE_ENFORCE_EQ(
      params.dest.size(),
      params.tensor_proto.string_data().size(),
      "incorrect data size in serialized data: ",
      params.dest.size(),
      " != ",
      params.tensor_proto.string_data().size());
  for (const auto i : c10::irange(params.dest.size())) {
    params.dest[i] = params.tensor_proto.string_data(i);
  }
  */
}}

pub trait IntegralOrBoolean {} macro_rules! i_or_b { 
    ($($t:ty),*) => { 
        $(impl IntegralOrBoolean for $t {})* 
    } 
}
i_or_b![i8, u8, i16, u16, i32, u32, i64, u64, bool];


pub fn serialize_tensor_data<T: IntegralOrBoolean, Context: BaseContext>(
    params: &SerializeParams<T, Context>) 
{
    todo!();
    /*
      SerializeUsingBytesOrInt32<T>(
          EnableByteEncoding<T>(),
          params.input,
          params.context,
          params.tensor_proto);
    */
}

register_blob_deserializer!{
    /*
    TensorCUDA, 
    TensorDeserializer
    */
}
