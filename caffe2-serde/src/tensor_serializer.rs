crate::ix!();

/**
  | -----------
  | @brief
  | 
  | TensorSerializer is the serializer
  | for Tensors.
  | 
  | TensorSerializer takes in a blob that
  | contains a Tensor, and serializes it
  | into a TensorProto protocol buffer.
  |
  */
pub struct TensorSerializer<Context: BaseContext> {
    context: Box<Context>,
}

impl<Context: BaseContext> TensorSerializer<Context> {
    
    #[inline] pub fn serialize(
        &mut self, 
        input:       &Tensor,
        name:        &String,
        proto_ptr:   *mut TensorProto,
        options:     &BlobSerializationOptions,
        chunk_begin: usize,
        chunk_size:  i32)  {

        todo!();
        /*
            CAFFE_ENFORCE(
          chunkBegin <= input.numel(),
          "Chunk begin is out of tensor: ",
          chunkBegin,
          ' ',
          input.numel());
      if (chunkBegin + chunkSize > input.numel()) {
        chunkSize = input.numel() - chunkBegin;
      }

      if (chunkSize != 0) {
        CAFFE_ENFORCE(
            input.raw_data(),
            "The input does not have data input yet. This is probably because you "
            "created a tensor of non-zero shape but never filled its data via "
            "mutable_data() calls. This means that it makes no sense to serialize "
            "the tensor content.");
      } else if (!input.dtype_initialized()) {
        C10_LOG_EVERY_MS(WARNING, 1000)
            << "You're trying to serialize tensor with zero numel and no dtype. "
            << "This is a legacy behavior and it WILL BREAK. Contact PyTorch team "
            << "for details. Offending blob name: " << name;
      }

      TensorProto& proto = *proto_ptr;
      proto.mutable_segment()->set_begin(chunkBegin);
      proto.mutable_segment()->set_end(chunkBegin + chunkSize);

      for (const auto i : c10::irange(input.dim())) {
        proto.add_dims(input.size(i));
      }
      StoreDeviceDetail(input, &proto);

      const TensorProto::DataType data_type = TypeMetaToDataType(input.dtype());
      proto.set_data_type(data_type);
      // TODO: use CUDAGuard here instead of context and employ explicit sync
      // copy
      auto context = CreateContext(input.GetDevice());
      switch (data_type) {
        SERIALIZE_TYPE_CASE(FLOAT, float)
        SERIALIZE_TYPE_CASE(INT32, int32_t)
        SERIALIZE_TYPE_CASE(STRING, std::string)
        SERIALIZE_TYPE_CASE(BOOL, bool)
        SERIALIZE_TYPE_CASE(UINT8, uint8_t)
        SERIALIZE_TYPE_CASE(INT8, int8_t)
        SERIALIZE_TYPE_CASE(UINT16, uint16_t)
        SERIALIZE_TYPE_CASE(INT16, int16_t)
        SERIALIZE_TYPE_CASE(INT64, int64_t)
        SERIALIZE_TYPE_CASE(FLOAT16, at::Half)
        SERIALIZE_TYPE_CASE(DOUBLE, double)
        case TensorProto_DataType_BYTE:
          LOG(FATAL) << "This should not happen. When serializing, "
                        "BYTE is deprecated and moved to UINT8.";
          return;
        case TensorProto_DataType_UNDEFINED:
          proto.mutable_string_data()->Reserve(chunkSize);
          if (chunkSize > 0) {
            const char* raw_data = static_cast<const char*>(input.raw_data());
            for (const auto i : c10::irange(chunkBegin, chunkBegin + chunkSize)) {
              proto.add_string_data(SerializeBlob(
                  raw_data + i * input.itemsize(), input.dtype(), ""));
            }
          }
          return;
        case TensorProto_DataType_ZERO_COLLISION_HASH:
          CAFFE_ENFORCE(
            false,
            "Serialization for zero collision hash type is supported by "
            "specialized serializer ZeroCollisionIdHashSerializer");
          return;
        case TensorProto_DataType_REBATCHING_BUFFER:
          CAFFE_ENFORCE(
            false,
            "Serialization for REBATCHING_BUFFER type is supported by "
            "specialized serializer RebatchingBufferSerialier");
          return;

          // Note: we intentially do not provide "default:" so if any new data types
          // are added, the compiler should warn the user to add the case here.
      }

      CAFFE_ENFORCE(false, "unexpected data type during tensor serialization");
        */
    }
    
    /// A utility function to store the device context detauls.
    #[inline] pub fn store_device_detail(
        &mut self, 
        input: &Tensor,
        proto: *mut TensorProto)  
    {
        todo!();
        /*
            ExtractDeviceOption(proto->mutable_device_detail(), input.GetDevice());
        */
    }
}

impl<Context: BaseContext> BlobSerializerBase for TensorSerializer<Context> {
    
    /**
     * Serializes a Blob. Note that this blob has to contain Tensor,
     * otherwise this function produces a fatal error.
     */
    #[inline] fn serialize(&mut self, 
        pointer:   *const c_void,
        type_meta: TypeMeta,
        name:      &String,
        acceptor:  SerializationAcceptor,
        options:   Option<&BlobSerializationOptions>)  
    {
        todo!();
        /*
            CAFFE_ENFORCE(typeMeta.Match<Tensor>());
      const auto& tensor = *static_cast<const Tensor*>(pointer);
      auto chunk_size = options.chunk_size();
      if (chunk_size == kNoChunking) {
        chunk_size = tensor.numel() + 1; // to account for empty tensors
      } else if (chunk_size == kDefaultChunkSize) {
        chunk_size = FLAGS_caffe2_tensor_chunk_size;
      }

      auto processChunk = [&](int64_t chunkStart) {
        BlobProto blob_proto;
        blob_proto.set_name(name);
        blob_proto.set_type(kTensorBlobType);
        TensorProto& proto = *blob_proto.mutable_tensor();
        proto.set_name(name);
        this->Serialize(
            tensor,
            name,
            blob_proto.mutable_tensor(),
            options,
            chunkStart,
            chunk_size);
        acceptor(
            c10::str(name, kChunkIdSeparator, chunkStart / chunk_size),
            SerializeBlobProtoAsString_EnforceCheck(blob_proto));
      };

    #ifndef __ANDROID__
      // Poorman's IOBound ThreadPool
      SimpleQueue<size_t> chunkQueue;
      auto task = [&]() {
        size_t chunkStart;
        while (chunkQueue.Pop(&chunkStart)) {
          processChunk(chunkStart);
        }
      };
      std::vector<std::future<void>> futures;
      if (tensor.numel() > chunk_size) {
        futures.reserve(FLAGS_caffe2_max_tensor_serializer_threads);
        for (const auto i : c10::irange(FLAGS_caffe2_max_tensor_serializer_threads)) {
          futures.emplace_back(std::async(std::launch::async, task));
        }
      }
    #endif

      VLOG(1) << "Serializing blob " << name;
      // Serialize whole vector. If vector is empty, it's shape still needs to be
      // serialized in empty proto
      for (size_t chunkBegin = 0;
           chunkBegin < std::max(tensor.numel(), static_cast<int64_t>(1));
           chunkBegin += chunk_size) {
        VLOG(2) << "Starting a chunk at " << chunkBegin;
    #ifndef __ANDROID__
        if (tensor.numel() > chunk_size) {
          chunkQueue.Push(chunkBegin);
        } else {
          // Sync mode for small tensors
          processChunk(chunkBegin);
        }
    #else
        // Since Android does not have std::future, we will always do sync mode
        processChunk(chunkBegin);
    #endif
      }

    #ifndef __ANDROID__
      chunkQueue.NoMoreJobs();
      for (auto& fut : futures) {
        fut.get();
      }
    #endif
        */
    }
}
