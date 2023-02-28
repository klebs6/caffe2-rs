crate::ix!();

/**
  | @brief
  | 
  | TensorDeserializer is the deserializer
  | for Tensors.
  | 
  | The device that the deserialized Tensor
  | will live under is determined by the
  | device_detail field. If you want to
  | specify the device of the deserialized
  | tensor, change the TensorProto's corresponding
  | fields before calling
  | 
  | Deserialize.
  |
  */
pub struct TensorDeserializer {

}

impl BlobDeserializerBase for TensorDeserializer {

    #[inline] fn deserialize(
        &mut self, 
        blob_proto: &BlobProto,
        blob:       *mut Blob)  
    {
        todo!();
        /*
          const auto& tensor_proto = blob_proto.tensor();
          auto context = ContextFromProto(tensor_proto);
          context->SwitchToDevice();
          if (NumelFromTensorProto(tensor_proto) == 0 &&
              tensor_proto.data_type() == TensorProto_DataType_UNDEFINED) {
            // TODO: remove after empty Tensor serialization is forbidden
            VLOG(1) << "Deseriralizing an empty Tensor.";
            BlobGetMutableTensor(
                blob,
                {0},
                at::dtype<float>().device(
                    OptionToDevice(tensor_proto.device_detail())));
          } else {
            DeserializeToTensor(
                tensor_proto,
                BlobGetMutableTensor(
                    blob,
                    DimsFromTensorProto(tensor_proto),
                    TensorOptionsFromProto(tensor_proto)));
          }
        */
    }
}

impl TensorDeserializer {
    
    /**
      | There are cases when a Tensor is split
      | into multiple protos and we have to call
      | Deserialize multiple times to get the
      | complete deserialized
      | 
      | Tensor, each call will fill part of the
      | Tensor given the segment begin and end
      | information in proto, therefore we
      | have to pass in the Tensor pointer rather
      | than create a new Tensor every time.
      | 
      | Precondition: Tensor must be initialized
      |
      */
    #[inline] pub fn deserialize_to_tensor(&mut self, 
        tensor_proto: &TensorProto,
        tensor:       *mut Tensor)  {
        
        todo!();
        /*
            CAFFE_ENFORCE(
              tensor->storage_initialized() && tensor->dtype_initialized(),
              "Tensor must be initialized before passed into Deserialize function.");
          // We create a local context for deserializing. Since Caffe2 contexts are
          // usually lightweight, this should not involve too much overhead.
          auto context = ContextFromProto(tensor_proto);
          context->SwitchToDevice();
          DeserializeTensor(tensor_proto, tensor, *context);
          context->FinishDeviceComputation();
        */
    }
    
    /**
      | Deserialize the proto and return a new
      | Tensor
      | 
      | This is a utility function that combines
      | EmptyTensorFromProto and
      | 
      | Deserialize(const TensorProto&,
      | Tensor*);
      |
      */
    #[inline] pub fn deserialize_from_tensor_proto(&mut self, 
        tensor_proto: &TensorProto) -> Tensor 
    {
        todo!();
        /*
          auto tensor = EmptyTensorFromProto(tensor_proto);
          DeserializeToTensor(tensor_proto, &tensor);
          return tensor;
        */
    }
    
}

pub fn deserialize_from_bytes_or_i32<T, D, Context: BaseContext>(
    tensor_proto: &TensorProto,
    dest:         Range<*mut D>,
    context:      &mut Context)
{
    todo!();
    /*
      if (tensor_proto.has_byte_data()) {
        auto typeSize = sizeof(T);
        CAFFE_ENFORCE(
            kIsLittleEndian || typeSize == 1,
            "Serialization with bytes not supported on big endian platform.");
        size_t numElems = tensor_proto.byte_data().size();
        if (tensor_proto.data_type() == TensorProto_DataType_UINT8) {
          if (tensor_proto.has_segment()) {
            const auto& segment = tensor_proto.segment();
            numElems = segment.end() - segment.begin();
          }
        }
        CAFFE_ENFORCE_EQ(
            typeSize * dest.size(), numElems, "Incorrect proto field size.");
        const uint8_t* protoData =
            reinterpret_cast<const uint8_t*>(tensor_proto.byte_data().data());
        context.template CopyToCPU<D>(
            dest.size(),
            reinterpret_cast<const D*>(protoData),
            dest.data());
      } else {
        // Backward compatibility with models which used int32_data field
        detail::CopyFromProtoWithCast(
            dest.size(),
            tensor_proto.int32_data(),
            reinterpret_cast<T*>(dest.data()),
            &context);
      }
    */
}
