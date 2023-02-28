crate::ix!();

///--------------------------------------
pub struct Int8TensorCPUSerializer {
    context: CPUContext,
}

impl BlobSerializerBase for Int8TensorCPUSerializer {

    #[inline] fn serialize(
        &mut self, 
        pointer:    *const c_void,
        type_meta:  TypeMeta,
        name:       &String,
        acceptor:   SerializationAcceptor,
        options:    Option<&BlobSerializationOptions>)  
    {
        todo!();
        /*
            CAFFE_ENFORCE(typeMeta.Match<Int8TensorCPU>());
        const auto& tensor = *static_cast<const Int8TensorCPU*>(pointer);
        BlobProto blob_proto;
        blob_proto.set_name(name);
        blob_proto.set_type("Int8TensorCPU");
        QTensorProto& proto = *blob_proto.mutable_qtensor();
        proto.set_name(name);
        for (int i = 0; i < tensor.t.dim(); ++i) {
          proto.add_dims(tensor.t.dim32(i));
        }
        proto.set_precision(8);
        proto.set_scale(tensor.scale);
        proto.set_bias(tensor.zero_point);
        proto.set_is_signed(false);

        const TensorProto::DataType data_type =
            TypeMetaToDataType(tensor.t.dtype());
        proto.set_data_type(data_type);
        switch (data_type) {
          case TensorProto_DataType_INT32:
            detail::CopyToProtoAsIs(
                tensor.t.numel(),
                tensor.t.template data<int32_t>(),
                proto.mutable_data(),
                &this->context_);
            break;
          case TensorProto_DataType_UINT8:
            detail::CopyToProtoWithCast(
                tensor.t.numel(),
                tensor.t.template data<uint8_t>(),
                proto.mutable_data(),
                &this->context_);
            break;
          default:
            CAFFE_ENFORCE(false, "Unsupported data type in Int8TensorCPU");
        }

        acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
        */
    }
}

///--------------------------------------
pub struct Int8TensorCPUDeserializer {
    base:    TensorDeserializer,
    context: CPUContext,
}

impl Int8TensorCPUDeserializer {

    #[inline] pub fn deserialize(
        &mut self, 
        blob_proto: &BlobProto, 
        blob:       *mut Blob)  
    {
        
        todo!();
        /*
            const QTensorProto& proto = blob_proto.qtensor();
        Int8TensorCPU* tensor = blob->template GetMutable<Int8TensorCPU>();
        tensor->scale = proto.scale();
        tensor->zero_point = proto.bias();
        vector<int> dims;
        for (const int d : proto.dims()) {
          dims.push_back(d);
        }
        tensor->t.Resize(dims);
        switch (proto.data_type()) {
          case TensorProto_DataType_INT32:
            detail::CopyFromProtoAsIs(
                tensor->t.numel(),
                proto.data(),
                tensor->t.template mutable_data<int32_t>(),
                &this->context_);
            break;
          case TensorProto_DataType_UINT8:
            detail::CopyFromProtoWithCast(
                tensor->t.numel(),
                proto.data(),
                tensor->t.template mutable_data<uint8_t>(),
                &this->context_);
            break;
          default:
            CAFFE_ENFORCE(false, "Unsupported data type in Int8TensorCPU");
        }
        */
    }
}

register_blob_serializer!{
    /*
    (TypeMeta::Id<int8::Int8TensorCPU>()), 
    int8::Int8TensorCPUSerializer 
    */
}

register_blob_deserializer!{
    /*
    Int8TensorCPU, 
    int8::Int8TensorCPUDeserializer
    */
}
