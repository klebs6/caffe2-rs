crate::ix!();

pub struct IndexSerializer {
    base: dyn BlobSerializerBase,
}

impl IndexSerializer {
    
    #[inline] pub fn serialize(
        &mut self, 
        pointer:   *const c_void,
        type_meta: TypeMeta,
        name:      &String,
        acceptor:  SerializationAcceptor)  
    {
        
        todo!();
        /*
            CAFFE_ENFORCE(typeMeta.Match<std::unique_ptr<IndexBase>>());
        const auto& base = *static_cast<const std::unique_ptr<IndexBase>*>(pointer);
        Blob tensor_blob;
        auto* tensor_out = BlobGetMutableTensor(&tensor_blob, CPU);

        if (base->Type().Match<std::string>()) {
          doStore<std::string>(base, tensor_out);
        } else if (base->Type().Match<int32_t>()) {
          doStore<int32_t>(base, tensor_out);
        } else if (base->Type().Match<int64_t>()) {
          doStore<int64_t>(base, tensor_out);
        } else {
          CAFFE_THROW("Index of this type can't be serialized.");
        }

        CAFFE_ENFORCE(
            tensor_out->numel() <= int32_t::max,
            "Index too large to be serialized.");
        BlobProto blob_proto;
        TensorSerializer ser;
        ser.Serialize(
            *tensor_out, name, blob_proto.mutable_tensor(), 0, tensor_out->numel());
        blob_proto.set_name(name);
        blob_proto.set_type("std::unique_ptr<caffe2::IndexBase>");

        std::ostringstream os;
        os << base->maxElements() << " " << base->isFrozen();
        blob_proto.set_content(os.str());

        acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
        */
    }

    #[inline] pub fn do_store<T>(
        &mut self,
        base: &Box<IndexBase>,
        tensor_out: *mut Tensor)
    {
        todo!();
        /*
            auto* dict = dynamic_cast_if_rtti<Index<T>*>(base.get());
            CAFFE_ENFORCE(dict, "Wrong dictionary type.");
            dict->Store(tensor_out);
        */
    }
}
