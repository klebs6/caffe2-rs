crate::ix!();

pub struct IndexDeserializer {
    base: dyn BlobDeserializerBase,
}

impl IndexDeserializer {
    
    #[inline] pub fn deserialize(
        &mut self, 
        proto: &BlobProto,
        blob: *mut Blob)  
    {
        todo!();
        /*
            TensorDeserializer deser;
        Blob tensor_blob;
        deser.Deserialize(proto, &tensor_blob);

        std::istringstream is(proto.content());
        int64_t maxElements{int64_t::max};
        bool isFrozen{false};
        is >> maxElements >> isFrozen;

        auto& tensor_in = tensor_blob.template Get<Tensor>();
        auto* base = blob->template GetMutable<std::unique_ptr<IndexBase>>();

        if (tensor_in.IsType<std::string>()) {
          doLoad<std::string>(base, maxElements, tensor_in);
        } else if (tensor_in.IsType<int32_t>()) {
          doLoad<int32_t>(base, maxElements, tensor_in);
        } else if (tensor_in.IsType<int64_t>()) {
          doLoad<int64_t>(base, maxElements, tensor_in);
        } else {
          CAFFE_THROW("Index of this type cannot be deserialized.");
        }

        if (isFrozen) {
          (*base)->Freeze();
        }
        */
    }

    #[inline] pub fn do_load<T>(
        &mut self,
        base: *mut Box<IndexBase>,
        max_elements: i64,
        tensor_in: &Tensor) {
        todo!();
        /*
            base->reset(new Index<T>(maxElements));
            auto* dict = dynamic_cast_if_rtti<Index<T>*>(base->get());
            dict->Load(tensor_in.data<T>(), tensor_in.numel());
        */
    }
}
