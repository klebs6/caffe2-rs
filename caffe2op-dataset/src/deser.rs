crate::ix!();

pub struct SharedTensorVectorPtrSerializer {
    base: dyn BlobSerializerBase,
}

impl SharedTensorVectorPtrSerializer {
    
    #[inline] pub fn serialize(
        &mut self, 
        pointer:   *const c_void,
        type_meta: TypeMeta,
        name:      &String,
        acceptor:  SerializationAcceptor)  
    {
        todo!();
        /*
          /* This is dummy serialize that doesn't save anything. If saving the content
          is desired in future use case, you can change this serializer. Note: special
          care need to be taken for the parameter initialization of
          LastNWindowCollectorOp and ReservoirSamplingOp if this serializer actually
          saves the content.
          */
          CAFFE_ENFORCE(typeMeta.Match<std::shared_ptr<std::vector<TensorCPU>>>());
          BlobProto blob_proto;
          blob_proto.set_name(name);
          blob_proto.set_type("std::shared_ptr<std::vector<TensorCPU>>");
          blob_proto.set_content("");
          acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
        */
    }
}

///---------------------------------------
pub struct SharedTensorVectorPtrDeserializer {
    base: dyn BlobDeserializerBase,
}

impl SharedTensorVectorPtrDeserializer {
    
    #[inline] pub fn deserialize(
        &mut self, 
        unused: &BlobProto,
        blob: *mut Blob)  
    {
        todo!();
        /*
            /* This is dummy deserialize which creates a nullptr
       */
      blob->GetMutable<std::shared_ptr<std::vector<TensorCPU>>>();
        */
    }
}

pub struct TreeCursorSerializer {
    base: dyn BlobSerializerBase,
}

impl TreeCursorSerializer {
    
    #[inline] pub fn serialize(
        &mut self, 
        pointer:   *const c_void,
        type_meta: TypeMeta,
        name:      &String,
        acceptor:  SerializationAcceptor)  
    {

        todo!();
        /*
            CAFFE_ENFORCE(typeMeta.Match<std::unique_ptr<TreeCursor>>());
        const auto& cursor =
            *static_cast<const std::unique_ptr<TreeCursor>*>(pointer);
        BlobProto blob_proto;

        // serialize offsets as a tensor
        if (cursor->offsets.size() > 0) {
          Blob offsets_blob;
          auto* offsets = BlobGetMutableTensor(&offsets_blob, CPU);
          offsets->Resize(cursor->offsets.size());
          std::copy(
              cursor->offsets.begin(),
              cursor->offsets.end(),
              offsets->template mutable_data<TOffset>());
          TensorSerializer ser;
          ser.Serialize(
              *offsets, name, blob_proto.mutable_tensor(), 0, offsets->numel());
        }
        blob_proto.set_name(name);
        blob_proto.set_type("std::unique_ptr<TreeCursor>");

        // serialize field names in the content
        std::ostringstream os;
        for (const auto& field : cursor->it.fields()) {
          os << field.name << " ";
        }
        blob_proto.set_content(os.str());

        acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
        */
    }
}

pub struct TreeCursorDeserializer {
    base: dyn BlobDeserializerBase,
}

impl TreeCursorDeserializer {
    
    #[inline] pub fn deserialize(&mut self, proto: &BlobProto, blob: *mut Blob)  {
        
        todo!();
        /*
            // Deserialize the field names
        std::vector<std::string> fieldNames;
        std::istringstream is(proto.content());
        std::string field;
        while (true) {
          is >> field;
          if (is.eof()) {
            break;
          }
          fieldNames.push_back(field);
        }
        TreeIterator it(fieldNames);

        auto* base = blob->template GetMutable<std::unique_ptr<TreeCursor>>();
        CAFFE_ENFORCE(base != nullptr, "TreeCursor doesn't exist.");
        (*base).reset(new TreeCursor(it));

        // Deserialize the offset vector when it is not empty. The proto.tensor()
        // function will return a TensorProto associated with offset vector. The
        // offset vector contains fields of type int64_t, and we verify it is not
        // empty before calling the deserializer.
        if (proto.tensor().int64_data().size() > 0) {
          TensorDeserializer deser;
          Blob offset_blob;
          deser.Deserialize(proto, &offset_blob);
          auto& offsets = offset_blob.template Get<Tensor>();
          auto* offsets_ptr = offsets.data<TOffset>();
          (*base)->offsets.assign(offsets_ptr, offsets_ptr + offsets.numel());
        }
        */
    }
}

