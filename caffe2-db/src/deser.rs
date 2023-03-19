crate::ix!();

pub struct DBReaderSerializer {
    base: dyn BlobSerializerBase,
}

pub struct DBReaderDeserializer {
    base: dyn BlobDeserializerBase,
}

caffe_known_type![db::DBReader];
caffe_known_type![db::Cursor];

impl BlobSerializerBase for DBReaderSerializer {
    
    /**
      | Serializes a DBReader. Note that this
      | blob has to contain DBReader, otherwise
      | this function produces a fatal error.
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
            CAFFE_ENFORCE(typeMeta.Match<DBReader>());
      const auto& reader = *static_cast<const DBReader*>(pointer);
      DBReaderProto proto;
      proto.set_name(name);
      proto.set_source(reader.source_);
      proto.set_db_type(reader.db_type_);
      if (reader.cursor() && reader.cursor()->SupportsSeek()) {
        proto.set_key(reader.cursor()->key());
      }
      BlobProto blob_proto;
      blob_proto.set_name(name);
      blob_proto.set_type("DBReader");
      blob_proto.set_content(SerializeAsString_EnforceCheck(proto));
      acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
        */
    }
}

impl DBReaderDeserializer {
    
    #[inline] pub fn deserialize(
        &mut self, 
        proto: &BlobProto, 
        blob: *mut Blob)  
    {
        todo!();
        /*
            DBReaderProto reader_proto;
      CAFFE_ENFORCE(
          reader_proto.ParseFromString(proto.content()),
          "Cannot parse content into a DBReaderProto.");
      blob->Reset(new DBReader(reader_proto));
        */
    }
}

// Serialize TensorCPU.
register_blob_serializer![
    /*
    (TypeMeta::Id<DBReader>()), 
    DBReaderSerializer
    */
];

register_blob_deserializer![
    /*
    DBReader, 
    DBReaderDeserializer
    */
];
