crate::ix!();

/**
  | The Blob serialization registry and
  | serializer creator functions.
  |
  */
declare_typed_registry!{
    /*
    BlobSerializerRegistry,
    TypeIdentifier,
    BlobSerializerBase,
    Box
    */
}

declare_registry!{
    BlobDeserializerRegistry, 
    BlobDeserializerBase
}

// The actual serialization registry objects.
define_typed_registry!{
    /*
    BlobSerializerRegistry,
    TypeIdentifier,
    BlobSerializerBase,
    Box
    */
}

define_registry!{
    /*
    BlobDeserializerRegistry, 
    BlobDeserializerBase
    */
}

register_blob_serializer!{
    /*
    (TypeMeta::Id<Tensor>()), 
    TensorSerializer
    */
}

register_blob_deserializer!{
    /*
    TensorCPU, 
    TensorDeserializer
    */
}

register_blob_serializer!{
    /*
    (TypeMeta::Id<std::string>()), 
    StringSerializer
    */
}

register_blob_deserializer!{
    /*
    String, 
    StringDeserializer
    */
}

define_registry!{
    /*
    NetRegistry,
    NetBase,
    const std::shared_ptr<const NetDef>&,
    Workspace*
    */
}
