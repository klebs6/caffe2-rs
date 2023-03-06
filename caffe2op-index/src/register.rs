crate::ix!();

pub type IndexKeyTypes = TensorTypes<(i32, i64, String)>;

register_cpu_operator!{IntIndexCreate, IndexCreateOp<int32_t>}
register_cpu_operator!{LongIndexCreate, IndexCreateOp<int64_t>}
register_cpu_operator!{StringIndexCreate, IndexCreateOp<std::string>}

register_cpu_operator!{IndexGet, IndexGetOp}
register_cpu_operator!{IndexLoad, IndexLoadOp}
register_cpu_operator!{IndexStore, IndexStoreOp}
register_cpu_operator!{IndexFreeze, IndexFreezeOp}
register_cpu_operator!{IndexSize, IndexSizeOp}

no_gradient!{IndexGetOp}
no_gradient!{IntIndexCreate}
no_gradient!{LongIndexCreate}
no_gradient!{StringIndexCreate}

should_not_do_gradient!{IndexFreeze}
should_not_do_gradient!{IndexLoad}
should_not_do_gradient!{IndexStore}
should_not_do_gradient!{IndexSize}

caffe_known_type!{Box<IndexBase>}

register_blob_serializer!{ 
    /*
    (TypeMeta::Id<std::unique_ptr<caffe2::IndexBase>>()), 
    IndexSerializer
    */
}

register_blob_deserializer!{ 
    /*
    std::unique_ptr<caffe2::IndexBase>, 
    IndexDeserializer
    */
}
