crate::ix!();

pub struct MapSerializer<KEY_T,VALUE_T> {

    /**
      | using MapType = typename MapTypeTraits<KEY_T,
      | VALUE_T>::MapType;
      |
      */
    phantomKEY_T: PhantomData<KEY_T>,
    phantomVALUE_T: PhantomData<VALUE_T>,
}

impl<K,V> BlobSerializerBase for MapSerializer<K,V> {

    #[inline] fn serialize(&mut self, 
        pointer:   *const libc::c_void,
        type_meta: TypeMeta,
        name:      &String,
        acceptor:  SerializationAcceptor,
        options:   Option<&BlobSerializationOptions>)  {
        
        todo!();
        /*
            CAFFE_ENFORCE(typeMeta.Match<MapType>());
                const MapType& map_data = *static_cast<const MapType*>(pointer);
                int64_t sz = map_data.size();
                Tensor key_tensor(CPU);
                key_tensor.Resize(sz);
                Tensor value_tensor(CPU);
                value_tensor.Resize(sz);
                auto* key_data = key_tensor.mutable_data<KEY_T>();
                auto* value_data = value_tensor.mutable_data<VALUE_T>();
                for (const auto& it : map_data) {
                    *key_data = it.first;
                    *value_data = it.second;
                    key_data++;
                    value_data++;
                }

                TensorProtos tensor_protos;
                TensorSerializer ser;
                ser.Serialize(
                    key_tensor, name, tensor_protos.add_protos(), 0, key_tensor.numel());
                ser.Serialize(
                    value_tensor,
                    name,
                    tensor_protos.add_protos(),
                    0,
                    value_tensor.numel());

                BlobProto blob_proto;
                blob_proto.set_name(name);
                blob_proto.set_type(MapTypeTraits<KEY_T, VALUE_T>::MapTypeName());
                blob_proto.set_content(SerializeAsString_EnforceCheck(tensor_protos));
                acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
        */
    }
}
