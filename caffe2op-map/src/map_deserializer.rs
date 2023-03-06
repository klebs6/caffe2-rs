crate::ix!();

pub struct MapDeserializer<KEY_T,VALUE_T> {
    //using MapType = typename MapTypeTraits<KEY_T, VALUE_T>::MapType;
    phantomKEY_T: PhantomData<KEY_T>,
    phantomVALUE_T: PhantomData<VALUE_T>,
}

impl<K,V> BlobDeserializerBase for MapDeserializer<K,V> {

    #[inline] fn deserialize(&mut self, proto: &BlobProto, blob: *mut Blob)  {
        
        todo!();
        /*
            TensorProtos tensor_protos;
                CAFFE_ENFORCE(
                    tensor_protos.ParseFromString(proto.content()),
                    "Fail to parse TensorProtos");
                TensorDeserializer deser;
                Tensor key_tensor = deser.Deserialize(tensor_protos.protos(0));
                Tensor value_tensor = deser.Deserialize(tensor_protos.protos(1));
                auto* key_data = key_tensor.data<KEY_T>();
                auto* value_data = value_tensor.data<VALUE_T>();

                auto* map_ptr = blob->template GetMutable<MapType>();
                for (int i = 0; i < key_tensor.numel(); ++i) {
                    map_ptr->emplace(key_data[i], value_data[i]);
                }
        */
    }
}
