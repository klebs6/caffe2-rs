crate::ix!();

const kQTensorBlobQType: &'static str = "QTensor";

pub struct QTensorSerializer<Context> {
    context: Context,
}

impl<Context> Default for QTensorSerializer<Context> {
    
    fn default() -> Self {
        todo!();
        /*
            : context_(
        */
    }
}

impl<Context> BlobSerializerBase for QTensorSerializer<Context> {
    
    /**
      | Serializes a Blob. Note that this blob
      | has to contain QTensor<Context>.
      |
      */
    #[inline] fn serialize(&mut self, 
        pointer:   *const c_void,
        type_meta: TypeMeta,
        name:      &String,
        acceptor:  SerializationAcceptor,
        options:    Option<&BlobSerializationOptions>) {

        todo!();
        /*
          CAFFE_ENFORCE(typeMeta.Match<QTensor<Context>>());
          const auto& qtensor = *static_cast<const QTensor<Context>*>(pointer);
          BlobProto blob_proto;
          blob_proto.set_name(name);
          blob_proto.set_type(kQTensorBlobQType);
          QTensorProto& proto = *blob_proto.mutable_qtensor();
          proto.set_name(name);
          for (int i = 0; i < qtensor.ndim(); ++i) {
            proto.add_dims(qtensor.dim32(i));
          }
          proto.set_precision(qtensor.precision());
          proto.set_scale(qtensor.scale());
          proto.set_bias(qtensor.bias());
          proto.set_is_signed(qtensor.is_signed());
          detail::CopyToProtoWithCast(
              qtensor.nbytes(), qtensor.data(), proto.mutable_data(), &this->context_);
          acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
       
        */
    }
}

///-----------------------------
pub struct QTensorDeserializer<Context> {
    phantom: PhantomData<Context>,
}

impl<Context> QTensorDeserializer<Context> {
    
    #[inline] pub fn deserialize_from_blob(&mut self, blob_proto: &BlobProto, blob: *mut Blob)  {
        
        todo!();
        /*
            Deserialize(blob_proto.qtensor(), blob->GetMutable<QTensor<Context>>());
        */
    }

}

impl<Context> QTensorDeserializer<Context> {
    
    #[inline] fn deserialize(&mut self, 
        proto: &QTensorProto, 
        qtensor: *mut QTensor<Context>)  {
        
        todo!();
        /*
            Context context{};
      vector<int> dims;
      for (const int d : proto.dims()) {
        dims.push_back(d);
      }
      qtensor->Resize(dims);
      qtensor->SetPrecision(proto.precision());
      qtensor->SetScale(proto.scale());
      qtensor->SetBias(proto.bias());
      qtensor->SetSigned(proto.is_signed());

      detail::CopyFromProtoWithCast(
          qtensor->nbytes(), proto.data(), qtensor->mutable_data(), &context);
        */
    }
}

register_blob_serializer!{
    /*
    (TypeMeta::Id<QTensor<CPUContext>>()),
    QTensorSerializer<CPUContext>
    */
}

register_blob_deserializer!{
    /*
    QTensor, 
    QTensorDeserializer<CPUContext>
    */
}
