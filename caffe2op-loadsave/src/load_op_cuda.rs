crate::ix!();

impl LoadOp<CUDAContext> {

    #[inline] pub fn set_current_device(&mut self, proto: *mut BlobProto)  {
        
        todo!();
        /*
            if (proto->has_tensor()) {
        proto->mutable_tensor()->clear_device_detail();
        auto* device_detail = proto->mutable_tensor()->mutable_device_detail();
        device_detail->set_device_type(PROTO_CUDA);
        device_detail->set_device_id(CaffeCudaGetDevice());
      }
        */
    }
}
