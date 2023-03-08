crate::ix!();

#[inline] pub fn update_timestep_blob(
    ws:        *mut Workspace,
    blob_name: String,
    t:         i32)  
{
    todo!();
    /*
        BlobGetMutableTensor(ws->CreateBlob(blob_name), CPU)->Resize(1);
      auto timestepBlob = ws->GetBlob(blob_name);
      CAFFE_ENFORCE(timestepBlob);
      BlobGetMutableTensor(timestepBlob, CPU)->template mutable_data<int32_t>()[0] =
          t;
    */
}
