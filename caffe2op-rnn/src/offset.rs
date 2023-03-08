crate::ix!();

#[inline] pub fn apply_offset_alias<T, Context>(
    oc:      &OffsetAlias,
    ws:      *mut Workspace,
    context: *mut Context) 
{
    todo!();
    /*
        VLOG(1) << "Aliasing: " << oc.src << " to: " << oc.dst
              << " at offset: " << oc.offset;
      auto srcBlob = ws->GetBlob(oc.src);
      CAFFE_ENFORCE(srcBlob);
      auto* src = BlobGetMutableTensor(srcBlob, Context::GetDeviceType());
      auto* dst =
          BlobGetMutableTensor(ws->GetBlob(oc.dst), Context::GetDeviceType());
      auto timestep = src->numel() / src->size(0);
      auto dims = src->sizes().vec();
      const int32_t startDstTimestep =
          oc.offset >= 0 ? oc.offset : src->size(0) + oc.offset;
      const int32_t numDstTimesteps = src->size(0) - startDstTimestep;
      if (numDstTimesteps >= 1) {
        dims[0] = numDstTimesteps;
        dst->Resize(dims);
        CAFFE_ENFORCE(timestep == dst->numel() / numDstTimesteps, "Invalid offset");
        dst->ShareExternalPointer(
            src->template mutable_data<T>() + startDstTimestep * timestep);
      } else {
        CAFFE_ENFORCE_EQ(
            numDstTimesteps, 0, "Invalid number of timesteps: ", numDstTimesteps);
        dims[0] = 0;
        dst->Resize(dims);
        dst->template mutable_data<T>();
      }
    */
}
