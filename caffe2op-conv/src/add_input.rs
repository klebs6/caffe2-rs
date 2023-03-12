crate::ix!();

#[inline] pub fn add_const_input(
    shape:   &Vec<i64>,
    value:   f32,
    name:    &String,
    ws:      *mut Workspace)  
{
    todo!();
    /*
        DeviceOption option;
      CPUContext context(option);
      Blob* blob = ws->CreateBlob(name);
      auto* tensor = BlobGetMutableTensor(blob, CPU);
      tensor->Resize(shape);
      math::Set<float, CPUContext>(
          tensor->numel(), value, tensor->template mutable_data<float>(), &context);
    */
}

#[inline] pub fn add_noise_input(
    shape: &Vec<i64>,
    name: &String,
    ws: *mut Workspace)  
{
    todo!();
    /*
        DeviceOption option;
      CPUContext context(option);
      Blob* blob = ws->CreateBlob(name);
      auto* tensor = BlobGetMutableTensor(blob, CPU);
      tensor->Resize(shape);

      math::RandGaussian<float, CPUContext>(
          tensor->numel(),
          0.0f,
          10.0f,
          tensor->template mutable_data<float>(),
          &context);
    */
}
