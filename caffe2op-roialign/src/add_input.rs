crate::ix!();

#[inline] pub fn add<T>(val: &T, address: *mut T) {
    todo!();
    /*
        *address += val;
    */
}

#[inline] pub fn rand_int(a: i32, b: i32) -> i32 {
    
    todo!();
    /*
        static std::random_device rd;
      static std::mt19937 gen(rd());
      return std::uniform_int_distribution<int>(a, b)(gen);
    */
}

#[inline] pub fn add_const_input<Context>(
    shape:     &Vec<i64>,
    value:     f32,
    name:      &String,
    context:   *mut Context,
    ws:        *mut Workspace) 
{
    todo!();
    /*
        Blob* blob = ws->CreateBlob(name);
      auto* tensor = BlobGetMutableTensor(blob, Context::GetDeviceType());
      tensor->Resize(shape);
      math::Set<float, Context>(
          tensor->size(), value, tensor->template mutable_data<float>(), context);
      return;
    */
}

pub fn add_input(
    shape:  &Vec<i64>,
    values: &Vec<f32>,
    name:   &String,
    ws:     *mut Workspace) 
{
    todo!("dispatch");
}

#[inline] pub fn add_input_cpu_context(
    shape:  &Vec<i64>,
    values: &Vec<f32>,
    name:   &String,
    ws:     *mut Workspace)  
{
    todo!();
    /*
        Blob* blob = ws->CreateBlob(name);
      auto* tensor = BlobGetMutableTensor(blob, CPU);
      tensor->Resize(shape);
      EigenVectorMap<float> tensor_vec(
          tensor->template mutable_data<float>(), tensor->numel());
      tensor_vec.array() = utils::AsEArrXt(values);
    */
}

#[inline] pub fn add_input_cudacontext(
    shape:  &Vec<i64>,
    values: &Vec<f32>,
    name:   &String,
    ws:     *mut Workspace)  
{
    todo!();
    /*
        Tensor tmp(shape, CPU);
      EigenVectorMap<float> tmp_vec(tmp.mutable_data<float>(), tmp.numel());
      tmp_vec.array() = utils::AsEArrXt(values);

      Blob* blob = ws->CreateBlob(name);
      auto* tensor = BlobGetMutableTensor(blob, CUDA);
      tensor->CopyFrom(tmp);
    */
}
