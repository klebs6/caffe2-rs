crate::ix!();

pub mod cpu {

    #[inline] pub fn add_const_input(
        shape: &Vec<i64>,
        value: f32,
        name:  &String,
        ws:    *mut Workspace)  
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
          return;
        */
    }

    #[inline] pub fn add_lin_spaced_input(
        shape:   &Vec<i64>,
        min_val: f32,
        max_val: f32,
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
          EigenVectorMap<float> tensor_vec(
              tensor->template mutable_data<float>(), tensor->numel());
          tensor_vec.setLinSpaced(min_val, max_val);

          return;
        */
    }

    #[inline] pub fn add_input(
        shape:  &Vec<i64>,
        values: &Vec<f32>,
        name:   &String,
        ws:     *mut Workspace)  
    {

        todo!();
        /*
            DeviceOption option;
          CPUContext context(option);
          Blob* blob = ws->CreateBlob(name);
          auto* tensor = BlobGetMutableTensor(blob, CPU);
          tensor->Resize(shape);
          EigenVectorMap<float> tensor_vec(
              tensor->template mutable_data<float>(), tensor->numel());
          tensor_vec.array() = utils::AsEArrXt(values);

          return;
        */
    }
}
