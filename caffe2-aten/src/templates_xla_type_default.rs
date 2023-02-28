crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/templates/aten_xla_type_default.h]

/**
  | TODO: maybe kill this, doesn't look
  | like XLA actually calls it anywhere
  |
  */
pub fn register_aten_type_functions()  {
    
    todo!();
        /*
        
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/templates/aten_xla_type_default.cpp]

/**
  | convenience helpers for extracting
  | out an optional Device
  |
  */
pub trait GetDeviceArg {

    fn get_device_arg(&self) -> Option<Device>;
}

impl GetDeviceArg for Tensor {

    fn get_device_arg(&self) -> Option<Device> {
        
        todo!();
            /*
                return self.device();
            */
    }
}

impl GetDeviceArg for Option<Tensor> {

    // self == maybe_tensor
    fn get_device_arg(&self) -> Option<Device> {
        
        todo!();
            /*
                return maybe_tensor ? optional<Device>(maybe_tensor.unwrap().device()) : nullopt;
            */
    }
}

impl GetDeviceArg for Vec<Tensor> {

    //self == tensors
    fn get_device_arg(&self) -> Option<Device> {
        
        todo!();
            /*
                return tensors.size() > 0 ? optional<Device>(tensors[0].device()) : nullopt;
            */
    }
}

impl GetDeviceArg for TensorList {

    // self == tensors: TensorList
    fn get_device_arg(&self) -> Option<Device> {
        
        todo!();
            /*
                return tensors.size() > 0 ? optional<Device>(tensors[0].device()) : nullopt;
            */
    }
}

impl GetDeviceArg for Option<Device> {

    // self == device: Option<Device>
    fn get_device_arg(&self) -> Option<Device> {
        
        todo!();
            /*
                return device;
            */
    }
}

impl GetDeviceArg for Device {

    // self == device: Device
    fn get_device_arg(&self) -> Option<Device> {
        
        todo!();
            /*
                return optional<Device>(device);
            */
    }
}

pub trait ToDeviceOpt {

    type Output;

    fn to_device_opt(&self, device: Option<Device>) -> Self::Output;
}

impl ToDeviceOpt for Tensor {

    type Output = Tensor;

    /**
      | convenience helpers for converting
      | tensors to an optional device
      |
      */
    fn to_device_opt(&self, device: Option<Device>) -> Self::Output {
        
        todo!();
            /*
                return device ? tensor.to(*device) : tensor;
            */
    }
}

impl ToDeviceOpt for Vec<Tensor> {

    type Output = Vec<Tensor>;

    // self == tensors
    fn to_device_opt(&self, device:  Option<Device>) -> Self::Output {
        
        todo!();
            /*
                vector<Tensor> output_tensors;
            for (const auto& t : tensors) {
                output_tensors.push_back(to_device_opt(t, device));
            }
            return output_tensors;
            */
    }
}

pub trait ToCpu {

    type Output;

    fn to_cpu(&self) -> Self::Output;
}

impl ToCpu for TensorList {

    type Output = Vec<Tensor>;

    /**
      | convenience helper for converting
      | tensors to cpu
      |
      */
    //self == tensors: &TensorList
    fn to_cpu(&self) -> Self::Output {
        
        todo!();
            /*
                // We can't just call to_cpu() on the entire list of Tensors
            // Because it will break on undefined tensors. Separate out undefined tensors first.
            vector<Tensor> cpu_tensors(tensors.size());
            vector<Tensor> valid_tensors;
            vector<bool> to_translate(tensors.size());
            for (usize i = 0; i < tensors.size(); ++i) {
                const Tensor& tensor = tensors[i];
                if (tensor.defined()) {
                    to_translate[i] = true;
                    valid_tensors.push_back(tensor);
                } else {
                    cpu_tensors[i] = tensor;
                }
            }
            auto cpu_valid_tensors = _to_cpu(valid_tensors);
            for (usize i = 0, defined_pos = 0; i < tensors.size(); ++i) {
                if (to_translate[i]) {
                    cpu_tensors[i] = move(cpu_valid_tensors[defined_pos++]);
                }
            }
          return cpu_tensors;
            */
    }
}

impl ToCpu for Vec<Option<Tensor>> {

    type Output = Vec<Option<Tensor>>;

    // self == tensors: &Vec<Option<Tensor>>
    //
    fn to_cpu(&self) -> Self::Output {
        
        todo!();
            /*
                vector<optional<Tensor>> opt_tensors(tensors.size());
            vector<Tensor> materialized_tensors;
            vector<bool> to_translate(tensors.size());
            for (usize i = 0; i < tensors.size(); ++i) {
                auto tensor = tensors[i];
                if (tensor.has_value()) {
                    to_translate[i] = true;
                    materialized_tensors.push_back(*tensor);
                }
            }
            auto aten_materialized_tensors = to_cpu(materialized_tensors);
            for (usize i = 0, defined_pos = 0; i < tensors.size(); ++i) {
                if (to_translate[i]) {
                  opt_tensors[i] =
                  move(aten_materialized_tensors[defined_pos++]);
                }
            }
            return opt_tensors;
            */
    }
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(aten, XLA, m) {
        ${dispatch_registrations}
    }
    */
}
