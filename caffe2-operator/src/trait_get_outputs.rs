crate::ix!();

pub trait GetOutputs {

    #[inline] fn outputs(&mut self) -> &Vec<*mut Blob> {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            isLegacyOperator(),
            "Outputs() not supported for operators exported to c10.");
        return outputs_;
        */
    }
}

pub trait GetXOutput {

    /**
      | XOutput is a modernized version of Output
      | which returns a Tensor rather than
      | a Tensor* (the raw pointer in the latter
      | case is useless, as Tensor is a pointer
      | type.)
      */
    #[inline] fn x_output(
        &mut self, 
        idx: i32,
        dims: &[i32],
        options: TensorOptions) -> Tensor 
    {
        todo!();
        /*
            // We'll default device to the device of the current Operator Context
        if (options.device_opt() == c10::nullopt) {
          return OperatorStorage::XOutputTensor(
              idx, dims, options.device(context_.device()));
        }
        return OperatorStorage::XOutputTensor(idx, dims, options);
        */
    }

    #[inline] fn xOutput_tensor(
        &mut self, 
        idx: i32,
        dims: &[i32],
        options: TensorOptions) -> Tensor 
    {
        todo!();
        /*
            CAFFE_ENFORCE_WITH_CALLER(
            options.device_opt() != c10::nullopt,
            "device must be provided in option.");
        if (isLegacyOperator()) {
          return XBlobGetMutableTensor(outputs_.at(idx), dims, options);
        }

        return OutputTensor(idx, dims, options)->UnsafeSharedInstance();
        */
    }
}

pub trait GetOutputAtIndex_Legacy {

    /**
      | Legacy: please consider using the version
      | of Output() which also takes dtype and
      | size as arguments.
      |
      */
    #[inline] fn output_with_idx_and_device_type(
        &mut self, 
        idx: i32,
        ty: Option<DeviceType>) -> *mut Tensor 
    {
        let ty = todo!(); //ty.unwrap_or(Context::GetDeviceType());
        
        todo!();
        /*
            return OperatorStorage::template Output<Tensor>(idx, type);
        */
    }
}

pub trait GetOutputAtIndexTensorCopy {

    /**
      | Get the output Tensor of an operator
      | (allocating it if it is not already
      | initialized), and copy the contents of src
      | into it.
      |
      | You probably don't actually want to use
      | this function (the fact that you have
      | a Tensor to copy from is probably
      | a mistake: you should have written the
      | output into the output tensor, from
      | Output, directly in the first place), but
      | this method is situationally useful.
      */
    #[inline] fn output_tensor_copy_from(
        &mut self, 
        idx:     i32,
        options: TensorOptions,
        src:     &Tensor,
        async_:  Option<bool>) -> *mut Tensor 
    {
        let async_: bool = async_.unwrap_or(false);

        todo!();
        /*
            if (options.device_opt() == c10::nullopt) {
          return OperatorStorage::OutputTensorCopyFrom(
              idx, options.device(context_.device()), src, async);
        }
        return OperatorStorage::OutputTensorCopyFrom(idx, options, src, async);
        */
    }
}
