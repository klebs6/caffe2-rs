/*!
  | TODO: Remove the condition on AT_ROCM_ENABLED
  | entirely, don't build this file as part
  | of CPU build.
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/miopen/BatchNorm_miopen.cpp]

// See Note [ATen preprocessor philosophy]
#[cfg(not(AT_ROCM_ENABLED))]
pub fn miopen_batch_norm(
        input:                      &Tensor,
        weight:                     &Tensor,
        bias_opt:                   &Option<Tensor>,
        running_mean_opt:           &Option<Tensor>,
        running_var_opt:            &Option<Tensor>,
        training:                   bool,
        exponential_average_factor: f64,
        epsilon:                    f64) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            AT_ERROR("miopen_batch_norm: ATen not compiled with MIOpen support");
        */
}

#[cfg(not(AT_ROCM_ENABLED))]
pub fn miopen_batch_norm_backward(
        input:            &Tensor,
        grad_output:      &Tensor,
        weight:           &Tensor,
        running_mean_opt: &Option<Tensor>,
        running_var_opt:  &Option<Tensor>,
        save_mean_opt:    &Option<Tensor>,
        save_var_opt:     &Option<Tensor>,
        epsilon:          f64) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            AT_ERROR("miopen_batch_norm_backward: ATen not compiled with MIOpen support");
        */
}

#[cfg(AT_ROCM_ENABLED)]
pub fn expand_scale(
        t:   &Tensor,
        dim: i64) -> Tensor {
    
    todo!();
        /*
            vector<i64> size{ 1, t.numel() };
      while (static_cast<i64>(size.size()) < dim) {
        size.emplace_back(1);
      }
      return t.view(size);
        */
}

#[cfg(AT_ROCM_ENABLED)]
pub fn miopen_batch_norm(
        input_t:                    &Tensor,
        weight_t:                   &Tensor,
        bias_t_opt:                 &Option<Tensor>,
        running_mean_t_opt:         &Option<Tensor>,
        running_var_t_opt:          &Option<Tensor>,
        training:                   bool,
        exponential_average_factor: f64,
        epsilon:                    f64) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_t_maybe_owned = borrow_from_optional_tensor(bias_t_opt);
      const Tensor& bias_t = *bias_t_maybe_owned;
      const Tensor& running_mean_t = value_or_else(running_mean_t_opt, [] {return Tensor();});
      const Tensor& running_var_t = value_or_else(running_var_t_opt, [] {return Tensor();});

      TensorArg input{ input_t, "input", 1 },
                weight{ weight_t, "weight", 2 },
                bias{ bias_t, "bias", 3 },
                running_mean{ running_mean_t, "running_mean", 4 },
                running_var{ running_var_t, "running_var", 5 };
      CheckedFrom c = "miopen_batch_norm";

      checkAllDefined(c, {input, weight, bias});
      if (!training) {
        checkAllDefined(c, {running_mean, running_var});
      }
      checkAllSameGPU(c, {input, weight, bias, running_mean, running_var});
      if (input->scalar_type() != ScalarType::Half) {
        checkAllSameType(c, {input, weight});
      }
      checkAllSameType(c, {weight, bias, running_mean, running_var});
      checkAllContiguous(c, {input, weight, bias, running_mean, running_var});
      checkDimRange(c, input, 2, 6 /* exclusive */);
      auto num_features = input->size(1);
      for (auto t : {weight, bias, running_mean, running_var}) {
        if (t->defined()) {
          checkNumel(c, t, num_features);
        }
      }

      miopenBatchNormMode_t mode;
      if (input->dim() == 2) {
        mode = miopenBNPerActivation;
      } else {
        mode = miopenBNSpatial;
      }

      auto output_t = empty(input->sizes(), input->options());
      TensorArg output{ output_t, "output", 0 };

      auto handle = getMiopenHandle();
      auto dataType = getMiopenDataType(*input);
      TensorDescriptor idesc{ *input, 4 };  // input descriptor
      TensorDescriptor wdesc{ expandScale(*weight, input->dim()), 4 };  // descriptor for weight, bias, running_mean, etc.

      Constant one(dataType, 1);
      Constant zero(dataType, 0);
      Tensor save_mean, save_var;

      if (training) {
        i64 num_features = input_t.size(1);
        save_mean = empty({ num_features }, weight_t.options());
        save_var = empty({ num_features }, weight_t.options());
        MIOPEN_CHECK(miopenBatchNormalizationForwardTraining(
          handle, mode, &one, &zero,
          idesc.desc(), input->data_ptr(),
          idesc.desc(), output->data_ptr(),
          wdesc.desc(),
          weight->data_ptr(),
          bias->data_ptr(),
          exponential_average_factor,
          maybe_data_ptr(running_mean),
          maybe_data_ptr(running_var),
          epsilon,
          save_mean.data_ptr(),
          save_var.data_ptr()));
      } else {
        MIOPEN_CHECK(miopenBatchNormalizationForwardInference(
          handle, mode, &one, &zero,
          idesc.desc(), input->data_ptr(),
          idesc.desc(), output->data_ptr(),
          wdesc.desc(),
          weight->data_ptr(),
          bias->data_ptr(),
          running_mean->data_ptr(),
          running_var->data_ptr(),
          epsilon));
      }

      // save_mean and save_var can be undefined
      // If this causes problems, we can initialize them to empty tensors
      // of the correct type
      return tuple<Tensor, Tensor, Tensor>{output_t, save_mean, save_var};
        */
}

#[cfg(AT_ROCM_ENABLED)]
pub fn miopen_batch_norm_backward(
        input_t:          &Tensor,
        grad_output_t:    &Tensor,
        weight_t:         &Tensor,

        // Unused: but we require them to be passed so that double backwards
        // has access
        running_mean_opt: &Option<Tensor>,
        running_var_opt:  &Option<Tensor>,
        save_mean_t_opt:  &Option<Tensor>,
        save_var_t_opt:   &Option<Tensor>,
        epsilon:          f64) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      const Tensor& running_mean =
          value_or_else(running_mean_opt, [] { return Tensor(); });
      const Tensor& running_var =
          value_or_else(running_var_opt, [] { return Tensor(); });
      const Tensor& save_mean_t =
          value_or_else(save_mean_t_opt, [] { return Tensor(); });
      const Tensor& save_var_t =
          value_or_else(save_var_t_opt, [] { return Tensor(); });

      TensorArg input{ input_t, "input", 1 },
                grad_output{ grad_output_t, "grad_output", 2 },
                weight{ weight_t, "weight", 3 },
                save_mean{ save_mean_t, "save_mean", 4 },
                save_var{ save_var_t, "save_var", 5 };
      CheckedFrom c = "miopen_batch_norm_backward";

      checkAllDefined(c, {input, grad_output, weight, save_mean, save_var});
      checkAllSameGPU(c, {input, grad_output, weight, save_mean, save_var});
      if (input->scalar_type() == ScalarType::Half) {
        checkScalarType(c, weight, ScalarType::Float);
      } else {
        checkAllSameType(c, {input, weight});
      }
      checkAllSameType(c, {input, grad_output});
      checkAllSameType(c, {weight, save_mean, save_var});
      checkAllContiguous(c, {input, grad_output, save_mean, save_var});
      checkDimRange(c, input, 2, 6 /* exclusive */);
      checkSameSize(c, input, grad_output);
      auto num_features = input->size(1);
      for (auto t : {weight, save_mean, save_var}) {
        checkNumel(c, t, num_features);
      }

      miopenBatchNormMode_t mode;
      if (input->dim() == 2) {
        mode = miopenBNPerActivation;
      } else {
        mode = miopenBNSpatial;
      }

      auto grad_input_t  = empty(input->sizes(), input->options());
      auto grad_weight_t = empty(weight->sizes(), weight->options());
      auto grad_bias_t   = empty(weight->sizes(), weight->options());

      auto handle = getMiopenHandle();
      auto dataType = getMiopenDataType(*input);

      TensorDescriptor idesc{ *input, 4 };  // input, output, grad_output descriptor
      TensorDescriptor wdesc{ expandScale(*weight, input->dim()), 4 };  // descriptor for weight, bias, save_mean, etc.

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      MIOPEN_CHECK(miopenBatchNormalizationBackward(
        handle, mode, &one, &zero, &one, &zero,
        idesc.desc(), input->data_ptr(),
        idesc.desc(), grad_output->data_ptr(),
        idesc.desc(), grad_input_t.data_ptr(),
        wdesc.desc(), weight->data_ptr(),
        grad_weight_t.data_ptr(),
        grad_bias_t.data_ptr(),
        epsilon,
        save_mean->data_ptr(),
        save_var->data_ptr()));

      return tuple<Tensor,Tensor,Tensor>{grad_input_t, grad_weight_t, grad_bias_t};
        */
}
