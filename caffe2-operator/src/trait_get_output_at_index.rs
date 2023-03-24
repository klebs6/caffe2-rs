crate::ix!();

pub trait GetOutputAtIndex {
    
    /**
      | Retrieve a non-owning pointer to the
      | output at position 'idx', initializing it
      | to have size 'dims' and properties
      | 'options' if there is no pre-existing
      | output or the pre-existing output does not
      | have the correct options.  The returned
      | pointer is valid for the duration of the
      | RunOnDevice call.  If device is not
      | explicitly specified in options, we
      | default to allocating output on the
      | current device of the device type implied
      | by the Context parameter of this Operator.
      |
      | Note [Operator::Output what?]
      | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |
      | The contract of Operator::Output is
      | somewhat complex; it is perhaps better
      | understood in terms of what was
      | historically an idiomatic Caffe2 operator
      | implementation:
      |
      |     void RunOnDevice() override {
      |         auto* output = Output(0, output_size, dtype<float>());
      |         float* output_ptr = output->data<float>();
      |         // write into output_ptr
      |     }
      |
      | In the simple case, this code does the
      | following things:
      |
      |   1. Allocates a new tensor with size
      |      'output_size' and dtype 'float' (and
      |      device type whatever the Operator's
      |      device type is)
      |
      |   2. "Registers" this tensor as the 0th
      |      output tensor of this operator
      |      (Caffe2 operators don't "return"
      |      outputs; instead, outputs are shoved
      |      into an output vector which the
      |      executor reads out.)
      |
      |   3. Returns the tensor, so the operator
      |      implementation can write the actual
      |      output data into the tensor.
      |
      | So what's this business with
      | "pre-existing" outputs?  Caffe2 commonly
      | applies an optimization whereby it reuses
      | tensors on subsequent runs of operators in
      | a graph.  It doesn't know ahead of time
      | what intermediate tensors it will need, so
      | the first time it runs a graph it has all
      | of the operators create the outputs
      | necessary (as described above).  However,
      | the second time around, it will reuse all
      | of the tensors created from the first
      | time. If they are lucky, this time the
      | Output() call is a no-op and just returns
      | the old tensor.
      |
      | However, we cannot /guarantee/ that the
      | output size will be the same the next time
      | the Operator is called; for example,
      | output size may be data dependent and vary
      | between runs.  In this case, we have to
      | resize it to the correct size.  Resizing
      | is still helpful, as we may be able to fit
      | the output in the same space that was
      | previously used.
      |
      */
    #[inline] fn output(
        &mut self, 
        idx:     i32,
        dims:    &[i32],
        options: TensorOptions) -> *mut Tensor 
    {
        todo!();
        /*
            // We'll default device to the device of the current Operator Context
        if (options.device_opt() == c10::nullopt) {
          return OperatorStorage::OutputTensor(
              idx, dims, options.device(context_.device()));
        }
        return OperatorStorage::OutputTensor(idx, dims, options);
        */
    }

    #[inline] fn output_from_idx<T>(idx: i32) -> *mut T {
        todo!();
        /*
            CAFFE_ENFORCE(
                isLegacyOperator(),
                "Output(idx) not supported for operators exported to c10. Please use XOutput instead.");

            static_assert(
                !std::is_same<T, Tensor>::value,
                "You should use Output<Tensor>(int, DeviceType) for "
                "Tensor.");
            return outputs_.at(idx)->template GetMutable<T>();
        */
    }

    // TODO(jerryzh): Remove this template
    #[inline] fn output_from_idx_and_device_type<T>(idx: i32, ty: DeviceType) -> *mut T {
        todo!();
        /*
            if (isLegacyOperator()) {
              static_assert(
                  std::is_same<T, Tensor>::value,
                  "Output(int, DeviceType) is only available for Tensor");
              // When you get a Tensor here it is not fully initialized
              return BlobGetMutableTensor(outputs_.at(idx), type);
            }
        #if defined(EXPOSE_C2_OPS) || \
            !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
            at::Tensor output = newstyle_outputs_[idx];
            if (!output.defined() || caffe2::Tensor(output).GetDeviceType() != type) {
              // Fix tensor type
              Tensor tensor = Tensor(type);
              output = at::Tensor(std::move(tensor.getIntrusivePtr()));
            }
            output_tensors_[idx] = caffe2::Tensor(output);
            newstyle_outputs_[idx] = std::move(output);
            return &output_tensors_[idx];
        #else
            CAFFE_THROW("Non-legacy operators are not legal in xplat/caffe2");
        #endif
        */
    }

    #[inline] fn output_tensor_or_undefined(&mut self, idx: i32) -> Tensor {
        
        todo!();
        /*
            if (isLegacyOperator()) {
          return BlobGetTensorOrUndefined(*outputs_.at(idx));
        }
        return output_tensors_[idx].UnsafeSharedInstance();
        */
    }

    #[inline] fn output_tensor(
        &mut self, 
        idx: i32,
        dims: &[i32],
        options: TensorOptions) -> *mut Tensor 
    {
        todo!();
        /*
            if (isLegacyOperator()) {
          CAFFE_ENFORCE_WITH_CALLER(
              options.device_opt() != c10::nullopt,
              "device must be provided in options.");
          return BlobGetMutableTensor(outputs_.at(idx), dims, options);
        }
    #if defined(EXPOSE_C2_OPS) || \
        !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
        at::Tensor output = newstyle_outputs_[idx];
        Tensor tensor = output.defined()
            ? GetSizedTensorWithOptions(caffe2::Tensor(output), dims, options)
            : caffe2::empty(dims, options);
        // assign it back in case it changed
        output = at::Tensor(std::move(tensor.getIntrusivePtr()));

        output_tensors_[idx] = caffe2::Tensor(output);
        newstyle_outputs_[idx] = std::move(output);
        return &output_tensors_[idx];
    #else
        CAFFE_THROW("Non-legacy operators are not legal in xplat/caffe2");
    #endif
        */
    }

    /**
      | Get output Tensor of the operator and
      | CopyFrom the given Tensor
      |
      */
    #[inline] fn output_tensor_copy_from_base(
        &mut self, 
        idx:     i32,
        options: TensorOptions,
        src:     &Tensor,
        async_:  Option<bool>) -> *mut Tensor 
    {
        let async_: bool = async_.unwrap_or(false);

        todo!();
        /*
            CAFFE_ENFORCE_WITH_CALLER(
            options.device_opt() != c10::nullopt,
            "device must be provided in options.");
        // Ouptut Tensor will always have the same data type as `src`
        if (!options.has_dtype()) {
          options = options.dtype(src.dtype());
        }
        CAFFE_ENFORCE_WITH_CALLER(
            options.dtype() == src.dtype(),
            "We don't allow change of src data type in OutputTensorCopyFrom");
        Tensor* t = OutputTensor(idx, src.sizes(), options);
        t->CopyFrom(src, async);
        return t;
        */
    }

    #[inline] fn output_tensor_alias(
        &mut self, 
        idx: i32,
        src: &Tensor) -> *mut Tensor 
    {
        todo!();
        /*
            CAFFE_ENFORCE(
            isLegacyOperator(),
            "OutputTensorAlias(idx, src) not (yet) supported for operators exported to c10.");
        return BlobSetTensor(OutputBlob(idx), src.Alias());
        */
    }

    #[inline] fn output_base<T>(idx: i32, allocated: *mut T) -> *mut T {
        todo!();
        /*
            CAFFE_ENFORCE(
                isLegacyOperator(),
                "Output(idx, allocated) not supported for operators exported to c10. Please use XOutput.");
            outputs_.at(idx)->Reset(allocated);
            return allocated;
        */
    }
}
