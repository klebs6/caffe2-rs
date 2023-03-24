crate::ix!();

pub trait GetInputs {

    #[inline] fn inputs(&self) -> &Vec<*const Blob> {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            isLegacyOperator(),
            "Inputs() not supported for operators exported to c10.");
        return inputs_;
        */
    }
}

pub trait GetInputTensorShapes {

    #[inline] fn input_tensor_shapes(&self) -> Vec<TensorShape> {
        
        todo!();
        /*
            CAFFE_ENFORCE(
          isLegacyOperator(),
          "InputTensorShapes() not supported for operators exported to c10.");
      vector<TensorShape> tps;
      for (const auto& blob : inputs_) {
        tps.push_back(GetTensorShapeOfBlob(blob));
      }
      return tps;
        */
    }
}

//TODO: clean up
pub trait GetInputAtIndex {

    /**
      | Retrieve a non-owning reference to the
      | input at position 'idx' for this operator.
      | The returned reference is valid for the
      | duration of the RunOnDevice call.  The
      | optional 'type' parameter can be used to
      | assert a required device type for the
      | input (by default, we assert that the
      | tensor is consistent with the device type
      | implied by the Context parameter of an
      | Operator.)
      */
    #[inline] fn input(
        &mut self, 
        idx: i32,
        ty: Option<DeviceType>) -> &Tensor 
    {
        let ty = todo!(); // ty.unwrap_or(Context::GetDeviceType());
        
        todo!();
        /*
            return OperatorStorage::template Input<Tensor>(idx, type);
        */
    }

    /**
      | Get the inputs and outputs as specific
      | types.
      |
      */
    #[inline] fn input_from_index<'a, T>(idx: i32) -> &'a T {
        todo!();
        /*
            static_assert(
                !std::is_same<T, Tensor>::value,
                "You should use Input<Tensor>(int, DeviceType) for "
                "Tensor.");
            DCHECK_LT((size_t)idx, inputs_.size());
            try {
              return inputs_.at(idx)->template Get<T>();
            } catch (::caffe2::EnforceNotMet& enf) {
              if (has_debug_def()) {
                TORCH_RETHROW(enf, "Offending Blob name: ", debug_def().input(idx), ".");
              }
              throw enf;
            }
        */
    }

    /**
      | TODO(jerryzh): Remove template and the type
      | argument?
      |
      | This is to keep the API changes minimal and
      | make refactoring a bit easier
      */
    #[inline] fn input_from_index_and_device_type<'a, T>(
        idx: i32, 
        ty:  DeviceType) -> &'a T 
    {
        todo!();
        /*
            if (isLegacyOperator()) {
              static_assert(
                  std::is_same<T, Tensor>::value,
                  "Input(int, DeviceType) is only available for Tensor");
              DCHECK_LT((size_t)idx, inputs_.size());
              try {
                // TODO(jerryzh): We'll need to check device type in Get<T>() later
                // Get<T>() -> Get<T>(type)
                const auto& tensor = inputs_.at(idx)->template Get<T>();
                return tensor;
              } catch (::caffe2::EnforceNotMet& enf) {
                if (has_debug_def()) {
                  TORCH_RETHROW(enf, "Offending Blob name: ", debug_def().input(idx), ".");
                }
                throw enf;
              }
            }
        #if defined(EXPOSE_C2_OPS) || \
            !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
            DCHECK_LT(0U, newstyle_inputs_.size());
            IValue ival;
            if (newstyle_inputs_[0].isTensorList()) {
              // if the first input is a tensor list, we get input tensors by indexing
              // into that list. currently, this means that only tensors from that list
              // are accessible as inputs. any hypothetical input tensors that come
              // after the list are not accessible.
              auto tensorList = newstyle_inputs_[0].toTensorVector();
              DCHECK_LT((size_t)idx, tensorList.size());
              ival = tensorList[idx];
            } else {
              // if the first input is not a tensor list, we get input tensors by
              // indexing into the inputs.
              DCHECK_LT((size_t)idx, newstyle_inputs_.size());
              ival = newstyle_inputs_[idx];
            }
            CAFFE_ENFORCE(
                ival.isTensor(),
                "Input(int, DeviceType) is only available for IValues that store Tensors");
            auto t = ival.toTensor();
            if (!t.is_contiguous()) {
              t = t.contiguous();
            }
            Tensor tensor = caffe2::Tensor(std::move(t));
            CAFFE_ENFORCE_EQ(tensor.GetDeviceType(), type);
            input_tensors_[idx] = std::move(tensor);
            return input_tensors_[idx];
        #else
            CAFFE_THROW("Non-legacy operators are not legal in xplat/caffe2");
        #endif
        */
    }
}
