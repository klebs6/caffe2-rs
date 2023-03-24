crate::ix!();

pub trait SetOutputTensor {

    #[inline] fn set_output_tensor(
        &mut self, 
        idx: i32,
        tensor: Tensor)  
    {
        todo!();
        /*
            if (!isLegacyOperator()) {
    #if defined(EXPOSE_C2_OPS) || \
        !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
          newstyle_outputs_[idx] = at::Tensor(tensor);

          // also update the tensor in the hack
          output_tensors_[idx] = std::move(tensor);
    #else
          CAFFE_THROW("Non-legacy operators are not legal in xplat/caffe2");
    #endif
        } else {
          // update the tensor in the workspace
          BlobSetTensor(outputs_.at(idx), std::move(tensor));
        }
        */
    }
}
