crate::ix!();

use crate::{
    ConvOp,
    ConvPoolOpBase,
    Workspace,
    OperatorDef,
    Blob
};

pub struct ConvReluOp<T,Context> {
    base: ConvPoolOpBase<Context>,
    local_ws:            Workspace,
    local_input_blobs:   Vec<*mut Blob>,
    local_output_blobs:  Vec<*mut Blob>,
    local_op:            Box<ConvOp<T,Context>>,
    phantom: PhantomData<T>,
}

impl<T,Context> ConvReluOp<T,Context> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : ConvPoolOpBase<Context>(operator_def, ws) 

        for (auto name : operator_def.input()) {
          local_input_blobs_.push_back(local_ws_.CreateBlob(name));
          CHECK_NOTNULL(local_input_blobs_.back());
        }
        local_op_.reset(new ConvOp<T, Context>(operator_def, &local_ws_));
        for (auto name : operator_def.output()) {
          local_output_blobs_.push_back(local_ws_.GetBlob(name));
          CHECK_NOTNULL(local_output_blobs_.back());
        }
        */
    }
    
    #[inline] pub fn run_on_device_with_ordernchw(&mut self) -> bool {
        
        todo!();
        /*
            // Delegate to local conv operator
      for (int i = 0; i < this->InputSize(); ++i) {
        local_input_blobs_[i]->ShareExternal(
            const_cast<void*>(this->Inputs()[i]->GetRaw()),
            this->Inputs()[i]->meta());
      }

      if (!local_op_->RunOnDeviceWithOrderNCHW()) {
        return false;
      }

      // Apply Relu
      Tensor* local_output =
          BlobGetMutableTensor(local_output_blobs_[0], Context::GetDeviceType());
      const T* output_local_data = local_output->template data<T>();

      Tensor* output =
          Operator<Context>::Output(0, local_output->sizes(), at::dtype<T>());
      T* output_data = output->template mutable_data<T>();
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
      for (int i = 0; i < output->numel(); ++i) {
        output_data[i] = std::max(static_cast<T>(0), output_local_data[i]);
      }

      return true;
        */
    }
    
    #[inline] pub fn run_on_device_with_ordernhwc(&mut self) -> bool {
        
        todo!();
        /*
            // Delegate to local conv operator
      for (int i = 0; i < this->InputSize(); ++i) {
        local_input_blobs_[i]->ShareExternal(
            const_cast<void*>(this->Inputs()[i]->GetRaw()),
            this->Inputs()[i]->meta());
      }

      if (!local_op_->RunOnDeviceWithOrderNHWC()) {
        return false;
      }

      // Apply Relu
      Tensor* local_output =
          BlobGetMutableTensor(local_output_blobs_[0], Context::GetDeviceType());
      const T* output_local_data = local_output->template data<T>();

      Tensor* output =
          Operator<Context>::Output(0, local_output->sizes(), at::dtype<T>());
      T* output_data = output->template mutable_data<T>();
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
      for (int i = 0; i < output->numel(); ++i) {
        output_data[i] = std::max(static_cast<T>(0), output_local_data[i]);
      }

      return true;
        */
    }
}

register_cpu_operator!{ConvRelu, ConvReluOp<float, CPUContext>}

num_inputs!{ConvRelu, (2,3)}

num_outputs!{ConvRelu, 1}

cost_inference_function!{ConvRelu, /* (OpSchema::CostInferenceFunctionType(
        ConvPoolOpBase<CPUContext>::CostInferenceForConv)) */ }

tensor_inference_function!{ConvRelu, /* (ConvPoolOpBase<CPUContext>::TensorInferenceForConv) */}
