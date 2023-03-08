crate::ix!();

use crate::{
    OperatorStorage
};

/**
  | Coalesce the N inputs into N outputs
  | and a single coalesced output blob.
  | 
  | This allows operations that operate
  | over multiple small kernels (e.g. biases
  | in a deep CNN) to be coalesced into a single
  | larger operation, amortizing the kernel
  | launch overhead, synchronization
  | costs for distributed computation,
  | etc.
  | 
  | The operator:
  | 
  | - computes the total size of the coalesced
  | blob by summing the input sizes
  | 
  | - allocates the coalesced output blob
  | as the total size
  | 
  | - copies the input vectors into the coalesced
  | blob, at the correct offset.
  | 
  | - aliases each Output(i) to- point into
  | the coalesced blob, at the corresponding
  | offset for Input(i).
  | 
  | This is 'unsafe' as the output vectors
  | are aliased, so use with caution.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct UnsafeCoalesceOp<Context> {

    storage: OperatorStorage,
    context: Context,
}

allow_inplace!{UnsafeCoalesce, 
    |input: i32, output: i32| {
        input == output
    }
}

num_inputs_outputs!{UnsafeCoalesce, 
    |inputs: i32, outputs: i32| {
        inputs + 1 == outputs
    }
}

impl<Context> UnsafeCoalesceOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            size_t coalesced_size = 0;
        for (int i = 0; i < InputSize(); ++i) {
          // For now only float type is supported
          CAFFE_ENFORCE(
              Input(i).dtype().template Match<float>(),
              "Must only coalesce float type, error at input: ",
              i);
        }

        for (int i = 0; i < InputSize(); ++i) {
          coalesced_size += Input(i).numel();
        }
        auto* coalesced = Output(OutputSize() - 1, coalesced_size, at::dtype<float>());
        auto coalesced_data = coalesced->template mutable_data<float>();

        size_t coalesced_offset = 0;
        for (auto i = 0; i < InputSize(); ++i) {
          const auto num_elems = Input(i).numel();
          auto input_sizes = Input(i).sizes().vec();
          // Don't do anything if both tensors are already pointing on the same data
          auto input_data = Input(i).template data<float>();
          if (input_data != coalesced_data + coalesced_offset) {
            // Make sure that we don't run operation on the same tensor
            CAFFE_ENFORCE_NE(
                input_data - Input(i).unsafeGetTensorImpl()->storage_offset(),
                coalesced_data -
                    Output(OutputSize() - 1)
                        ->unsafeGetTensorImpl()
                        ->storage_offset(),
                "Tensors used in UnsafeCoalesce operator cannot share storage, unless it's inplace operation");
            context_.CopyItemsSameDevice(
                Input(i).dtype(),
                num_elems,
                input_data,
                coalesced_data + coalesced_offset);

            // Note: this could cause Input(i) to free it's data if
            // Output(i) and Input(i) alias each other. This is safe on a
            // GPU (as the copy will happen-before the free), but it's
            // worth mentioning.
            OperatorStorage::SetOutputTensor(i, coalesced->Alias());
            Output(i)->unsafeGetTensorImpl()->set_storage_offset(coalesced_offset);
            Output(i)->Resize(input_sizes);
          }
          coalesced_offset += num_elems;
        }
        return true;
        */
    }
}

register_cpu_operator!{UnsafeCoalesce, UnsafeCoalesceOp<CPUContext>}
