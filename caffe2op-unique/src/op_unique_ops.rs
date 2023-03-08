crate::ix!();

use crate::{
    OperatorStorage,
    CPUContext,
    Tensor,
};

/**
  | Deduplicates input indices vector
  | and optionally produces reverse remapping.
  | 
  | There's no guarantees on the ordering
  | of the output indices.
  | 
  | Current implementation produces a
  | sorted list but it's not guaranteed
  | in general.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct UniqueOp<Context> {

    storage:              OperatorStorage,
    context:              Context,
    order:                Vec<i32>,
    thrust_unique_buffer: Tensor,
    cuda_order_buffer:    Tensor, // {Context::GetDeviceType()};
    second_order_buffer:  Tensor, // {Context::GetDeviceType()};
}

num_inputs!{Unique, 1}

num_outputs!{Unique, (1,2)}

inputs!{Unique, 
    0 => ("indices", "1D tensor of int32 or int64 indices.")
}

outputs!{Unique, 
    0 => ("unique_indices", "1D tensor of deduped entries."),
    1 => ("remapping",      "(optional) mapping from `indices` to `unique_indices`. This has the same shape as `indices`. Its elements are the indices into `unique_indices` such that `Gather(['unique_indices', 'remapping'])` yields `indices`.")
}

tensor_inference_function!{
    Unique, 
    /* [](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      std::vector<TensorShape> out(1);
      out[0].set_data_type(in[0].data_type());
      CAFFE_ENFORCE_EQ(in[0].dims_size(), 1);
      if (in[0].dims(0) <= 1) {
        // This special case is useful in some situation, e.g., when feeding
        // tensor inference with empty tensor (where the first dim is the batch
        // size)
        out[0].add_dims(in[0].dims(0));
      } else {
        out[0].set_unknown_shape(true);
      }
      if (def.output_size() > 1) {
        // Remapping has the same shape as the input tensor
        out.push_back(in[0]);
        out.back().set_data_type(TensorProto::INT32);
      }
      return out;
    } */}


should_not_do_gradient!{Unique}

output_tags!{
    UniqueOp {
        Unique,
        Remapping
    }
}

impl<Context> UniqueOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(0));
        */
    }
}

impl UniqueOp<CPUContext> {

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& inputTensor = Input(0);
          // use dim32 to enforce that it's fine to have remapping of type int
          int N = inputTensor.dim32(0);
          CAFFE_ENFORCE_EQ(inputTensor.dim(), 1, "Input should be a vector");

          int* remapping = nullptr;
          if (REMAPPING < OutputSize()) {
            auto* remappingTensor =
                Output(REMAPPING, inputTensor.sizes(), at::dtype<int>());
            remapping = remappingTensor->template mutable_data<int>();
          }

          const T* input = inputTensor.template data<T>();
          // TODO(dzhulgakov): if perf becomes an issue consider doing hash table
          // instead of sorting
          order_.resize(N);
          std::iota(order_.begin(), order_.end(), 0);
          std::sort(order_.begin(), order_.end(), [input](const int x, const int y) {
            return input[x] < input[y];
          });
          int K = N;
          for (int i = 1; i < N; ++i) {
            K -= input[order_[i]] == input[order_[i - 1]];
          }
          auto* uniqueTensor = Output(UNIQUE, {K}, at::dtype<T>());
          T* unique = uniqueTensor->template mutable_data<T>();
          K = 0;
          T prev = -1;
          for (int i = 0; i < N; ++i) {
            if (i == 0 || prev != input[order_[i]]) {
              prev = unique[K++] = input[order_[i]];
            }
            if (remapping) {
              remapping[order_[i]] = K - 1;
            }
          }
          return true;
        */
    }
}

register_cpu_operator!{Unique, UniqueOp<CPUContext>}
