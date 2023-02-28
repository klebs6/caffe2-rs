crate::ix!();

use crate::{
    OperatorStorage,
    OperatorDef,
    GradientMakerBase,
    Tensor
};

/**
  | Convert sparse representations to
  | dense with given indices.
  | 
  | Transforms a sparse representation
  | of map<id, value> represented as `indices`
  | vector and `values` tensor into a compacted
  | tensor where the first dimension is
  | determined by the first dimension of
  | the 3rd input if it is given or the max
  | index. Missing values are filled with
  | zeros.
  | 
  | The op supports duplicated indices
  | and performs summation over corresponding
  | values. This behavior is useful for
  | converting GradientSlices into dense
  | representation.
  | 
  | After running this op:
  | 
  | -output[indices[i], :] += values[i]
  | // sum over all indices[i] equal to the
  | index
  | 
  | -output[j, ...] = 0 if j not in indices
  |
  */
pub struct SparseToDenseOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    //USE_DISPATCH_HELPER
    storage:           OperatorStorage,
    context:           Context,

    output_first_dim:  i32,
    scratch:           Tensor, // {Context::GetDeviceType()};
    max_element_host:  Tensor, // default = CPU
    max_element:       Tensor,
}

register_cpu_operator!{SparseToDense, SparseToDenseOp<CPUContext>}

num_inputs!{SparseToDense, (2,3)}

num_outputs!{SparseToDense, 1}

inputs!{SparseToDense, 
    0 => ("indices",           "1-D int32/int64 tensor of concatenated ids of data"),
    1 => ("values",            "Data tensor, first dimension has to match `indices`, basic numeric types are supported"),
    2 => ("data_to_infer_dim", "Optional: if provided, the first dimension of output is the first dimension of this tensor.")
}

outputs!{SparseToDense, 
    0 => ("output",            "Output tensor of the same type as `values` of shape `[len(lengths), len(mask)] + shape(default_value)` (if `lengths` is not provided the first dimension is omitted)")
}

tensor_inference_function!{SparseToDense, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          vector<TensorShape> out(1);
          if (in.size() == 3) {
            out[0].add_dims(in[2].dims(0));
          } else {
            out[0].set_unknown_shape(true);
            return out;
          }
          for (int i = 1; i < in[1].dims().size(); i++) {
            out[0].add_dims(in[1].dims(i));
          }
          out[0].set_data_type(in[1].data_type());
          return out;
        */
    }
}

input_tags!{
    SparseToDenseOp
    {
        Indices,
        Values,
        DataToInferDim
    }
}

impl<Context> SparseToDenseOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            output_first_dim_( this->template GetSingleArgument<int>("output_first_dim", 0))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, Input(INDICES));
        */
    }
    
    #[inline] pub fn get_output_first_dim<TInd>(&mut self, sparse_indices_vec: *const TInd, sparse_indices_len: i32) -> i32 {
    
        todo!();
        /*
            if (output_first_dim_ > 0) {
          CAFFE_ENFORCE_EQ(InputSize(), 2);
          return output_first_dim_;
        }
        if (InputSize() == 3) {
          auto& data_to_infer_dim = Input(DATA_TO_INFER_DIM);
          CAFFE_ENFORCE_GE(data_to_infer_dim.dim(), 1);
          return data_to_infer_dim.dim32(0);
        }
        if (sparse_indices_len <= 0) {
          return 0;
        }

        // Awkward way to get the max element to make it work with both CUDA
        // and CPU.
        ReinitializeTensor(&max_element_, {1}, at::dtype<TInd>().device(Context::GetDeviceType()));
        TInd* max_element_ptr = max_element_.template mutable_data<TInd>();
        math::ReduceMax<TInd>(sparse_indices_len, sparse_indices_vec, max_element_ptr,
              &scratch_, &context_);
        max_element_host_.CopyFrom(max_element_);
        return 1 + max_element_host_.template data<TInd>()[0];
        */
    }
    
    #[inline] pub fn do_run_with_type<TInd>(&mut self) -> bool {
    
        todo!();
        /*
            return DispatchHelper<
            TensorTypes2<
                float,
                int32_t,
                int64_t,
                GenericTensorImplementation>,
            TInd>::call(this, Input(VALUES));
        */
    }
    
    #[inline] pub fn do_run_with_type2<TInd, TData>(&mut self) -> bool {
    
        todo!();
        /*
            auto& sparse_indices = Input(INDICES);
        CAFFE_ENFORCE_EQ(sparse_indices.dim(), 1);
        auto& sparse_values = Input(VALUES);
        CAFFE_ENFORCE_GE(sparse_values.dim(), 1);
        CAFFE_ENFORCE_EQ(sparse_indices.numel(), sparse_values.size(0));

        const TInd* sparse_indices_vec = sparse_indices.template data<TInd>();
        const int32_t sparse_indices_len = sparse_indices.dim32(0);
        const int output_first_dim =
            GetOutputFirstDim(sparse_indices_vec, sparse_indices_len);

        auto shape = sparse_values.sizes().vec();
        shape[0] = output_first_dim;

        auto* output = Output(0, shape, at::dtype<TData>());

        TData* output_data = output->template mutable_data<TData>();
        if (!output_first_dim) {
          return true;
        }
        memset(output_data, 0, output->nbytes());
        const auto block_nitems = sparse_values.size_from_dim(1);
        const TData* sparse_values_vec = sparse_values.template data<TData>();

        for (int32_t i = 0; i < sparse_indices_len; i++) {
          const TInd idx = sparse_indices_vec[i];
          CAFFE_ENFORCE_GE(idx, 0);
          CAFFE_ENFORCE_LT(idx, output_first_dim);
          math::Add(
              block_nitems,
              output_data + idx * block_nitems,
              sparse_values_vec + i * block_nitems,
              output_data + idx * block_nitems,
              &context_);
        }
        return true;
        */
    }
    
    #[inline] pub fn do_run_with_other_type2<TInd>(&mut self) -> bool {
    
        todo!();
        /*
            CAFFE_THROW(
            "SparseToDense is not implemented on tensor of type ",
            Input(VALUES).dtype().name(),
            "consider adding it as a type in the DispatchHelper list or "
            "implementing a generic version (which won't work for "
            "duplicated indices though)");
        */
    }
}

pub struct GetSparseToDenseGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSparseToDenseGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "Gather", "", vector<string>{GO(0), I(0)}, vector<string>{GI(1)});
        */
    }
}

register_gradient!{SparseToDense, GetSparseToDenseGradient}
