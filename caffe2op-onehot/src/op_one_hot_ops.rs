crate::ix!();

#[test] fn one_hot_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "OneHot",
        ["indices", "index_size_tensor"],
        ["one_hots"],
    )

    workspace.FeedBlob("indices", np.array([0,1,2,3,4]).astype(np.long))
    print("indices:\n", workspace.FetchBlob("indices"))

    workspace.FeedBlob("index_size_tensor", np.array([5]).astype(np.long))
    print("index_size_tensor:\n", workspace.FetchBlob("index_size_tensor"))

    workspace.RunOperatorOnce(op)
    print("one_hots: \n", workspace.FetchBlob("one_hots"))

    indices:
     [0 1 2 3 4]
    index_size_tensor:
     [5]
    one_hots:
     [[1. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0.]
     [0. 0. 1. 0. 0.]
     [0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 1.]]
    */
}

/**
  | The *OneHot* op accepts two inputs *indices*
  | and *index_size_tensor*, and produces
  | a single output one_hots*. For each
  | index in *indices* the op creates a one-hot
  | row in *one_hots* of length index_size_tensor*
  | where all entries are zero except the
  | entry at the index is 1. The size of one_hots*
  | is *len(indices)* x *index_size_tensor*.
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/one_hot_ops.h
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/one_hot_ops.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct OneHotOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{OneHot, 2}

num_outputs!{OneHot, 1}

inputs!{OneHot, 
    0 => ("indices", "The active index for each example in the batch."),
    1 => ("index_size_tensor", "Scalar with the size of the index. Must be in CPU context")
}

outputs!{OneHot, 
    0 => ("one_hots", "Matrix of size len(indices) x index_size")
}

// TODO: enable the filler
disallow_input_fillers!{OneHot}

impl<Context> OneHotOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& indices = Input(0);
        CAFFE_ENFORCE_EQ(
            indices.dim(),
            1,
            "indices input must be 1D tensor of data type int64_t");

        // Index size input must be in CPU context
        auto& index_size_tensor = this->template Input<Tensor>(1, CPU);
        CAFFE_ENFORCE_EQ(
            index_size_tensor.numel(),
            1,
            "index_size_tensor input must be scalar of data type int64_t");

        auto batch_size = indices.numel();
        auto index_size = *index_size_tensor.template data<int64_t>();
        auto one_hots = Output(0);
        one_hots->Resize(batch_size, index_size);
        auto output_size = one_hots->numel();
        if (output_size == 0) {
          return true;
        }

        DoOneHotOp(batch_size, index_size, indices, one_hots);
        return true;
        */
    }
}

/**
  | Input is a matrix tensor. Its first dimension
  | is the batch size. Expand each column
  | of it using one hot encoding. The `lengths`
  | specifies the size of each column after
  | encoding, and the `values` is the dictionary
  | value of one-hot encoding for each column.
  | For example
  | 
  | If data = [[2, 3], [4, 1], [2, 5]], lengths
  | = [2, 3], and values = [2, 4, 1, 3, 5], then
  | 
  | output = [[1, 0, 0, 1, 0], [0, 1, 1, 0, 0],
  | [1, 0, 0, 0, 1]]
  |
  */
pub struct BatchOneHotOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,

    /**
      | allows for fast random access to a given
      | dict and is re-used across runs
      |
      */
    vals_offsets: Vec<i64>,
}

num_inputs!{BatchOneHot, 3}

num_outputs!{BatchOneHot, 1}

inputs!{BatchOneHot, 
    0 => ("data", "input tensor matrix"),
    1 => ("lengths", "the size is the same as the width of the `data`"),
    2 => ("values", "one hot encoding dictionary values")
}

outputs!{BatchOneHot, 
    0 => ("output", "output matrix that expands each input column with one hot encoding")
}

cost_inference_function!{BatchOneHot, /* (OpSchema::CostInferenceFunctionType(CostInferenceForBatchOneHot)) */ }

tensor_inference_function!{BatchOneHot, /* (TensorInferenceForBatchOneHot) */}

value_key_length_input_fillers!{
    /*
    BatchOneHot, (
        BatchOneHotOp<CPUContext>::X,
        BatchOneHotOp<CPUContext>::VALS,
        BatchOneHotOp<CPUContext>::LENS
    )
    */
}

input_tags!{
    BatchOneHotOp {
        X,
        Lens,
        Vals
    }
}

output_tags!{
    BatchOneHotOp {
        OneHot
    }
}

impl<Context> BatchOneHotOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(X));
        */
    }
}

/**
  | Input is a matrix tensor. Its first dimension
  | is the batch size. For each column, bucketize
  | it based on the boundary values and then
  | do one hot encoding. The `lengths` specifies
  | the number of boundary values for each
  | column. The final number of buckets
  | is this number plus 1. This would also
  | be the expanded feature size. `boundaries`
  | specifies all the boundary values.
  | 
  | -----------
  | @note
  | 
  | each bucket is right-inclusive. That
  | is, given boundary values [b1, b2, b3],
  | the buckets are defined as (-int, b1],
  | (b1, b2], (b2, b3], (b3, inf).
  | 
  | For example
  | 
  | data = [[2, 3], [4, 1], [2, 5]], lengths
  | = [2, 3],
  | 
  | If boundaries = [0.1, 2.5, 1, 3.1, 4.5],
  | then
  | 
  | output = [[0, 1, 0, 0, 1, 0, 0], [0, 0,
  | 1, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1]]
  | 
  | If boundaries = [0.1, 2.5, 1, 1, 3.1],
  | then
  | 
  | output = [[0, 1, 0, 0, 0, 1, 0], [0, 0,
  | 1, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 1]]
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct BatchBucketOneHotOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{BatchBucketOneHot, 3}

num_outputs!{BatchBucketOneHot, 1}

inputs!{BatchBucketOneHot, 
    0 => ("data", "input tensor matrix"),
    1 => ("lengths", "the size is the same as the width of the `data`"),
    2 => ("boundaries", "bucket boundaries")
}

outputs!{BatchBucketOneHot, 
    0 => ("output", "output matrix that expands each input column with one hot encoding based on the bucketization")
}

tensor_inference_function!{BatchBucketOneHot, /* (TensorInferenceForBucketBatchOneHot) */}

disallow_input_fillers!{BatchBucketOneHot}

impl<Context> BatchBucketOneHotOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}

input_tags!{
    BatchBucketOneHotOp {
        X,
        Lens,
        Boundaries
    }
}

output_tags!{
    BatchBucketOneHotOp {
        OneHot
    }
}

impl BatchOneHotOp<CPUContext> {
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& input = Input(X);
      auto& lens = Input(LENS);
      auto& vals = Input(VALS);
      CAFFE_ENFORCE_GE(input.dim(), 1);
      auto N = input.size(0);
      auto D = input.size_from_dim(1);
      CAFFE_ENFORCE_EQ(lens.numel(), D);

      const auto* lens_data = lens.template data<int32_t>();
      int64_t output_dim = 0;
      valsOffsets_.resize(D + 1);
      for (int64_t i = 0; i < D; i++) {
        CAFFE_ENFORCE_GE(lens_data[i], 0);
        valsOffsets_[i] = output_dim;
        output_dim += lens_data[i];
      }
      valsOffsets_[D] = output_dim;

      CAFFE_ENFORCE_EQ(vals.numel(), output_dim);

      auto* output = Output(ONE_HOT, {N, output_dim}, at::dtype<T>());

      const auto* input_data = input.template data<T>();
      const auto* vals_data = vals.template data<T>();
      auto* output_data = output->template mutable_data<T>();

      for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < D; j++) {
          const auto input_val = input_data[i * D + j];
          for (int64_t k = valsOffsets_[j]; k < valsOffsets_[j + 1]; ++k) {
            output_data[k] = vals_data[k] == input_val;
          }
        }
        output_data += output_dim;
      }

      return true;
        */
    }
}

#[inline] pub fn tensor_inference_for_batch_one_hot(
    def:   &OperatorDef, 
    input: &Vec<TensorShape>) -> Vec<TensorShape> 
{
    todo!();
    /*
        std::vector<int64_t> output_dims(2);
      output_dims[0] = in[0].dims(0); // N
      output_dims[1] = in[2].dims(0); // vals.size()
      return vector<TensorShape>{
          CreateTensorShape(vector<int64_t>{output_dims}, in[0].data_type())};
    */
}

#[inline] pub fn tensor_inference_for_bucket_batch_one_hot(
    def:   &OperatorDef, 
    input: &Vec<TensorShape>) -> Vec<TensorShape> {
    
    todo!();
    /*
        std::vector<int64_t> output_dims(2);
      output_dims[0] = in[0].dims(0); // N
      output_dims[1] = in[1].dims(0) + in[2].dims(0); // vals.size() + length.size()
      return vector<TensorShape>{
          CreateTensorShape(vector<int64_t>{output_dims}, in[0].data_type())};
    */
}

#[inline] pub fn cost_inference_for_batch_one_hot(
    def:   &OperatorDef, 
    input: &Vec<TensorShape>) -> OpSchemaCost {
    
    todo!();
    /*
        CAFFE_ENFORCE_EQ(in.size(), 3, "BatchOneHot requires three inputs");
      struct OpSchema::Cost c;
      const TensorShape output = TensorInferenceForBatchOneHot(def, in)[0];

      const auto& data = in[0];
      const auto& length = in[1];
      const auto& values = in[2];

      uint64_t nBytesData = nElemFromDim(data) * sizeof(data.data_type());
      uint64_t nBytesLength = nElemFromDim(length) * sizeof(length.data_type());
      uint64_t nBytesValues = nElemFromDim(values) * sizeof(values.data_type());
      c.flops = 0;
      c.bytes_read = nBytesData + nBytesLength + nBytesValues;
      c.bytes_written = nElemFromDim(output) * sizeof(output.data_type());
      c.params_bytes = 0;
      return c;
    */
}

impl OneHotOp<CPUContext> {

    #[inline] pub fn do_one_hot_op(&mut self, 
        batch_size: i64,
        index_size: i64,
        indices:    &Tensor,
        one_hots:   *mut Tensor)  {
        
        todo!();
        /*
            const int64_t* indices_ptr = indices.template data<int64_t>();
      float* one_hots_ptr = one_hots->template mutable_data<float>();
      memset(one_hots_ptr, 0, one_hots->nbytes());
      for (int i = 0; i < batch_size; ++i) {
        auto label_idx = indices_ptr[i];
        DCHECK((0 <= label_idx) && (label_idx < index_size));
        one_hots_ptr[label_idx] = 1.0;
        one_hots_ptr += index_size;
      }
        */
    }
}

impl BatchBucketOneHotOp<CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(X);
      auto& lens = Input(LENS);
      auto& boundaries = Input(BOUNDARIES);
      CAFFE_ENFORCE_GE(input.dim(), 1);
      auto N = input.size(0);
      auto D = input.size_from_dim(1);
      CAFFE_ENFORCE_EQ(lens.numel(), D);

      const auto* lens_data = lens.template data<int32_t>();

      CAFFE_ENFORCE_EQ(
          std::accumulate(lens_data, lens_data + lens.numel(), 0),
          boundaries.numel(),
          "The sum of length should be equal to the length of boundaries");

      int64_t output_dim = 0;
      for (int64_t i = 0; i < D; i++) {
        CAFFE_ENFORCE_GT(lens_data[i], 0);
        // Number of buckets is number of bucket edges + 1
        output_dim += (lens_data[i] + 1);
      }

      auto* output = Output(ONE_HOT, {N, output_dim}, at::dtype<float>());

      const auto* input_data = input.template data<float>();
      const auto* boundaries_data = boundaries.template data<float>();
      auto* output_data = output->template mutable_data<float>();

      math::Set<float, CPUContext>(output->numel(), 0.f, output_data, &context_);

      int64_t pos = 0;
      for (int64_t i = 0; i < N; i++) {
        auto* boundaries_offset = boundaries_data;
        int64_t output_offset = 0;

        for (int64_t j = 0; j < D; j++) {
          // here we assume the boundary values for each feature are sorted
          int64_t lower_bucket_idx = std::lower_bound(
                                        boundaries_offset,
                                        boundaries_offset + lens_data[j],
                                        input_data[pos]) -
              boundaries_offset;

          int64_t upper_bucket_idx = std::upper_bound(
                                        boundaries_offset,
                                        boundaries_offset + lens_data[j],
                                        input_data[pos]) -
              boundaries_offset;

          int64_t bucket_idx = (lower_bucket_idx + upper_bucket_idx) / 2;
          output_data[i * output_dim + output_offset + bucket_idx] = 1.0;
          boundaries_offset += lens_data[j];
          output_offset += (lens_data[j] + 1);
          pos++;
        }
      }

      return true;
        */
    }
}

/**
  | Given a sequence of indices, segmented
  | by the lengths tensor, returns a matrix
  | that has the elements in each sequence
  | set to 1.0, and 0.0 everywhere else.
  |
  */
pub struct SegmentOneHotOp {
    storage: OperatorStorage,
    context: CPUContext,
}

num_inputs!{SegmentOneHot,  3}
num_outputs!{SegmentOneHot, 1}

inputs!{SegmentOneHot, 
    0 => ("lengths", "Size of each segment."),
    1 => ("indices", "Active indices, of size sum(lengths)"),
    2 => ("index_size_tensor", "Size of the index")
}

outputs!{SegmentOneHot, 
    0 => ("one_hots", "Matrix of size len(lengths) x index_size")
}

// TODO: enable the filler
disallow_input_fillers!{SegmentOneHot}

impl SegmentOneHotOp {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& lengths = Input(0);
        auto& indices = Input(1);
        auto& index_size_tensor = Input(2);
        CAFFE_ENFORCE(lengths.dim() == 1);
        CAFFE_ENFORCE(indices.dim() == 1);
        CAFFE_ENFORCE(index_size_tensor.numel() == 1);
        auto batch_size = lengths.numel();
        auto index_size = *index_size_tensor.data<int64_t>();
        CAFFE_ENFORCE(index_size > 0);

        auto* lengths_ptr = lengths.data<int32_t>();
        auto* indices_ptr = indices.data<int64_t>();

        auto* one_hots = Output(0, {batch_size, index_size}, at::dtype<float>());
        auto* one_hots_ptr = one_hots->template mutable_data<float>();
        if (one_hots->numel() == 0) {
          return true;
        }
        memset(one_hots_ptr, 0, one_hots->nbytes());
        int el_idx = 0;
        for (int i = 0; i < batch_size; ++i) {
          for (int j = 0; j < lengths_ptr[i]; ++j) {
            DCHECK(el_idx < indices.numel());
            auto label_idx = indices_ptr[el_idx++];
            DCHECK((0 <= label_idx) && (label_idx < index_size));
            one_hots_ptr[label_idx] = 1.0;
          }
          one_hots_ptr += index_size;
        }
        return true;
        */
    }
}

register_cpu_operator!{BatchBucketOneHot,  BatchBucketOneHotOp<CPUContext>}
register_cpu_operator!{BatchOneHot,        BatchOneHotOp<CPUContext>}
register_cpu_operator!{OneHot,             OneHotOp<CPUContext>}
register_cpu_operator!{SegmentOneHot,      SegmentOneHotOp}

no_gradient!{BatchOneHot}
no_gradient!{OneHot}
no_gradient!{SegmentOneHot}
no_gradient!{BucketBatchOneHot}
