crate::ix!();

use crate::{
    OperatorStorage,
    GradientMakerBase,
    OperatorDef
};

declare_export_caffe2_op_to_c10!{SparseToDenseMask}

pub struct SparseToDenseMaskBase<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage:         OperatorStorage,
    context:         Context,

    sparse:          HashMap<i64,i32>,
    dense:           Vec<i32>,
    features_count:  usize,
}

impl<Context> SparseToDenseMaskBase<Context> {

    const kMaxDenseSize: i64 = 1024 * 128;

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        std::vector<int64_t> mask =
            this->template GetRepeatedArgument<int64_t>("mask");
        featuresCount_ = mask.size();

        CAFFE_ENFORCE(!mask.empty(), "mask can't be empty");
        auto biggest = *std::max_element(mask.begin(), mask.end());
        dense_.assign(std::min(kMaxDenseSize, biggest + 1), -1);
        for (int i = 0; i < mask.size(); i++) {
          int64_t id = mask[i];
          CAFFE_ENFORCE_GE(id, 0, "Only positive IDs are allowed.");
          if (id >= kMaxDenseSize) {
            CAFFE_ENFORCE(sparse_.count(id) == 0, "Duplicated id: ", id);
            sparse_[id] = i;
          } else {
            CAFFE_ENFORCE(dense_[id] == -1, "Duplicated id: ", id);
            dense_[id] = i;
          }
        }
        */
    }
    
    #[inline] pub fn get_feature_idx(&self, id: i64) -> i32 {
        
        todo!();
        /*
            if (id >= kMaxDenseSize) {
          const auto& iter = sparse_.find(id);
          if (iter == sparse_.end()) {
            return -1;
          } else {
            return iter->second;
          }
        } else {
          return (id >= dense_.size()) ? -1 : dense_[id];
        }
        */
    }
}

/**
  | Convert sparse representations to
  | dense with given indices.
  | 
  | Transforms a sparse representation
  | of map<id, value> represented as `indices`
  | vector and `values` tensor into a compacted
  | tensor where the first dimension corresponds
  | to each id provided in the mask argument.
  | Missing values are filled with the value
  | of `default_value`. After running
  | this op:
  | 
  | output[j, :] = values[i] // where mask[j]
  | == indices[i]
  | 
  | output[j, ...] = default_value //
  | when mask[j] doesn't appear in indices
  | 
  | If `lengths` is provided and not empty,
  | an extra "batch" dimension is prepended
  | to the output.
  | 
  | `values` and `default_value` can have
  | additional matching dimensions (the
  | operation is performed on the entire
  | subtensor in this case).
  | 
  | For example, if `lengths` is supplied
  | and `values` is a 1-D vector of floats
  | and `default_value` is a float scalar,
  | the output is going to be a float matrix
  | of size `len(lengths) X len(mask)`.
  |
  */
pub struct SparseToDenseMaskOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    base:                  SparseToDenseMaskBase<Context>,

    return_presence_mask:  bool,
    max_skipped_rows:      u32, // default = 0
    skipped_rows:          u32, // default = 0
}

num_inputs!{SparseToDenseMask, (3,4)}

num_outputs!{SparseToDenseMask, (1,2)}

// TODO: enable the filler
disallow_input_fillers!{SparseToDenseMask}

inputs!{SparseToDenseMask, 
    0 => ("indices",              "1-D int32/int64 tensor of concatenated ids of data"),
    1 => ("values",               "Data tensor, first dimension has to match `indices`"),
    2 => ("default_value",        "Default value for the output if the id is not present in `indices`. Must have the same type as `values` and the same shape, but without the first dimension"),
    3 => ("lengths",              "Optional lengths to represent a batch of `indices` and `values`.")
}

outputs!{SparseToDenseMask, 
    0 => ("output",               "Output tensor of the same type as `values` of shape `[len(lengths), len(mask)] + shape(default_value)` (if `lengths` is not provided the first dimension is omitted)"),
    1 => ("presence_mask",        "Bool tensor of shape `[len(lengths), len(mask)]` (if `lengths` is not provided the first dimension is omitted). True when a value for given id was present, false otherwise.")
}

args!{SparseToDenseMask, 
    0 => ("mask",                 "list(int) argument with desired ids on the 'dense' output dimension"),
    1 => ("return_presence_mask", "bool whether to return presence mask, false by default"),
    2 => ("max_skipped_indices",  "int argument representing the maximum number of invalid row ids that can be skipped before returning an error. 50 by default")
}

tensor_inference_function!{SparseToDenseMask, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          ArgumentHelper helper(def);
          auto mask = helper.template GetRepeatedArgument<int64_t>("mask");
          bool return_presence_mask = helper.template GetSingleArgument<bool>(
              "return_presence_mask", false);
          vector<TensorShape> out(1);

          if (in.size() == 4) {
            out[0].add_dims(in[3].dims(0));
          }
          out[0].add_dims(mask.size());
          for (const auto dim : in[2].dims()) {
            out[0].add_dims(dim);
          }
          out[0].set_data_type(in[2].data_type());

          if (return_presence_mask) {
            out.emplace_back();
            if (in.size() == 4) {
              out[1].add_dims(in[3].dims(0));
            }
            out[1].add_dims(mask.size());
            out[1].set_data_type(TensorProto::BOOL);
          }

          return out;
        */
    }
}

input_tags!{
    SparseToDenseMaskOp
    {
        Indices,
        Values,
        Default,
        Lengths
    }
}

output_tags!{
    SparseToDenseMaskOp
    {
        Outputvalue,
        Presencemask
    }
}

impl<Context> SparseToDenseMaskOp<Context> {

    const kMaxSkippedSparseIndices: u32 = 50;

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : SparseToDenseMaskBase<Context>(std::forward<Args>(args)...) 

        returnPresenceMask_ =
            this->template GetSingleArgument<bool>("return_presence_mask", false);
        maxSkippedRows_ = this->template GetSingleArgument<int32_t>(
            "max_skipped_indices", kMaxSkippedSparseIndices);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, Input(INDICES));
        */
    }
    
    #[inline] pub fn do_run_with_type<TInd>(&mut self) -> bool {
    
        todo!();
        /*
            auto& sparse_indices = Input(INDICES);
        CAFFE_ENFORCE_EQ(sparse_indices.dim(), 1);
        auto& sparse_values = Input(VALUES);
        CAFFE_ENFORCE_GE(sparse_values.dim(), 1);
        CAFFE_ENFORCE_EQ(sparse_indices.numel(), sparse_values.size(0));
        auto& default_value = Input(DEFAULT);
        CAFFE_ENFORCE_EQ(default_value.dim() + 1, sparse_values.dim());
        CAFFE_ENFORCE_EQ(default_value.numel(), sparse_values.size_from_dim(1));
        CAFFE_ENFORCE(sparse_values.dtype() == default_value.dtype());

        const TInd* sparse_indices_vec = sparse_indices.template data<TInd>();
        const char* sparse_values_vec =
            static_cast<const char*>(sparse_values.raw_data());
        const void* default_val = default_value.raw_data();

        int64_t block_size = default_value.numel();
        size_t block_nbytes = default_value.nbytes();

        const size_t cols = this->featuresCount_;
        int rows = -1;
        int32_t sparse_indices_length = sparse_indices.dim32(0);
        const int32_t* lengths_vec = nullptr;
        auto* output = Output(OUTPUTVALUE);
        Tensor* presence_mask = nullptr;
        if (returnPresenceMask_) {
          presence_mask = Output(PRESENCEMASK);
        }
        vector<int64_t> shape;
        if (InputSize() == 4) {
          auto& lengths = Input(LENGTHS);
          CAFFE_ENFORCE_EQ(lengths.dim(), 1);
          lengths_vec = lengths.template data<int32_t>();
          rows = lengths.dim32(0);
        }
        if (rows == -1) {
          // if the LENGTHS is not set, the output will be a vector
          rows = 1;
          lengths_vec = &sparse_indices_length;
        } else {
          shape.push_back(rows);
        }
        shape.push_back(cols);
        if (returnPresenceMask_) {
          presence_mask->Resize(shape);
        }
        shape.insert(
            shape.end(),
            default_value.sizes().begin(),
            default_value.sizes().end());
        output->Resize(shape);

        // init
        // TODO: consider unrolling CopyItems to make elemental types copy faster
        char* output_data =
            static_cast<char*>(output->raw_mutable_data(sparse_values.dtype()));
        for (int i = 0; i < cols * rows; i++) {
          context_.CopyItemsSameDevice(
              default_value.dtype(),
              block_size,
              default_val,
              output_data + i * block_nbytes);
        }
        bool* presence_mask_data = nullptr;
        if (returnPresenceMask_) {
          presence_mask_data = presence_mask->template mutable_data<bool>();
          math::Set<bool, Context>(
              rows * cols, false, presence_mask_data, &context_);
        }

        int64_t offset = 0;
        for (int r = 0; r < rows; r++) {
          bool skippedSparseIndex = false;
          for (int c = 0; c < lengths_vec[r]; c++) {
            const auto sparse_index = sparse_indices_vec[offset + c];
            if (sparse_index < 0 ||
                sparse_index >= TInd::max) {
              skippedSparseIndex = true;
              LOG(WARNING) << "Skipping invalid sparse index: " << sparse_index;
              continue;
            }
            int idx = this->getFeatureIdx(sparse_index);
            if (idx != -1) {
              context_.CopyItemsSameDevice(
                  sparse_values.dtype(),
                  block_size,
                  sparse_values_vec + (offset + c) * block_nbytes,
                  output_data + (r * cols + idx) * block_nbytes);
              if (returnPresenceMask_) {
                presence_mask_data[r * cols + idx] = true;
              }
            }
          }
          skippedRows_ += skippedSparseIndex;
          CAFFE_ENFORCE_LT(
              skippedRows_,
              maxSkippedRows_,
              "Too many rows with invalid sparse indices skipped");
          offset += lengths_vec[r];
        }

        return true;
        */
    }
}

/**
  | The output is the gradient of the input
  | value from SparseToDenseMask.
  | 
  | The gradient for default_value has
  | not been implemented.
  |
  */
pub struct SparseToDenseMaskGradientOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    base: SparseToDenseMaskBase<Context>,
}

num_inputs!{SparseToDenseMaskGradient, (2,3)}

num_outputs!{SparseToDenseMaskGradient, 1}

// TODO: enable the filler
disallow_input_fillers!{SparseToDenseMaskGradient}

input_tags!{
    SparseToDenseMaskGradientOp {
        Indices,
        Goutput,
        Lengths
    }
}

output_tags!{
    SparseToDenseMaskGradientOp {
        Gvalues
    }
}


impl<Context> SparseToDenseMaskGradientOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : SparseToDenseMaskBase<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, Input(INDICES));
        */
    }
    
    #[inline] pub fn do_run_with_type<TInd>(&mut self) -> bool {
    
        todo!();
        /*
            auto& sparse_indices = Input(INDICES);
        CAFFE_ENFORCE_EQ(sparse_indices.dim(), 1);
        auto& gradient_output = Input(GOUTPUT);

        int64_t block_size = gradient_output.size_from_dim(1);
        size_t block_nbytes = gradient_output.itemsize() * block_size;

        const size_t cols = this->featuresCount_;
        int rows = -1;
        int iter_offset = 1;
        int32_t default_length = sparse_indices.dim32(0);
        const int32_t* lengths_vec = nullptr;
        auto* output = Output(GVALUES);
        vector<int64_t> shape;
        if (InputSize() > LENGTHS) {
          // if the LENGTHS is set, the gradient_output has dim:
          // lengths * mask.size() * feature_dim
          auto& lengths = Input(LENGTHS);
          lengths_vec = lengths.template data<int32_t>();
          rows = lengths.dim32(0);
          CAFFE_ENFORCE_EQ(lengths.dim(), 1);
          CAFFE_ENFORCE_GE(gradient_output.dim(), 2);
          CAFFE_ENFORCE_EQ(gradient_output.size(0), rows);
          CAFFE_ENFORCE_EQ(gradient_output.size(1), cols);
          block_nbytes /= gradient_output.size(1);
          block_size /= gradient_output.size(1);
          iter_offset += 1;
        }
        if (rows == -1) {
          // if the LENGTHS is not set, the gradient_output has dim:
          // mask.size() * feature_dim
          rows = 1;
          lengths_vec = &default_length;
          CAFFE_ENFORCE_GE(gradient_output.dim(), 1);
          CAFFE_ENFORCE_EQ(gradient_output.size(0), cols);
        }
        shape.push_back(default_length);
        // insert feature_dim
        shape.insert(
            shape.end(),
            gradient_output.sizes().begin() + iter_offset,
            gradient_output.sizes().end());
        output->Resize(shape);

        const TInd* sparse_indices_vec = sparse_indices.template data<TInd>();
        const char* gradient_output_vec =
            static_cast<const char*>(gradient_output.raw_data());

        char* output_data =
            static_cast<char*>(output->raw_mutable_data(gradient_output.dtype()));
        memset(output_data, 0, output->nbytes());
        math::Set<char, Context>(
            default_length * gradient_output.itemsize(), 0, output_data, &context_);

        int32_t offset = 0;
        // SparseToDenseMask is not injective; gradient_used records
        // if the gradient is used for other input value from the same row
        vector<bool> gradient_used(cols, false);
        for (int r = 0; r < rows; r++) {
          std::fill(gradient_used.begin(), gradient_used.end(), false);
          for (int c = lengths_vec[r] - 1; c >= 0; c--) {
            int idx = this->getFeatureIdx(sparse_indices_vec[offset + c]);
            if (idx != -1 && !gradient_used[idx]) {
              gradient_used[idx] = true;
              context_.CopyItemsSameDevice(
                  gradient_output.dtype(),
                  block_size,
                  gradient_output_vec + (r * cols + idx) * block_nbytes,
                  output_data + (offset + c) * block_nbytes);
            }
          }
          offset += lengths_vec[r];
        }
        return true;
        */
    }
}

register_cpu_operator!{SparseToDenseMask,         SparseToDenseMaskOp<CPUContext>}

register_cpu_operator!{SparseToDenseMaskGradient, SparseToDenseMaskGradientOp<CPUContext>}

///--------------
pub struct GetSparseToDenseMaskGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSparseToDenseMaskGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> blob_names{I(0), GO(0)};

        // Add lengths blob if given
        if (def_.input_size() == 4) {
          blob_names.push_back(I(3));
        }
        return SingleGradientDef(
            "SparseToDenseMaskGradient", "", blob_names, vector<string>{GI(1)});
        */
    }
}

register_gradient!{SparseToDenseMask, GetSparseToDenseMaskGradient}

export_caffe2_op_to_c10_cpu!{SparseToDenseMask,
    "_caffe2::SparseToDenseMask(
        Tensor indices, 
        Tensor values, 
        Tensor default_value, 
        Tensor? lengths, 
        int[] mask, 
        bool? return_presence_mask = False, 
        int? max_skipped_indices = 50) -> (
        Tensor output, 
        Tensor presence_mask)",
    SparseToDenseMaskOp<CPUContext>
}
