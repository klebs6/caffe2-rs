crate::ix!();

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

