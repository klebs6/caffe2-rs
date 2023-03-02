crate::ix!();

/**
 | Read the next batch of examples out of the given cursor,
 | idx blob, offset matrix and data blobs.
 |
 | Input(0) is a blob pointing to a TreeCursor,
 | Input(1) is a blob pointing to the shuffled idx
 | Input(2) is a blob pointing to the offset matrix and
 | [Input(3),... Input(num_fields)] a list of tensors containing the data for
 | each field of the dataset.
 |
 | ReadRandomBatch is thread safe.
 */
pub struct ReadRandomBatchOp {
    storage: OperatorStorage,
    context: CPUContext,
    batch_size:         i32,
    enforce_batch_size: bool,
    loop_over:          bool,
}

num_inputs!{ReadRandomBatch, (1,INT_MAX)}

num_outputs!{ReadRandomBatch, (1,INT_MAX)}

inputs!{ReadRandomBatch, 
    0 => ("cursor", "A blob containing a pointer to the cursor."),
    1 => ("idx", "idx with a shuffled order."),
    2 => ("offsetsmat", "offset matrix containing length offset info."),
    3 => ("dataset_field_0", "First dataset field")
}

outputs!{ReadRandomBatch, 
    0 => ("field_0", "Tensor containing the next batch for field 0.")
}

args!{ReadRandomBatch, 
    0 => ("batch_size", "Number of top-level entries to read."),
    1 => ("loop_over", "(bool) Repeat the dataset indefinitely")
}

impl ReadRandomBatchOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...),
            batchSize_(OperatorStorage::GetSingleArgument<int>("batch_size", 1)),
            enforceBatchSize_(
                OperatorStorage::GetSingleArgument<bool>("enforce_batch_size", false)),
            loopOver_(OperatorStorage::GetSingleArgument<bool>("loop_over", false))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& cursor = OperatorStorage::Input<std::unique_ptr<TreeCursor>>(0);
        auto& idxblob = Input(1);
        auto& offsetsmat = Input(2);
        CAFFE_ENFORCE(InputSize() == cursor->it.fields().size() + 3);
        auto idxvec = idxblob.template data<int64_t>();
        auto offsetdim = offsetsmat.sizes();
        // gather data
        std::vector<int64_t> outDim;
        int64_t idx;
        {
          std::lock_guard<std::mutex> lock(cursor->mutex_);
          cursor->offsets.resize(1);
          idx = cursor->offsets.at(0);
          // if we want to enforce batch size but we dont have a complete
          // batch, skip the last rows.
          if (enforceBatchSize_ && idx + batchSize_ > idxblob.numel()) {
            idx = idxblob.numel();
          }
          if (loopOver_ && idx >= idxblob.numel()) {
            cursor->offsets.at(0) = 0;
            idx = 0;
          }
          cursor->offsets.at(0) += batchSize_;
        }

        for (int i = 0; i < cursor->it.fields().size(); ++i) {
          auto lengthIdx = cursor->it.fields()[i].lengthFieldId + 1;
          auto& in = Input(i + 3);
          outDim = in.sizes().vec();
          outDim.at(0) = 0;
          auto idxbegin = idx;
          for (int j = 0; j < batchSize_; ++j) {
            if (idx >= idxblob.numel()) {
              break;
            }
            CAFFE_ENFORCE(
                (idxvec[idx] + 1) * offsetdim[1] + lengthIdx < offsetsmat.numel(),
                "Out of bound when trying to get elem from offsetsmat");
            auto offsetptr = offsetsmat.template data<TOffset>() +
                idxvec[idx] * offsetdim[1] + lengthIdx;
            auto offset = *offsetptr;
            auto size = *(offsetptr + offsetdim[1]) - offset;
            outDim.at(0) += size; // accumulate over the batch
            idx++;
          }
          idx = idxbegin; // reSet
          auto* out = Output(i);
          out->Resize(outDim);
          if (out->numel() == 0) {
            continue;
          }
          auto dst = static_cast<char*>(out->raw_mutable_data(in.dtype()));
          int block_size = in.numel() / in.size(0);
          auto block_bytesize = in.size_from_dim(1) * in.dtype().itemsize();
          CAFFE_ENFORCE(
              block_bytesize == in.nbytes() / in.size(0),
              "block_bytesize should be consistent with data dim");
          auto src_base = static_cast<const char*>(in.raw_data());
          int start = 0;
          for (int j = 0; j < batchSize_; ++j) {
            if (idx >= idxblob.numel()) {
              break;
            }
            auto offsetptr = offsetsmat.template data<TOffset>() +
                idxvec[idx] * offsetdim[1] + lengthIdx;
            auto offset = *offsetptr;
            auto size = *(offsetptr + offsetdim[1]) - offset;
            // copy data
            auto src = src_base + offset * block_bytesize;
            context_.CopyItemsSameDevice(
                in.dtype(), size * block_size, src, dst + start * block_bytesize);
            start += size;
            idx++;
          }
          idx = idxbegin; // reSet
        }
        return true;
        */
    }
}

