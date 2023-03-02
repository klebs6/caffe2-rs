crate::ix!();

/**
  | Read the next batch of examples out of
  | the given cursor and data blobs.
  | 
  | Input(0) is a blob pointing to a TreeCursor,
  | and [Input(1),... Input(num_fields)]
  | a list of tensors containing the data
  | for each field of the dataset.
  | 
  | ReadNextBatch is thread safe.
  |
  */
pub struct ReadNextBatchOp {
    storage:            OperatorStorage,
    context:            CPUContext,
    batch_size:         i32,
    enforce_batch_size: bool,
}

num_inputs!{ReadNextBatch, (1,INT_MAX)}

num_outputs!{ReadNextBatch, (1,INT_MAX)}

inputs!{ReadNextBatch, 
    0 => ("cursor", "A blob containing a pointer to the cursor."),
    1 => ("dataset_field_0", "First dataset field")
}

outputs!{ReadNextBatch, 
    0 => ("field_0", "Tensor containing the next batch for field 0.")
}

args!{ReadNextBatch, 
    0 => ("batch_size", "Number of top-level entries to read.")
}

impl ReadNextBatchOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...),
            batchSize_(OperatorStorage::GetSingleArgument<int>("batch_size", 1)),
            enforceBatchSize_(OperatorStorage::GetSingleArgument<bool>(
                "enforce_batch_size",
                false))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& cursor = OperatorStorage::Input<std::unique_ptr<TreeCursor>>(0);
        CAFFE_ENFORCE(InputSize() == cursor->it.fields().size() + 1);
        std::vector<const TLength*> lengths;
        std::vector<TOffset> limits;
        std::vector<TOffset> sizes;
        std::vector<TOffset> offsets;
        TLength lenZero = 0;
        sizes.resize(cursor->it.numOffsetFields());
        // gather length data
        lengths.resize(cursor->it.numLengthFields());
        for (int i = 0; i < lengths.size(); ++i) {
          auto& a = Input(cursor->it.lengthField(i).id + 1);
          if (a.numel() > 0) {
            lengths[i] = a.data<int>();
          } else {
            lengths[i] = &lenZero;
          }
        }
        // gather size limits
        limits.assign(sizes.size(), TOffset::max);
        for (int i = 0; i < cursor->it.fields().size(); ++i) {
          int lengthFieldIdx = cursor->it.fields()[i].lengthFieldId + 1;
          limits[lengthFieldIdx] =
              std::min(limits[lengthFieldIdx], (TOffset)Input(i + 1).sizes()[0]);
        }
        // advance cursor
        {
          std::lock_guard<std::mutex> lock(cursor->mutex_);
          if (cursor->offsets.empty()) {
            cursor->offsets.assign(sizes.size(), 0);
          }
          offsets = cursor->offsets;
          cursor->it.advance(lengths, cursor->offsets, sizes, limits, batchSize_);
          if (enforceBatchSize_ && sizes[0] < batchSize_) {
            // if we enforce batch_size but don't have enough rows left to
            // complete a full batch, return empty for all columns.
            // This signals end of dataset to the caller.
            sizes.assign(sizes.size(), 0);
          }
        }
        // gather data
        std::vector<int64_t> outDim;
        for (int i = 0; i < cursor->it.fields().size(); ++i) {
          auto lengthIdx = cursor->it.fields()[i].lengthFieldId + 1;
          auto size = sizes[lengthIdx];
          auto offset = offsets[lengthIdx];
          auto& in = Input(i + 1);
          auto innerSize = in.size_from_dim(1);
          outDim = in.sizes().vec();
          outDim[0] = size;
          auto* out = Output(i);
          out->Resize(outDim);
          void* src =
              (char*)in.raw_data() + offset * innerSize * in.dtype().itemsize();
          void* dst = out->raw_mutable_data(in.dtype()); // create the tensor
          if (out->numel() == 0) {
            continue;
          }
          context_.CopyItemsSameDevice(in.dtype(), out->numel(), src, dst);
        }
        return true;
        */
    }
}

