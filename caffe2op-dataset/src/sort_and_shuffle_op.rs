crate::ix!();

/**
 | Compute the sorted indices given a field index to
 | sort by and break the sorted indices into chunks
 | of shuffle_size * batch_size and shuffle each
 | chunk, finally we shuffle between batches. If
 | sort_by_field_idx is -1 we skip sort.
 |
 | For example, we have data sorted as
 | 1,2,3,4,5,6,7,8,9,10,11,12
 |
 | and batchSize = 2 and shuffleSize = 3, when we
 | shuffle we get:
 | [3,1,4,6,5,2] [12,10,11,8,9,7]
 |
 | After this we will shuffle among different batches
 | with size 2
 | [3,1],[4,6],[5,2],[12,10],[11,8],[9,7]
 |
 | We may end up with something like
 | [9,7],[5,2],[12,10],[4,6],[3,1],[11,8]
 |
 | Input(0) is a blob pointing to a TreeCursor, and
 | [Input(1),... Input(num_fields)] a list of tensors
 | containing the data for each field of the dataset.
 |
 | SortAndShuffle is thread safe.
 */
pub struct SortAndShuffleOp {
    storage: OperatorStorage,
    context: CPUContext,
    sort_by_field_idx: i32,
    batch_size:        i32,
    shuffle_size:      i32,
}

num_inputs!{SortAndShuffle, (1,INT_MAX)}

num_outputs!{SortAndShuffle, 1}

inputs!{SortAndShuffle, 
    0 => ("cursor", "A blob containing a pointer to the cursor."),
    1 => ("dataset_field_0", "First dataset field")
}

outputs!{SortAndShuffle, 
    0 => ("indices", "Tensor containing sorted indices.")
}

impl SortAndShuffleOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...),
            sort_by_field_idx_(
                OperatorStorage::GetSingleArgument<int>("sort_by_field_idx", 1)),
            batch_size_(OperatorStorage::GetSingleArgument<int>("batch_size", 1)),
            shuffle_size_(OperatorStorage::GetSingleArgument<int>("shuffle_size", 1))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& cursor = OperatorStorage::Input<std::unique_ptr<TreeCursor>>(0);
        CAFFE_ENFORCE(InputSize() == cursor->it.fields().size() + 1);
        CAFFE_ENFORCE(-1 <= sort_by_field_idx_);
        CAFFE_ENFORCE(cursor->it.fields().size() - sort_by_field_idx_ > 0);
        int size;
        if (sort_by_field_idx_ != -1) {
          size = Input(sort_by_field_idx_ + 1).sizes()[0];
        } else {
          size = Input(1).sizes()[0];
        }

        CAFFE_ENFORCE(
            batch_size_ > 0 && shuffle_size_ > 0 &&
            0 < batch_size_ * shuffle_size_);
        // adjust shuffle_size_ if it is too large
        if (batch_size_ * shuffle_size_ > size) {
          shuffle_size_ = size / batch_size_;
        }

        int num_batch = size / batch_size_;
        auto* out = Output(0);
        out->Resize(size);
        auto* out_data = out->template mutable_data<int64_t>();

        vector<int> shuffle_idx(size);
        iota(shuffle_idx.begin(), shuffle_idx.end(), 0);

        if (sort_by_field_idx_ != -1) {
          auto& sortblob = Input(sort_by_field_idx_ + 1);
          auto* sortdata = sortblob.data<int>();
          // must sort by a field at the root level
          CAFFE_ENFORCE(
              cursor->it.fields()[sort_by_field_idx_].lengthFieldId == -1);
          sort(shuffle_idx.begin(), shuffle_idx.end(), [&sortdata](int i1, int i2) {
            return sortdata[i1] < sortdata[i2];
          });
        }

        if (batch_size_ * shuffle_size_ > 1) {
          int offset = 0;
          while (offset + batch_size_ * shuffle_size_ < size) {
            std::shuffle(
                shuffle_idx.begin() + offset,
                shuffle_idx.begin() + offset + batch_size_ * shuffle_size_,
                std::default_random_engine());
            offset += batch_size_ * shuffle_size_;
          }
        }

        vector<int> batch_idx(num_batch);
        iota(batch_idx.begin(), batch_idx.end(), 0);
        std::shuffle(
            batch_idx.begin(), batch_idx.end(), std::default_random_engine());

        for (int i = 0; i < num_batch; i++) {
          std::copy(
              shuffle_idx.begin() + batch_idx[i] * batch_size_,
              shuffle_idx.begin() + (batch_idx[i] + 1) * batch_size_,
              out_data);
          out_data += batch_size_;
        }
        std::copy(
            shuffle_idx.begin() + num_batch * batch_size_,
            shuffle_idx.end(),
            out_data);

        return true;
        */
    }
}

