crate::ix!();

use crate::{
    OperatorStorage,
    CPUContext,
};

/**
  | Given DATA tensor of rank 1, and RANGES
  | tensor of rank 3, gather values corresponding
  | to each range into a separate output
  | tensor.
  | 
  | If the optional input KEY tensor is also
  | given, the output will be sorted by KEY
  | for each example.
  | 
  | RANGES dimensions description:
  | 
  | 1: represents list of examples within
  | a batch
  | 
  | 2: represents list features
  | 
  | 3: two values which are start and length
  | or a range (to be applied on DATA)
  | 
  | Each feature has fixed lengths which
  | are passed as lengths argument and a
  | separate tensor will be produced for
  | each feature.
  | 
  | i.e. DATA.dim(1) = len(lengths) = NumOuptuts.
  | 
  | Missing features (represented by empty
  | ranges) filled with default_value.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct GatherRangesToDenseOp<Context> {

    storage: OperatorStorage,
    context: Context,

    lengths:           Vec<i32>,
    total_ranges:      i64,
    empty_ranges:      Vec<i64>,
    mismatched_ranges: Vec<i64>,

    /**
      | To avoid false alarm due to insufficient
      | sample (e.g., first batch being mismatched
      | and causing 100% to be mismatched), use
      | a threshold to ensure enough samples are
      | gathered before decideding whether there
      | is an alarm or not.
      */
    min_observation:      i64,
    max_mismatched_ratio: f32,
    max_empty_ratio:      f32,
}

num_inputs!{GatherRangesToDense, (2,3)}

num_outputs!{GatherRangesToDense, (1,INT_MAX)}

inputs!{GatherRangesToDense, 
    0 => ("DATA",   "Tensor of rank 1."),
    1 => ("RANGES", "Tensor of int32/int64 ranges, of dims (N, M, 2). Where N is number of examples and M is a size of each example. Last dimension represents a range in the format (start, lengths)"),
    2 => ("KEY",    "Tensor of rank 1 and type int64.")
}

outputs!{GatherRangesToDense, 
    0 => ("OUTPUT", "1-D tensor of size sum of range lengths")
}

args!{GatherRangesToDense, 
    0 => ("lengths",                "Expected lengths for ranges"),
    1 => ("min_observation",        "The number of observations needed before deciding that the ratio of mismatched ranges is alarming, also determines whether an info sumarizing the empty and mismatch ratio will be printed at the end."),
    2 => ("max_mismatched_ratio",   "An error is raised when ratio of mismatched ranges exceeds this."),
    3 => ("max_empty_ratio",        "An error is raised when ratio of empty ranges exceeds this (default is 1, which means by default no error will be triggered).")
}

tensor_inference_function!{GatherRangesToDense, /* [](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      auto lengths = helper.GetRepeatedArgument<int>("lengths");
      CAFFE_ENFORCE_EQ(in[0].dims_size(), 1, "DATA should be 1-D tensor.");
      CAFFE_ENFORCE_EQ(in[1].dims_size(), 3, "RANGES should be 3-D tensor.");
      if (in.size() > 2) {
        CAFFE_ENFORCE_EQ(in[2].dims_size(), 1, "KEY should be 1-D tensor.");
      }
      CAFFE_ENFORCE_GT(lengths.size(), 0, "lengths should be non-empty.");
      std::vector<TensorShape> out(lengths.size());
      for (int i = 0; i < lengths.size(); ++i) {
        out[i].set_data_type(in[0].data_type());
        out[i].add_dims(in[1].dims(0));
        out[i].add_dims(lengths[i]);
      }
      return out;
    } */}

register_cpu_operator!{GatherRangesToDense, GatherRangesToDenseOp<CPUContext>}

no_gradient!{GatherRangesToDense}

input_tags!{
    GatherRangesToDenseOp {
        Data,
        Ranges,
        Key
    }
}

impl<Context> GatherRangesToDenseOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            lengths_(this->template GetRepeatedArgument<int>("lengths")),
            minObservation_(this->template GetSingleArgument<int64_t>(
                "min_observation",
                10000)),
            maxMismatchedRatio_(this->template GetSingleArgument<float>(
                "max_mismatched_ratio",
                0.01)),
            maxEmptyRatio_(
                this->template GetSingleArgument<float>("max_empty_ratio", 1.0)) 

        CAFFE_ENFORCE_GT(lengths_.size(), 0, "There has to be at least one length");
        for (auto length : lengths_) {
          CAFFE_ENFORCE_GT(length, 0, "Each length should be positive");
        }
        CAFFE_ENFORCE_GT(
            minObservation_, 0, "The number of observations is at least 1");
        // Initialize the empty and mismatch counter.
        for (int i = 0; i < OutputSize(); ++i) {
          emptyRanges_.push_back(0);
          mismatchedRanges_.push_back(0);
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, this->template Input<Tensor>(RANGES, CPU));
        */
    }

    #[inline] pub fn do_run_with_type<Index>(&mut self) -> bool {
        todo!();
        /*
            auto& data = Input(DATA);
            auto& ranges = Input(RANGES);
            CAFFE_ENFORCE_EQ(data.dim(), 1, "Data has to be 1-D");
            CAFFE_ENFORCE_EQ(ranges.dim(), 3, "Ranges has to be 3-D");
            if (InputSize() == 3) {
              auto& key = Input(KEY);
              CAFFE_ENFORCE_EQ(key.dim(), 1, "Key has to be 1-D");
              CAFFE_ENFORCE(
                  key.dtype().template Match<int64_t>(), "Key has to be type int64_t");
            }
            CAFFE_ENFORCE_EQ(
                ranges.size(1),
                lengths_.size(),
                "Number of ranges should match number of lengths");
            CAFFE_ENFORCE_EQ(
                ranges.size(1),
                OutputSize(),
                "Number of ranges should match number of outputs");
            CAFFE_ENFORCE_EQ(
                ranges.size(2), 2, "Ranges last dimension should be of size 2");

            auto* rawData = static_cast<const char*>(data.raw_data());
            auto* rangesData = ranges.template data<Index>();
            int rangesDataOffset = 0;
            auto itemsize = data.dtype().itemsize();

            auto batchSize = ranges.size(0);
            vector<int64_t> outputDims{batchSize, 0};
            vector<char*> outputRawData;
            for (int i = 0; i < OutputSize(); ++i) {
              auto* output = Output(i);
              outputDims[1] = lengths_[i];
              output->Resize(outputDims);
              char* ptr = static_cast<char*>(output->raw_mutable_data(data.dtype()));
              memset(ptr, 0, output->nbytes());
              outputRawData.push_back(ptr);
            }

            for (int i = 0; i < batchSize; ++i) {
              for (int j = 0; j < OutputSize(); ++j) {
                auto rangeStart = rangesData[rangesDataOffset++];
                auto rangeLength = rangesData[rangesDataOffset++];

                if (rangeLength == 0) {
                  // empty range, will be filled with zeros
                  emptyRanges_[j]++;
                  continue;
                }
                if (rangeLength != lengths_[j]) {
                  // Range lengths missmatch for output #, will be filled with zeros
                  // Note, empty ranges are not counted as mismatched because empty
                  // are more common and more tolerable.
                  mismatchedRanges_[j]++;
                  continue;
                }

                if (InputSize() == 2) {
                  context_.CopyItemsSameDevice(
                      data.dtype(),
                      rangeLength,
                      rawData + rangeStart * itemsize,
                      outputRawData[j] + i * itemsize * lengths_[j]);
                } else {
                  auto& key = Input(KEY);
                  auto* key_data = key.template data<int64_t>();
                  vector<std::pair<int64_t, const char*>> buffer;
                  for (int b_i = 0; b_i < rangeLength; ++b_i) {
                    int64_t one_key_item = key_data[rangeStart + b_i];
                    auto* one_data_item = rawData + (rangeStart + b_i) * itemsize;
                    buffer.emplace_back(one_key_item, one_data_item);
                  }
                  std::sort(
                      buffer.begin(),
                      buffer.end(),
                      [](const std::pair<int64_t, const char*>& left,
                         const std::pair<int64_t, const char*>& right) {
                        return left.first < right.first;
                      });
                  for (int b_i = 0; b_i < rangeLength; ++b_i) {
                    // Since this CPU only, directly copy to the destination.
                    std::memcpy(
                        outputRawData[j] + (i * lengths_[j] + b_i) * itemsize,
                        buffer[b_i].second,
                        itemsize);
                  }
                }
              }
            }

            CAFFE_ENFORCE_EQ(rangesDataOffset, ranges.numel());

            // Check whether the empty and mismatch ratio exceeded the threshold.
            totalRanges_ += batchSize;
            for (int j = 0; j < OutputSize(); ++j) {
              // Only check when the ratio is not set to allow all mismatches.
              if (maxMismatchedRatio_ < 1.0) {
                CAFFE_ENFORCE_GE(
                    std::max(totalRanges_, minObservation_) * maxMismatchedRatio_,
                    mismatchedRanges_[j],
                    "Ratio of range length mismatch for feature at index ",
                    j,
                    " is ",
                    (static_cast<double>(mismatchedRanges_[j]) /
                     static_cast<double>(totalRanges_)),
                    " (",
                    mismatchedRanges_[j],
                    "/",
                    totalRanges_,
                    ") which exceeds ",
                    maxMismatchedRatio_);
              }

              // Only check when the ratio is not set to allow all examples to be empty.
              if (maxEmptyRatio_ < 1.0) {
                CAFFE_ENFORCE_GE(
                    std::max(totalRanges_, minObservation_) * maxEmptyRatio_,
                    emptyRanges_[j],
                    "Ratio of empty ranges for feature at index ",
                    j,
                    " is ",
                    (static_cast<double>(emptyRanges_[j]) /
                     static_cast<double>(totalRanges_)),
                    " (",
                    emptyRanges_[j],
                    "/",
                    totalRanges_,
                    ") which exceeds ",
                    maxEmptyRatio_);
              }
            }

            return true;
        */
    }
}

impl<Context> Drop for GatherRangesToDenseOp<Context> {

    fn drop(&mut self) {

        todo!();

        /*
        if (totalRanges_ > minObservation_) {
          string debugString;
          if (this->has_debug_def()) {
            debugString =
                "Info from operator: " + ProtoDebugString(this->debug_def());
          } else {
            debugString = "Info from operator: no op def";
          }

          LOG(INFO) << "In GatherRangesToDenseOp:\n"
                    << "  Lifetime empty ranges for each feature is "
                    << emptyRanges_ << ".\n"
                    << "  Lifetime mismatched ranges for each feature is "
                    << mismatchedRanges_ << ".\n"
                    << "  With a total of " << totalRanges_ << " examples.\n"
                    << debugString;
        }
        */
    }
}

#[test] fn gather_ranges_to_dense_example1() {

    todo!();

    /*
    Example 1:
      DATA  = [1, 2, 3, 4, 5, 6, 7, 8]
      RANGES = [
        [
          [2, 4],
          [0, 2],
        ],
        [
          [0, 0],
          [6, 2],
        ]
      ]
      lengths = [4, 2]
      OUTPUT[0] = [[3, 4, 5, 6], [0, 0, 0, 0]]
      OUTPUT[1] = [[1, 2], [7, 8]]
    */
}

/**
  | Contrast Example 2 with Example 1. For
  | each data point per feature, the values
  | are sorted by the corresponding KEY.
  |
  */
#[test] fn gather_ranges_to_dense_example2() {

    todo!();

    /*
    Example 2 (with KEY):
    DATA  = [1, 2, 3, 4, 5, 6, 7, 8]
    KEY   = [0, 1, 3, 2, 1, 0, 1, 0]
    RANGES = [
      [
        [2, 4],
        [0, 2],
      ],
      [
        [0, 0],
        [6, 2],
      ]
    ]
    lengths = [4, 2]
    OUTPUT[0] = [[6, 5, 4, 3], [0, 0, 0, 0]]
    OUTPUT[1] = [[1, 2], [8, 7]]
    */
}

pub type GatherRangesToDenseCPUOp = GatherRangesToDenseOp<CPUContext>;

