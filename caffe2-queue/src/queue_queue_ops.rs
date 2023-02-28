crate::ix!();

use crate::{
    OperatorStorage,
    Workspace,
    Operator,
    OperatorDef,
    BlobsQueue,
    Blob,
};

pub struct CreateBlobsQueueOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,

    ws:    *mut Workspace, // default = nullptr
    name:  String,
}

register_cpu_operator!{CreateBlobsQueue, CreateBlobsQueueOp<CPUContext>}

register_cuda_operator!{CreateBlobsQueue,    CreateBlobsQueueOp<CUDAContext>}

no_gradient!{CreateBlobsQueue}

num_inputs!{CreateBlobsQueue, 0}

num_outputs!{CreateBlobsQueue, 1}

impl<Context> CreateBlobsQueueOp<Context> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            ws_(ws),
            name(operator_def.output().Get(0))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto capacity = GetSingleArgument("capacity", 1);
        const auto numBlobs = GetSingleArgument("num_blobs", 1);
        const auto enforceUniqueName =
            GetSingleArgument("enforce_unique_name", false);
        const auto fieldNames =
            OperatorStorage::template GetRepeatedArgument<std::string>("field_names");
        CAFFE_ENFORCE_EQ(this->OutputSize(), 1);
        auto queuePtr = Operator<Context>::Outputs()[0]
                            ->template GetMutable<std::shared_ptr<BlobsQueue>>();
        CAFFE_ENFORCE(queuePtr);
        *queuePtr = std::make_shared<BlobsQueue>(
            ws_, name, capacity, numBlobs, enforceUniqueName, fieldNames);
        return true;
        */
    }
}

///---------------------------------------
pub struct EnqueueBlobsOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
}

register_cpu_operator!{EnqueueBlobs, EnqueueBlobsOp<CPUContext>}

register_cuda_operator!{EnqueueBlobs,        EnqueueBlobsOp<CUDAContext>}

no_gradient!{EnqueueBlobs}

num_inputs_outputs!{EnqueueBlobs, 
    |inputs: i32, outputs: i32| {
        inputs >= 2 
            && outputs >= 1 
            && inputs == outputs + 1
    }
}

enforce_inplace!{EnqueueBlobs, /*[](int input, int output) { return input == output + 1; } */}

impl<Context> EnqueueBlobsOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE(InputSize() > 1);
        auto queue = Operator<Context>::Inputs()[0]
                         ->template Get<std::shared_ptr<BlobsQueue>>();
        CAFFE_ENFORCE(queue && OutputSize() == queue->getNumBlobs());
        return queue->blockingWrite(this->Outputs());
        */
    }
}

///Dequeue the blobs from queue.
pub struct DequeueBlobsOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,

    timeout_secs:  f32,
}

register_cpu_operator!{DequeueBlobs, DequeueBlobsOp<CPUContext>}

register_cuda_operator!{DequeueBlobs,        DequeueBlobsOp<CUDAContext>}

no_gradient!{DequeueBlobs}

num_inputs_outputs!{DequeueBlobs, 
    |inputs: i32, outputs: i32| {
        inputs == 1 && outputs >= 1
    }
}

inputs!{DequeueBlobs, 
    0 => ("queue", "The shared pointer for the BlobsQueue")
}

outputs!{DequeueBlobs, 
    0 => ("blob", "The blob to store the dequeued data")
}

args!{DequeueBlobs, 
    0 => ("timeout_secs", "Timeout in secs, default: no timeout")
}

impl<Context> DequeueBlobsOp<Context> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<Context>(operator_def, ws) 
        timeout_secs_ = OperatorStorage::GetSingleArgument<float>("timeout_secs", 0);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE(InputSize() == 1);
        auto queue =
            OperatorStorage::Inputs()[0]->template Get<std::shared_ptr<BlobsQueue>>();
        CAFFE_ENFORCE(queue && OutputSize() == queue->getNumBlobs());
        return queue->blockingRead(this->Outputs(), timeout_secs_);
        */
    }
}

///-------------------------------------
pub struct CloseBlobsQueueOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
}

register_cpu_operator!{CloseBlobsQueue, CloseBlobsQueueOp<CPUContext>}

register_cuda_operator!{CloseBlobsQueue,     CloseBlobsQueueOp<CUDAContext>}

no_gradient!{CloseBlobsQueue}

num_inputs!{CloseBlobsQueue, 1}

num_outputs!{CloseBlobsQueue, 0}

impl<Context> CloseBlobsQueueOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(InputSize(), 1);
        auto queue =
            OperatorStorage::Inputs()[0]->template Get<std::shared_ptr<BlobsQueue>>();
        CAFFE_ENFORCE(queue);
        queue->close();
        return true;
        */
    }
}

/**
  | Enqueue the blobs into queue. When the
  | queue is closed and full, the output
  | status will be set to true which can be
  | used as exit criteria for execution
  | step.
  | 
  | The 1st input is the queue and the last
  | output is the status. The rest are data
  | blobs.
  |
  */
pub struct SafeEnqueueBlobsOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS

    storage: OperatorStorage,
    context: Context,
}

register_cpu_operator!{SafeEnqueueBlobs, SafeEnqueueBlobsOp<CPUContext>}

register_cuda_operator!{SafeEnqueueBlobs,    SafeEnqueueBlobsOp<CUDAContext>}

no_gradient!{SafeEnqueueBlobs}

num_inputs_outputs!{SafeEnqueueBlobs, 
    |inputs: i32, outputs: i32| {
        inputs >= 2 && outputs >= 2 && inputs == outputs
    }
}

inputs!{SafeEnqueueBlobs, 
    0 => ("queue", "The shared pointer for the BlobsQueue")
}

enforce_inplace!{SafeEnqueueBlobs, 
    |inputs: i32, outputs: i32| {
        input == output + 1
    }
}

impl<Context> SafeEnqueueBlobsOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto queue = Operator<Context>::Inputs()[0]
                         ->template Get<std::shared_ptr<BlobsQueue>>();
        CAFFE_ENFORCE(queue);
        auto size = queue->getNumBlobs();
        CAFFE_ENFORCE(
            OutputSize() == size + 1,
            "Expected " + c10::to_string(size + 1) + ", " +
                " got: " + c10::to_string(size));
        bool status = queue->blockingWrite(this->Outputs());
        Output(size)->Resize();
        math::Set<bool, Context>(
            1, !status, Output(size)->template mutable_data<bool>(), &context_);
        return true;
        */
    }
    
    #[inline] pub fn cancel(&mut self)  {
        
        todo!();
        /*
            auto queue = Operator<Context>::Inputs()[0]
                         ->template Get<std::shared_ptr<BlobsQueue>>();
        queue->close();
        */
    }
}

/**
  | Dequeue the blobs from queue. When the
  | queue is closed and empty, the output
  | status will be set to true which can be
  | used as exit criteria for execution
  | step.
  | 
  | The 1st input is the queue and the last
  | output is the status. The rest are data
  | blobs.
  |
  */
pub struct SafeDequeueBlobsOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,

    num_records:  i32,
    blobs:        Vec<Blob>,
    blob_ptrs:    Vec<*mut Blob>,
}

register_cpu_operator!{SafeDequeueBlobs, SafeDequeueBlobsOp<CPUContext>}

register_cuda_operator!{SafeDequeueBlobs,    SafeDequeueBlobsOp<CUDAContext>}

no_gradient!{SafeDequeueBlobs}

num_inputs_outputs!{SafeDequeueBlobs, 
    |inputs: i32, outputs: i32| {
        inputs == 1 && outputs >= 2
    }
}

inputs!{SafeDequeueBlobs, 
    0 => ("queue", "The shared pointer for the BlobsQueue")
}

outputs!{SafeDequeueBlobs, 
    0 => ("blob",   "The blob to store the dequeued data"),
    1 => ("status", "Is set to 0/1 depending on the success of dequeue")
}

args!{SafeDequeueBlobs, 
    0 => ("num_records", "(default 1) If > 1, multiple records will be dequeued and tensors for each column will be concatenated. This requires all tensors in the records to be at least 1D, and to have the same inner dimensions.")
}

impl<Context> SafeDequeueBlobsOp<Context> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            numRecords_(OperatorStorage::GetSingleArgument<int>("num_records", 1)) 

        CAFFE_ENFORCE_GT(numRecords_, 0);
        */
    }
    
    #[inline] pub fn dequeue_many(&mut self, queue: &mut Arc<BlobsQueue>) -> bool {
        
        todo!();
        /*
            auto size = queue->getNumBlobs();

        if (blobs_.size() != size) {
          blobs_.resize(size);
          blobPtrs_.resize(size);
          for (int col = 0; col < size; ++col) {
            blobPtrs_.at(col) = &blobs_.at(col);
          }
        }

        const int kTensorGrowthPct = 40;
        for (int i = 0; i < numRecords_; ++i) {
          if (!queue->blockingRead(blobPtrs_)) {
            // if we read at least one record, status is still true
            return i > 0;
          }
          for (int col = 0; col < size; ++col) {
            auto* out = this->Output(col);
            const auto& in = blobPtrs_.at(col)->template Get<Tensor>();
            if (i == 0) {
              out->CopyFrom(in);
            } else {
              auto oldSize = out->numel();

              CAFFE_ENFORCE(
                  in.dim() > 0,
                  "Empty tensor to dequeue at column ",
                  col,
                  " within ",
                  size,
                  " total columns");

              out->Extend(in.sizes()[0], kTensorGrowthPct);
              auto* dst =
                  (char*)out->raw_mutable_data() + oldSize * in.dtype().itemsize();
              context_.template CopyItems<Context, Context>(
                  in.meta(), in.numel(), in.raw_data(), dst);
            }
          }
        }
        return true;
        */
    }
    
    #[inline] pub fn dequeue_one(&mut self, queue: &mut Arc<BlobsQueue>) -> bool {
        
        todo!();
        /*
            return queue->blockingRead(this->Outputs());
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE(InputSize() == 1);
        auto queue = Operator<Context>::Inputs()[0]
                         ->template Get<std::shared_ptr<BlobsQueue>>();
        CAFFE_ENFORCE(queue);

        auto size = queue->getNumBlobs();
        CAFFE_ENFORCE_EQ(OutputSize(), size + 1);

        bool status = numRecords_ > 1 ? dequeueMany(queue) : dequeueOne(queue);

        Output(size)->Resize();
        math::Set<bool, Context>(
            1, !status, Output(size)->template mutable_data<bool>(), &context_);
        return true;
        */
    }
    
    #[inline] pub fn cancel(&mut self)  {
        
        todo!();
        /*
            auto queue = Operator<Context>::Inputs()[0]
                         ->template Get<std::shared_ptr<BlobsQueue>>();
        queue->close();
        */
    }
}

/**
  | Dequeue the blobs from multiple queues.
  | When one of queues is closed and empty,
  | the output status will be set to true
  | which can be used as exit criteria for
  | execution step.
  | 
  | The 1st input is the queue and the last
  | output is the status. The rest are data
  | blobs.
  |
  */
pub struct WeightedSampleDequeueBlobsOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,

    cum_probs:       Vec<f32>,
    table_idx_blob:  i32,
}

register_cpu_operator!{WeightedSampleDequeueBlobs, WeightedSampleDequeueBlobsOp<CPUContext>}

no_gradient!{WeightedSampleDequeueBlobs}

num_inputs!{WeightedSampleDequeueBlobs, (1,INT_MAX)}

num_outputs!{WeightedSampleDequeueBlobs, (2,INT_MAX)}

args!{WeightedSampleDequeueBlobs, 
    0 => ("weights", "Weights for sampling from multiple queues"),
    1 => ("table_idx_blob", "The index of the blob (among the output blob list) that will be used to store the index of the table chosen to read the current batch.")
}

impl<Context> WeightedSampleDequeueBlobsOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            table_idx_blob_( OperatorStorage::GetSingleArgument<int>("table_idx_blob", -1)) 

        CAFFE_ENFORCE_LT(table_idx_blob_, OutputSize() - 1);
        vector<float> weights = OperatorStorage::GetRepeatedArgument<float>("weights");
        if (weights.empty()) {
          weights.resize(InputSize(), 1.0f);
        }
        CAFFE_ENFORCE_EQ(InputSize(), weights.size());

        float sum = accumulate(weights.begin(), weights.end(), 0.0f);
        CAFFE_ENFORCE(sum > 0.0f, "Sum of weights must be positive");
        cumProbs_.resize(weights.size());
        for (int i = 0; i < weights.size(); i++) {
          cumProbs_[i] = weights[i] / sum;
          CAFFE_ENFORCE_GE(
              cumProbs_[i], 0.0f, "Each probability must be non-negative");
        }
        std::partial_sum(cumProbs_.begin(), cumProbs_.end(), cumProbs_.begin());
        // Put last value to be 1.0001 to avoid numerical issues.
        cumProbs_.back() = 1.0001f;

        LOG(INFO) << "Dequeue weights: " << weights;
        LOG(INFO) << "cumProbs: " << cumProbs_;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            float r;
        math::RandUniform<float, Context>(1, 0.0f, 1.0f, &r, &context_);
        auto lb = lower_bound(cumProbs_.begin(), cumProbs_.end(), r);
        CAFFE_ENFORCE(lb != cumProbs_.end(), "Cannot find ", r, " in cumProbs_.");
        const int32_t idx = lb - cumProbs_.begin();
        auto queue = Operator<Context>::Inputs()[idx]
                         ->template Get<std::shared_ptr<BlobsQueue>>();

        CAFFE_ENFORCE(queue);
        auto size = queue->getNumBlobs();
        CAFFE_ENFORCE_EQ(OutputSize(), size + 1);
        bool status = queue->blockingRead(this->Outputs());
        if (table_idx_blob_ >= 0) {
          auto* table_idx_blob_out =
              Output(table_idx_blob_, {1}, at::dtype<int32_t>());
          int32_t* data = table_idx_blob_out->template mutable_data<int32_t>();
          data[0] = idx;
        }

        Output(size)->Resize();
        math::Set<bool, Context>(
            1, !status, Output(size)->template mutable_data<bool>(), &context_);
        return true;
        */
    }
}

caffe_known_type!{std::shared_ptr<BlobsQueue>}
