crate::ix!();

/**
  | TODO: This is a very naive implementation with
  | a single mutex. We can do the atomic index
  | + circular queue optimizations or pull
  | something more heavy-weight later
  */
pub struct RebatchingQueue {
    capacity:     usize,
    num_blobs:    usize,
    mutex:        parking_lot::RawMutex,
    is_closed:    bool, // default = false
    head:         u64,  // default = 0
    tail:         u64,  // default = 0
    cv_empty:     std::sync::Condvar,
    cv_overflow:  std::sync::Condvar,
    queue:        Vec<Vec<TensorCPU>>,
}

/**
  | This concat function will always create
  | a new first dimension to concat
  |
  */
#[inline] pub fn concat(
    context: &mut CPUContext,
    inputs:  &Vec<Vec<TensorCPU>>,
    outputs: &Vec<*mut TensorCPU>)  {
    
    todo!();
    /*
        CAFFE_ENFORCE(!inputs.empty());

      const auto& inputZero = inputs[0];
      const auto numTensors = inputZero.size();
      const auto numRows = inputs.size();

      // Precompute the output sizes to avoid resizing
      std::vector<std::vector<int64_t>> outputDims(numTensors);

      for (size_t i = 0; i < numTensors; ++i) {
        SmartTensorPrinter::PrintTensor(inputZero.at(i));
        outputDims[i] = inputZero.at(i).sizes().vec();
        outputDims[i].insert(outputDims[i].begin(), numRows);
      }

      // Resize to the final output size
      std::vector<void*> destinations(numTensors);
      for (size_t i = 0; i < numTensors; ++i) {
        outputs[i]->Resize(outputDims[i]);
        destinations[i] = outputs[i]->raw_mutable_data(inputZero[i].meta());
      }

      for (size_t i = 0; i < numRows; ++i) {
        CAFFE_ENFORCE_EQ(inputs[i].size(), numTensors);

        for (int j = 0; j < numTensors; ++j) {
          const auto& input = inputs[i][j];

          CAFFE_ENFORCE(inputZero[j].meta() == input.dtype());
          CAFFE_ENFORCE_EQ(inputZero[j].itemsize(), input.itemsize());
          CAFFE_ENFORCE_EQ(inputZero[j].ndim(), input.dim());
          for (int k = 0; k < input.dim(); ++k) {
            CAFFE_ENFORCE_EQ(input.sizes()[k], inputZero[j].size(k));
          }

          // Skip empty tensors
          if (input.numel() == 0) {
            continue;
          }

          context.CopyItemsToCPU(
              input.dtype(),
              input.numel(),
              input.raw_data() /* src */,
              destinations[j] /* dst */
          );

          destinations[j] =
              (char*)destinations[j] + input.numel() * input.itemsize();
        }
      }
    */
}

#[inline] pub fn split(
    context: &mut CPUContext, 
    inputs:  &Vec<*const TensorCPU>) -> Vec<Vec<TensorCPU>> {
    
    todo!();
    /*
        CAFFE_ENFORCE(!inputs.empty());

      const auto outputSize = inputs[0]->sizes().at(0);
      std::vector<std::vector<TensorCPU>> outputs(outputSize);

      for (const auto* inputPtr : inputs) {
        CAFFE_ENFORCE(inputPtr);

        const auto& input = *inputPtr;
        const auto innerSize = input.size_from_dim(1);
        const auto itemSize = input.dtype().itemsize();

        auto outputDims = input.sizes().vec();
        CAFFE_ENFORCE(!outputDims.empty());
        outputDims.erase(outputDims.begin());
        CAFFE_ENFORCE_EQ(input.sizes().at(0), outputSize);

        for (int i = 0; i < outputSize; ++i) {
          outputs[i].push_back(Tensor(outputDims, CPU));
          context.CopyItemsToCPU(
              input.dtype(),
              innerSize,
              (char*)input.raw_data() + i * innerSize * itemSize /* src */,
              outputs[i].back().raw_mutable_data(input.dtype()) /* dst */);
        }
      }

      return outputs;
    */
}

impl Drop for RebatchingQueue {
    fn drop(&mut self) {
        todo!();
        /* 
      close();
         */
    }
}

impl RebatchingQueue {
    
    pub fn new(capacity: usize, num_blobs: usize) -> Self {
    
        todo!();
        /*
            : capacity_(capacity), numBlobs_(numBlobs), queue_(capacity)
        */
    }
    
    #[inline] pub fn can_read(&self) -> bool {
        
        todo!();
        /*
            return tail_ < head_;
        */
    }
    
    #[inline] pub fn dequeue(&mut self, 
        context:      &mut CPUContext,
        num_elements: usize,
        outputs:      &Vec<*mut TensorCPU>) -> bool {
        
        todo!();
        /*
            std::vector<std::vector<TensorCPU>> results;
      results.reserve(numElements);

      for (;;) {
        if (results.size() == numElements) {
          break;
        }

        {
          std::unique_lock<std::mutex> lock(mutex_);

          cvEmpty_.wait(lock, [this] { return canRead() || isClosed_; });

          // We only want to stop reading if the queue is empty and closed
          if (!canRead() && isClosed_) {
            break;
          }

          do {
            results.push_back(std::move(queue_[tail_++ % capacity()]));
          } while (canRead() && results.size() < numElements);
        }

        if (numElements == 1) {
          cvOverflow_.notify_one();
        } else {
          cvOverflow_.notify_all();
        }
      }

      if (results.empty()) {
        return false;
      }

      concat(context, results, outputs);

      return true;
        */
    }
    
    #[inline] pub fn can_write(&self) -> bool {
        
        todo!();
        /*
            return tail_ + capacity() > head_;
        */
    }
    
    #[inline] pub fn enqueue_one(
        &mut self, 
        context: &mut CPUContext, 
        inputs: &Vec<*const TensorCPU>) -> bool 
    {
        
        todo!();
        /*
            std::vector<std::vector<TensorCPU>> splittedInputs;
      splittedInputs.emplace_back();
      auto& tensorVector = splittedInputs.back();
      tensorVector.reserve(inputs.size());
      for (const auto* tensorPtr : inputs) {
        tensorVector.push_back(tensorPtr->Clone());
      }

      return enqueue(std::move(splittedInputs));
        */
    }
    
    #[inline] pub fn enqueue_many(
        &mut self, 
        context: &mut CPUContext, 
        inputs: &Vec<*const TensorCPU>) -> bool 
    {
        todo!();
        /*
            CAFFE_ENFORCE_EQ(numBlobs_, inputs.size());

      std::vector<std::vector<TensorCPU>> splittedInputs;
      splittedInputs = split(context, inputs);
      return enqueue(std::move(splittedInputs));
        */
    }
    
    #[inline] pub fn enqueue(&mut self, splitted_inputs: Vec<Vec<TensorCPU>>) -> bool {
        
        todo!();
        /*
            int idx = 0;
      for (;;) {
        if (idx >= splittedInputs.size()) {
          break;
        }

        {
          std::unique_lock<std::mutex> lock(mutex_);

          cvOverflow_.wait(lock, [this] { return canWrite() || isClosed_; });

          if (isClosed_) {
            // If we are here it means that we didn't apply the entire batch and if
            // we get closed in the middle of enquing we treat it as a non-success.
            return false;
          }

          do {
            queue_[head_++ % capacity()] = std::move(splittedInputs[idx++]);
          } while (canWrite() && idx < splittedInputs.size());
        }

        cvEmpty_.notify_all();
      }

      return true;
        */
    }
    
    #[inline] pub fn capacity(&self) -> usize {
        
        todo!();
        /*
            return capacity_;
        */
    }
    
    #[inline] pub fn num_blobs(&self) -> usize {
        
        todo!();
        /*
            return numBlobs_;
        */
    }
    
    #[inline] pub fn is_closed(&self) -> bool {
        
        todo!();
        /*
            std::lock_guard<std::mutex> g(mutex_);
      return isClosed_;
        */
    }
    
    #[inline] pub fn close(&mut self)  {
        
        todo!();
        /*
            {
        std::lock_guard<std::mutex> g(mutex_);
        isClosed_ = true;
      }

      cvEmpty_.notify_all();
      cvOverflow_.notify_all();
        */
    }
}
