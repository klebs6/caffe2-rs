crate::ix!();


/**
  | A thread-safe, bounded, blocking queue.
  |
  | Modelled as a circular buffer.
  |
  | Containing blobs are owned by the workspace.
  |
  | On read, we swap out the underlying data for
  | the blob passed in for blobs
  */
pub struct BlobsQueue {

    closing:     AtomicBool, // default = false

    num_blobs:   usize,

    /// protects all variables in the class.
    mutex:       parking_lot::RawMutex,

    cv:          std::sync::Condvar,
    reader:      i64, // default = 0
    writer:      i64, // default = 0

    queue:       Vec<Vec<*mut Blob>>,
    name:        String,

    stats:       QueueStats,
}

pub struct QueueStats {
    /*
       CAFFE_STAT_CTOR(QueueStats);
       CAFFE_EXPORTED_STAT(queue_balance);
       CAFFE_EXPORTED_STAT(queue_dequeued_records);
       CAFFE_DETAILED_EXPORTED_STAT(queue_dequeued_bytes);
       CAFFE_AVG_EXPORTED_STAT(read_time_ns);
       CAFFE_AVG_EXPORTED_STAT(write_time_ns);
       */
}

impl Drop for BlobsQueue {
    fn drop(&mut self) {
        todo!();
        /* 
        close();
       */
    }
}

impl BlobsQueue {
    
    #[inline] pub fn get_num_blobs(&self) -> usize {
        
        todo!();
        /*
            return numBlobs_;
        */
    }
}

/// Constants for user tracepoints
pub const SDT_NONBLOCKING_OP: i32 = 0;
pub const SDT_BLOCKING_OP:    i32 = 1;
pub const SDT_TIMEOUT:        u64 = u64::MAX - 1;
pub const SDT_ABORT:          u64 = u64::MAX - 2;
pub const SDT_CANCEL:         u64 = u64::MAX - 3;

impl BlobsQueue {
    
    pub fn new(
        ws:                  *mut Workspace,
        queue_name:          &String,
        capacity:            usize,
        num_blobs:           usize,
        enforce_unique_name: bool,
        field_names:         &Vec<String>) -> Self {

        todo!();
        /*
            : numBlobs_(numBlobs), name_(queueName), stats_(queueName) 

          if (!fieldNames.empty()) {
            CAFFE_ENFORCE_EQ(
                fieldNames.size(), numBlobs, "Wrong number of fieldNames provided.");
            stats_.queue_dequeued_bytes.setDetails(fieldNames);
          }
          queue_.reserve(capacity);
          for (size_t i = 0; i < capacity; ++i) {
            std::vector<Blob*> blobs;
            blobs.reserve(numBlobs);
            for (size_t j = 0; j < numBlobs; ++j) {
              const auto blobName = queueName + "_" + to_string(i) + "_" + to_string(j);
              if (enforceUniqueName) {
                CAFFE_ENFORCE(
                    !ws->GetBlob(blobName),
                    "Queue internal blob already exists: ",
                    blobName);
              }
              blobs.push_back(ws->CreateBlob(blobName));
            }
            queue_.push_back(blobs);
          }
          DCHECK_EQ(queue_.size(), capacity);
        */
    }
    
    #[inline] pub fn blocking_read(&mut self, inputs: &Vec<*mut Blob>, timeout_secs: f32) -> bool {
        
        todo!();
        /*
            Timer readTimer;
      auto keeper = this->shared_from_this();
      const auto& name = name_.c_str();
      CAFFE_SDT(queue_read_start, name, (void*)this, SDT_BLOCKING_OP);
      std::unique_lock<std::mutex> g(mutex_);
      auto canRead = [this]() {
        CAFFE_ENFORCE_LE(reader_, writer_);
        return reader_ != writer_;
      };
      // Decrease queue balance before reading to indicate queue read pressure
      // is being increased (-ve queue balance indicates more reads than writes)
      CAFFE_EVENT(stats_, queue_balance, -1);
      if (timeout_secs > 0) {
        std::chrono::milliseconds timeout_ms(int(timeout_secs * 1000));
        cv_.wait_for(
            g, timeout_ms, [this, canRead]() { return closing_ || canRead(); });
      } else {
        cv_.wait(g, [this, canRead]() { return closing_ || canRead(); });
      }
      if (!canRead()) {
        if (timeout_secs > 0 && !closing_) {
          LOG(ERROR) << "DequeueBlobs timed out in " << timeout_secs << " secs";
          CAFFE_SDT(queue_read_end, name, (void*)this, SDT_TIMEOUT);
        } else {
          CAFFE_SDT(queue_read_end, name, (void*)this, SDT_CANCEL);
        }
        return false;
      }
      DCHECK(canRead());
      auto& result = queue_[reader_ % queue_.size()];
      CAFFE_ENFORCE(inputs.size() >= result.size());
      for (const auto i : c10::irange(result.size())) {
        auto bytes = BlobStat::sizeBytes(*result[i]);
        CAFFE_EVENT(stats_, queue_dequeued_bytes, bytes, i);
        using std::swap;
        swap(*(inputs[i]), *(result[i]));
      }
      CAFFE_SDT(queue_read_end, name, (void*)this, writer_ - reader_);
      CAFFE_EVENT(stats_, queue_dequeued_records);
      ++reader_;
      cv_.notify_all();
      CAFFE_EVENT(stats_, read_time_ns, readTimer.NanoSeconds());
      return true;
        */
    }
    
    #[inline] pub fn try_write(&mut self, inputs: &Vec<*mut Blob>) -> bool {
        
        todo!();
        /*
            Timer writeTimer;
      auto keeper = this->shared_from_this();
      const auto& name = name_.c_str();
      CAFFE_SDT(queue_write_start, name, (void*)this, SDT_NONBLOCKING_OP);
      std::unique_lock<std::mutex> g(mutex_);
      if (!canWrite()) {
        CAFFE_SDT(queue_write_end, name, (void*)this, SDT_ABORT);
        return false;
      }
      // Increase queue balance before writing to indicate queue write pressure is
      // being increased (+ve queue balance indicates more writes than reads)
      CAFFE_EVENT(stats_, queue_balance, 1);
      DCHECK(canWrite());
      doWrite(inputs);
      CAFFE_EVENT(stats_, write_time_ns, writeTimer.NanoSeconds());
      return true;
        */
    }
    
    #[inline] pub fn blocking_write(&mut self, inputs: &Vec<*mut Blob>) -> bool {
        
        todo!();
        /*
            Timer writeTimer;
      auto keeper = this->shared_from_this();
      const auto& name = name_.c_str();
      CAFFE_SDT(queue_write_start, name, (void*)this, SDT_BLOCKING_OP);
      std::unique_lock<std::mutex> g(mutex_);
      // Increase queue balance before writing to indicate queue write pressure is
      // being increased (+ve queue balance indicates more writes than reads)
      CAFFE_EVENT(stats_, queue_balance, 1);
      cv_.wait(g, [this]() { return closing_ || canWrite(); });
      if (!canWrite()) {
        CAFFE_SDT(queue_write_end, name, (void*)this, SDT_ABORT);
        return false;
      }
      DCHECK(canWrite());
      doWrite(inputs);
      CAFFE_EVENT(stats_, write_time_ns, writeTimer.NanoSeconds());
      return true;
        */
    }
    
    #[inline] pub fn close(&mut self)  {
        
        todo!();
        /*
            closing_ = true;

      std::lock_guard<std::mutex> g(mutex_);
      cv_.notify_all();
        */
    }
    
    #[inline] pub fn can_write(&mut self) -> bool {
        
        todo!();
        /*
            // writer is always within [reader, reader + size)
      // we can write if reader is within [reader, reader + size)
      CAFFE_ENFORCE_LE(reader_, writer_);
      CAFFE_ENFORCE_LE(writer_, reader_ + queue_.size());
      return writer_ != reader_ + queue_.size();
        */
    }
    
    #[inline] pub fn do_write(&mut self, inputs: &Vec<*mut Blob>)  {
        
        todo!();
        /*
            auto& result = queue_[writer_ % queue_.size()];
      CAFFE_ENFORCE(inputs.size() >= result.size());
      const auto& name = name_.c_str();
      for (const auto i : c10::irange(result.size())) {
        using std::swap;
        swap(*(inputs[i]), *(result[i]));
      }
      CAFFE_SDT(
          queue_write_end, name, (void*)this, reader_ + queue_.size() - writer_);
      ++writer_;
      cv_.notify_all();
        */
    }
}
