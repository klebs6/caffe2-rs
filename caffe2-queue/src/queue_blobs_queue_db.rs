crate::ix!();


#[inline] pub fn get_string_from_blob<'a>(blob: *mut Blob) -> &'a String {
    
    todo!();
    /*
        if (blob->template IsType<string>()) {
        return blob->template Get<string>();
      } else if (blob->template IsType<Tensor>()) {
        return *blob->template Get<Tensor>().template data<string>();
      } else {
        CAFFE_THROW("Unsupported Blob type");
      }
    */
}

///----------------------------------
pub struct BlobsQueueDBCursor {
    queue:             Arc<BlobsQueue>,
    key_blob_index:    i32,
    value_blob_index:  i32,
    timeout_secs:      f32,
    inited:            bool,
    key:               String,
    value:             String,
    valid:             bool,
}

impl BlobsQueueDBCursor {

    pub fn new(
        queue:            Arc<BlobsQueue>,
        key_blob_index:   i32,
        value_blob_index: i32,
        timeout_secs:     f32) -> Self {
    
        todo!();
        /*
            : queue_(queue),
            key_blob_index_(key_blob_index),
            value_blob_index_(value_blob_index),
            timeout_secs_(timeout_secs),
            inited_(false),
            valid_(false) 

        LOG(INFO) << "BlobsQueueDBCursor constructed";
        CAFFE_ENFORCE(queue_ != nullptr, "queue is null");
        CAFFE_ENFORCE(value_blob_index_ >= 0, "value_blob_index < 0");
        */
    }
}

impl Cursor for BlobsQueueDBCursor {
    
    #[inline] fn seek(&mut self, unused: &String)  {
        
        todo!();
        /*
            CAFFE_THROW("Seek is not supported.");
        */
    }
    
    #[inline] fn supports_seek(&mut self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    #[inline] fn seek_to_first(&mut self)  {
        
        todo!();
        /*
            // not applicable
        */
    }
    
    #[inline] fn next(&mut self)  {
        
        todo!();
        /*
            unique_ptr<Blob> blob = make_unique<Blob>();
        vector<Blob*> blob_vector{blob.get()};
        auto success = queue_->blockingRead(blob_vector, timeout_secs_);
        if (!success) {
          LOG(ERROR) << "Timed out reading from BlobsQueue or it is closed";
          valid_ = false;
          return;
        }

        if (key_blob_index_ >= 0) {
          key_ = GetStringFromBlob(blob_vector[key_blob_index_]);
        }
        value_ = GetStringFromBlob(blob_vector[value_blob_index_]);
        valid_ = true;
        */
    }
    
    #[inline] fn key(&mut self) -> String {
        
        todo!();
        /*
            if (!inited_) {
          Next();
          inited_ = true;
        }
        return key_;
        */
    }
    
    #[inline] fn value(&mut self) -> String {
        
        todo!();
        /*
            if (!inited_) {
          Next();
          inited_ = true;
        }
        return value_;
        */
    }
    
    #[inline] fn valid(&mut self) -> bool {
        
        todo!();
        /*
            return valid_;
        */
    }
}

///---------------------------
pub struct BlobsQueueDB {
    queue:             Arc<BlobsQueue>,
    key_blob_index:    i32,
    value_blob_index:  i32,
    timeout_secs:      f32,
}

impl BlobsQueueDB {

    pub fn new(
        source:           &String,
        mode:             DatabaseMode,
        queue:            Arc<BlobsQueue>,
        key_blob_index:   Option<i32>,
        value_blob_index: Option<i32>,
        timeout_secs:     Option<f32>) -> Self 
    {
        let key_blob_index:   i32 = key_blob_index.unwrap_or(-1);
        let value_blob_index: i32 = value_blob_index.unwrap_or(0);
        let timeout_secs:     f32 = timeout_secs.unwrap_or(0.0);

        todo!();
        /*
            : DB(source, mode),
            queue_(queue),
            key_blob_index_(key_blob_index),
            value_blob_index_(value_blob_index),
            timeout_secs_(timeout_secs) 

        LOG(INFO) << "BlobsQueueDB constructed";
        */
    }
}

impl DB for BlobsQueueDB {
    
    #[inline] fn close(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    #[inline] fn new_cursor(&mut self) -> Box<dyn Cursor> {
        
        todo!();
        /*
            return make_unique<BlobsQueueDBCursor>(
            queue_, key_blob_index_, value_blob_index_, timeout_secs_);
        */
    }
    
    #[inline] fn new_transaction(&mut self) -> Box<dyn Transaction> {
        
        todo!();
        /*
            CAFFE_THROW("Not implemented.");
        */
    }
}

impl Drop for BlobsQueueDB {
    fn drop(&mut self) {
        todo!();
        /* 
        Close();
       */
    }
}

/**
  | Create a DBReader from a BlobsQueue
  |
  */
pub struct CreateBlobsQueueDBOp {
    storage: OperatorStorage,
    context: CPUContext,
}

num_inputs!{CreateBlobsQueueDB, 1}

num_outputs!{CreateBlobsQueueDB, 1}

inputs!{CreateBlobsQueueDB, 
    0 => ("queue", "The shared pointer to a queue containing Blobs.")
}

outputs!{CreateBlobsQueueDB, 
    0 => ("reader", "The DBReader for the given BlobsQueue")
}

args!{CreateBlobsQueueDB, 
    0 => ("key_blob_index",    "(default: -1 (no key)) index of blob for DB key in the BlobsQueue."),
    1 => ("value_blob_index",  "(default: 0) index of blob for DB value in the BlobsQueue."),
    2 => ("timeout_secs",      "(default: 0.0 (no timeout)) Timeout in seconds for reading from the BlobsQueue.")
}

should_not_do_gradient!{CreateBlobsQueueDB}

impl CreateBlobsQueueDBOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            std::unique_ptr<db::DB> db = std::make_unique<BlobsQueueDB>(
            "",
            db::READ,
            OperatorStorage::Input<std::shared_ptr<BlobsQueue>>(0),
            OperatorStorage::template GetSingleArgument<int>("key_blob_index", -1),
            OperatorStorage::template GetSingleArgument<int>("value_blob_index", 0),
            OperatorStorage::template GetSingleArgument<float>("timeout_secs", 0.0));
        OperatorStorage::Output<db::DBReader>(0)->Open(std::move(db), 1, 0);
        return true;
        */
    }
}

register_cpu_operator!{CreateBlobsQueueDB, CreateBlobsQueueDBOp<CPUContext>}

#[cfg(caffe2_use_mkldnn)]
register_ideep_operator!{
    CreateBlobsQueueDB,
    IDEEPFallbackOp<CreateBlobsQueueDBOp<CPUContext>, SkipIndices<0>>
}

