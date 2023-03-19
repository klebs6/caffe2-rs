crate::ix!();

pub struct MiniDB {

    file: *mut libc::FILE,

    /**
      | access mutex makes sure we don't have
      | multiple cursors/transactions reading
      | the same file.
      |
      */
    file_access_mutex: parking_lot::RawMutex,

    mode: DatabaseMode,
}

impl Drop for MiniDB {

    fn drop(&mut self) {
        todo!();
        /* 
        Close();
       */
    }
}

impl DB for MiniDB {

    #[inline] fn close(&mut self)  {
        
        todo!();
        /*
            if (file_) {
          fclose(file_);
        }
        file_ = nullptr;
        */
    }
    
    #[inline] fn new_cursor(&mut self) -> Box<dyn Cursor> {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(this->mode_, READ);
        return make_unique<MiniDBCursor>(file_, &file_access_mutex_);
        */
    }
    
    #[inline] fn new_transaction(&mut self) -> Box<dyn Transaction> {
        
        todo!();
        /*
            CAFFE_ENFORCE(this->mode_ == NEW || this->mode_ == WRITE);
        return make_unique<MiniDBTransaction>(file_, &file_access_mutex_);
        */
    }
}

impl MiniDB {
    
    pub fn new(source: &String, mode: Mode) -> Self {
        todo!();
        /*
            : DB(source, mode), file_(nullptr) 

        switch (mode) {
          case NEW:
            file_ = fopen(source.c_str(), "wb");
            break;
          case WRITE:
            file_ = fopen(source.c_str(), "ab");
            fseek(file_, 0, SEEK_END);
            break;
          case READ:
            file_ = fopen(source.c_str(), "rb");
            break;
        }
        CAFFE_ENFORCE(file_, "Cannot open file: " + source);
        VLOG(1) << "Opened MiniDB " << source;
        */
    }
}

register_caffe2_db![MiniDB, MiniDB];
register_caffe2_db![minidb, MiniDB];
