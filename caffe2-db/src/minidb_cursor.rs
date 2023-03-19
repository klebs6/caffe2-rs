crate::ix!();

/**
  | Below, we provide a bare minimum database
  | "minidb" as a reference implementation as well
  | as a portable choice to store data.
  |
  | Note that the MiniDB classes are not exposed
  | via a header file - they should be created
  | directly via the db interface. See MiniDB for
  | details.
  */
pub struct MiniDBCursor<'a> {
    lock:      MutexGuard<'a, parking_lot::RawMutex>,
    file:      *mut libc::FILE,
    valid:     bool,
    key_len:   i32,
    key:       Vec<u8>,
    value_len: i32,
    value:     Vec<u8>,
}

impl<'a> MiniDBCursor<'a> {

    pub fn new(
        f: *mut libc::FILE, 
        mutex: *mut parking_lot::RawMutex) -> Self 
    {
        todo!();
        /*
            : file_(f), lock_(*mutex), valid_(true) 
        // We call Next() to read in the first entry.
        Next();
        */
    }
}

impl<'a> Cursor for MiniDBCursor<'a> {
    
    #[inline] fn seek(&mut self, key: &String)  {
        
        todo!();
        /*
            LOG(FATAL) << "MiniDB does not support seeking to a specific key.";
        */
    }
    
    #[inline] fn seek_to_first(&mut self)  {
        
        todo!();
        /*
            fseek(file_, 0, SEEK_SET);
        CAFFE_ENFORCE(!feof(file_), "Hmm, empty file?");
        // Read the first item.
        valid_ = true;
        Next();
        */
    }
    
    #[inline] fn next(&mut self)  {
        
        todo!();
        /*
            // First, read in the key and value length.
        if (fread(&key_len_, sizeof(int), 1, file_) == 0) {
          // Reaching EOF.
          VLOG(1) << "EOF reached, setting valid to false";
          valid_ = false;
          return;
        }
        CAFFE_ENFORCE_EQ(fread(&value_len_, sizeof(int), 1, file_), 1);
        CAFFE_ENFORCE_GT(key_len_, 0);
        CAFFE_ENFORCE_GT(value_len_, 0);
        // Resize if the key and value len is larger than the current one.
        if (key_len_ > (int)key_.size()) {
          key_.resize(key_len_);
        }
        if (value_len_ > (int)value_.size()) {
          value_.resize(value_len_);
        }
        // Actually read in the contents.
        CAFFE_ENFORCE_EQ(
            fread(key_.data(), sizeof(char), key_len_, file_), key_len_);
        CAFFE_ENFORCE_EQ(
            fread(value_.data(), sizeof(char), value_len_, file_), value_len_);
        // Note(Yangqing): as we read the file, the cursor naturally moves to the
        // beginning of the next entry.
        */
    }
    
    #[inline] fn key(&mut self) -> String {
        
        todo!();
        /*
            CAFFE_ENFORCE(valid_, "Cursor is at invalid location!");
        return string(key_.data(), key_len_);
        */
    }
    
    #[inline] fn value(&mut self) -> String {
        
        todo!();
        /*
            CAFFE_ENFORCE(valid_, "Cursor is at invalid location!");
        return string(value_.data(), value_len_);
        */
    }
    
    #[inline] fn valid(&mut self) -> bool {
        
        todo!();
        /*
            return valid_;
        */
    }
}
