crate::ix!();

pub struct MiniDBTransaction<'a> {
    file: *mut libc::FILE,
    lock: MutexGuard<'a, parking_lot::RawMutex>,
}

impl<'a> Drop for MiniDBTransaction<'a> {
    fn drop(&mut self) {
        todo!();
        /* 
        Commit();
       */
    }
}

impl<'a> MiniDBTransaction<'a> {

    pub fn new(
        f: *mut libc::FILE, 
        mutex: *mut parking_lot::RawMutex) -> Self 
    {
        todo!();
        /*
            : file_(f), lock_(*mutex)
        */
    }
    
}

impl<'a> Transaction for MiniDBTransaction<'a> {
    
    #[inline] fn put(
        &mut self, 
        key:   &String, 
        value: &String)  
    {
        todo!();
        /*
            int key_len = key.size();
        int value_len = value.size();
        CAFFE_ENFORCE_EQ(fwrite(&key_len, sizeof(int), 1, file_), 1);
        CAFFE_ENFORCE_EQ(fwrite(&value_len, sizeof(int), 1, file_), 1);
        CAFFE_ENFORCE_EQ(
            fwrite(key.c_str(), sizeof(char), key_len, file_), key_len);
        CAFFE_ENFORCE_EQ(
            fwrite(value.c_str(), sizeof(char), value_len, file_), value_len);
        */
    }
    
    #[inline] fn commit(&mut self)  {
        
        todo!();
        /*
            if (file_ != nullptr) {
          CAFFE_ENFORCE_EQ(fflush(file_), 0);
          file_ = nullptr;
        }
        */
    }
}
