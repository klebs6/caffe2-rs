crate::ix!();

use crate::{
    IStreamAdapter,
    ReadAdapterInterface
};

pub struct FileAdapter<R: Read> {
    file_stream:      std::fs::File,
    istream_adapter:  Box<IStreamAdapter<R>>,
}

impl<R: Read> FileAdapter<R> {
    
    pub fn new(file_name: &String) -> Self {
    
        todo!();
        /*
            file_stream_.open(file_name, std::ifstream::in | std::ifstream::binary);
      if (!file_stream_) {
        AT_ERROR("open file failed, file path: ", file_name);
      }
      istream_adapter_ = std::make_unique<IStreamAdapter>(&file_stream_);
        */
    }
}

impl<R: Read> ReadAdapterInterface for FileAdapter<R> {
    
    #[inline] fn size(&self) -> usize {
        
        todo!();
        /*
            return istream_adapter_->size();
        */
    }
    
    #[inline] fn read(&self, 
        pos:  u64,
        buf:  *mut c_void,
        n:    usize,
        what: *const u8) -> usize {

        todo!();
        /*
            return istream_adapter_->read(pos, buf, n, what);
        */
    }
}

impl<R: Read> Drop for FileAdapter<R> {
    fn drop(&mut self) {
        todo!();
        /*  */
    }
}
