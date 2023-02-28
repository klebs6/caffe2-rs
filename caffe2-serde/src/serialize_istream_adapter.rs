crate::ix!();

use crate::ReadAdapterInterface;

/// this is a reader implemented by std::istream
pub struct IStreamAdapter<R: Read> {
    istream:  *mut BufReader<R>,
}

impl<R: Read> IStreamAdapter<R> {
    
    pub fn new(istream: *mut BufReader<R>) -> Self {
    
        todo!();
        /*
            : istream_(istream)
        */
    }

    #[inline] pub fn validate(&self, what: *const u8)  {
        
        todo!();
        /*
            if (!*istream_) {
        AT_ERROR("istream reader failed: ", what, ".");
      }
        */
    }
}

impl<R: Read> ReadAdapterInterface for IStreamAdapter<R> {
    
    #[inline] fn size(&self) -> usize {
        
        todo!();
        /*
            auto prev_pos = istream_->tellg();
      validate("getting the current position");
      istream_->seekg(0, istream_->end);
      validate("seeking to end");
      auto result = istream_->tellg();
      validate("getting size");
      istream_->seekg(prev_pos);
      validate("seeking to the original position");
      return result;
        */
    }
    
    #[inline] fn read(&self, 
        pos:  u64,
        buf:  *mut c_void,
        n:    usize,
        what: *const u8) -> usize {
        
        todo!();
        /*
            istream_->seekg(pos);
      validate(what);
      istream_->read(static_cast<char*>(buf), n);
      validate(what);
      return n;
        */
    }
}
    

impl<R: Read> Drop for IStreamAdapter<R> {
    fn drop(&mut self) {
        todo!();
        /*  */
    }
}
