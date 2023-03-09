crate::ix!();

pub struct FileReader<'a> {
    buffer_size:  usize,
    fd:           i32,
    buffer:       Box<&'a [u8]>,
}

impl<'a> Drop for FileReader<'a> {

    fn drop(&mut self) {
        todo!();
        /* 
          if (fd_ >= 0) {
            close(fd_);
          }
         */
    }
}

impl<'a> FileReader<'a> {
    
    pub fn new(path: &String, buffer_size: Option<usize>) -> Self 
    {
        let buffer_size = buffer_size.unwrap_or(65536);
    
        todo!();
        /*
            : bufferSize_(bufferSize), buffer_(new char[bufferSize]) 

      fd_ = open(path.c_str(), O_RDONLY, 0777);
      if (fd_ < 0) {
        throw std::runtime_error(
            "Error opening file for reading: " + std::string(std::strerror(errno)) +
            " Path=" + path);
      }
        */
    }
}

impl<'a> StringProvider for FileReader<'a> {

    #[inline] fn reset(&mut self)  {
        
        todo!();
        /*
            if (lseek(fd_, 0, SEEK_SET) == -1) {
        throw std::runtime_error(
            "Error reseting file cursor: " + std::string(std::strerror(errno)));
      }
        */
    }
    
    #[inline] fn invoke(&mut self, range: &mut CharRange)  {
        
        todo!();
        /*
            char* buffer = buffer_.get();
      auto numRead = read(fd_, buffer, bufferSize_);
      if (numRead == -1) {
        throw std::runtime_error(
            "Error reading file: " + std::string(std::strerror(errno)));
      }
      if (numRead == 0) {
        range.start = nullptr;
        range.end = nullptr;
        return;
      }
      range.start = buffer;
      range.end = buffer + numRead;
        */
    }
}
