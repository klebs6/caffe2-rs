crate::ix!();

pub struct VideoIOContext {
    work_buffersize:      i32,
    work_buffer:          AvDataPtr,

    /// for file mode
    input_file:           *mut libc::FILE,

    /// for memory mode
    input_buffer:         *const u8,
    input_buffer_size:    i32,
    offset:               i32, // default = 0
    ctx:                  *mut AVIOContext,
}

impl Drop for VideoIOContext {

    fn drop(&mut self) {
        todo!();
        /* 
        av_free(ctx_);
        if (inputFile_) {
          fclose(inputFile_);
        }
       */
    }
}

impl VideoIOContext {
    
    pub fn new_with_filename(fname: &String) -> Self {
        todo!();
        /*
            : workBuffersize_(VIO_BUFFER_SZ),
            workBuffer_((uint8_t*)av_malloc(workBuffersize_)),
            inputFile_(nullptr),
            inputBuffer_(nullptr),
            inputBufferSize_(0) 

        inputFile_ = fopen(fname.c_str(), "rb");
        if (inputFile_ == nullptr) {
          LOG(ERROR) << "Error opening video file " << fname;
          return;
        }
        ctx_ = avio_alloc_context(
            static_cast<unsigned char*>(workBuffer_.get()),
            workBuffersize_,
            0,
            this,
            &VideoIOContext::readFile,
            nullptr, // no write function
            &VideoIOContext::seekFile);
        */
    }
    
    pub fn new(buffer: *const u8, size: i32) -> Self {
        todo!();
        /*
            : workBuffersize_(VIO_BUFFER_SZ),
            workBuffer_((uint8_t*)av_malloc(workBuffersize_)),
            inputFile_(nullptr),
            inputBuffer_(buffer),
            inputBufferSize_(size) 
        ctx_ = avio_alloc_context(
            static_cast<unsigned char*>(workBuffer_.get()),
            workBuffersize_,
            0,
            this,
            &VideoIOContext::readMemory,
            nullptr, // no write function
            &VideoIOContext::seekMemory);
        */
    }
    
    #[inline] pub fn read(&mut self, buf: *mut u8, buf_size: i32) -> i32 {
        
        todo!();
        /*
            if (inputBuffer_) {
          return readMemory(this, buf, buf_size);
        } else if (inputFile_) {
          return readFile(this, buf, buf_size);
        } else {
          return -1;
        }
        */
    }
    
    #[inline] pub fn seek(&mut self, offset: i64, whence: i32) -> i64 {
        
        todo!();
        /*
            if (inputBuffer_) {
          return seekMemory(this, offset, whence);
        } else if (inputFile_) {
          return seekFile(this, offset, whence);
        } else {
          return -1;
        }
        */
    }
    
    #[inline] pub fn read_file(
        opaque:   *mut c_void,
        buf:      *mut u8,
        buf_size: i32) -> i32 {
        
        todo!();
        /*
            VideoIOContext* h = static_cast<VideoIOContext*>(opaque);
        if (feof(h->inputFile_)) {
          return AVERROR_EOF;
        }
        size_t ret = fread(buf, 1, buf_size, h->inputFile_);
        if (ret < buf_size) {
          if (ferror(h->inputFile_)) {
            return -1;
          }
        }
        return ret;
        */
    }
    
    #[inline] pub fn seek_file(
        opaque: *mut c_void,
        offset: i64,
        whence: i32) -> i64 {
        
        todo!();
        /*
            VideoIOContext* h = static_cast<VideoIOContext*>(opaque);
        switch (whence) {
          case SEEK_CUR: // from current position
          case SEEK_END: // from eof
          case SEEK_SET: // from beginning of file
            return fseek(h->inputFile_, static_cast<long>(offset), whence);
            break;
          case AVSEEK_SIZE:
            int64_t cur = ftell(h->inputFile_);
            fseek(h->inputFile_, 0L, SEEK_END);
            int64_t size = ftell(h->inputFile_);
            fseek(h->inputFile_, cur, SEEK_SET);
            return size;
        }

        return -1;
        */
    }
    
    #[inline] pub fn read_memory(
        opaque:   *mut c_void,
        buf:      *mut u8,
        buf_size: i32) -> i32 {

        todo!();
        /*
            VideoIOContext* h = static_cast<VideoIOContext*>(opaque);
        if (buf_size < 0) {
          return -1;
        }

        int reminder = h->inputBufferSize_ - h->offset_;
        int r = buf_size < reminder ? buf_size : reminder;
        if (r < 0) {
          return AVERROR_EOF;
        }

        memcpy(buf, h->inputBuffer_ + h->offset_, r);
        h->offset_ += r;
        return r;
        */
    }

    #[inline] pub fn seek_memory(
        opaque: *mut c_void,
        offset: i64,
        whence: i32) -> i64 {
        
        todo!();
        /*
            VideoIOContext* h = static_cast<VideoIOContext*>(opaque);
        switch (whence) {
          case SEEK_CUR: // from current position
            h->offset_ += offset;
            break;
          case SEEK_END: // from eof
            h->offset_ = h->inputBufferSize_ + offset;
            break;
          case SEEK_SET: // from beginning of file
            h->offset_ = offset;
            break;
          case AVSEEK_SIZE:
            return h->inputBufferSize_;
        }
        return h->offset_;
        */
    }
    
    #[inline] pub fn get_avio(&mut self) -> *mut AVIOContext {
        
        todo!();
        /*
            return ctx_;
        */
    }
}
