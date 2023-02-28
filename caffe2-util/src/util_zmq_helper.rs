crate::ix!();

///------------------------------
pub struct ZmqContext {
    ptr: *mut c_void,
}

impl Drop for ZmqContext {
    fn drop(&mut self) {
        todo!();
        /* 
        int rc = zmq_ctx_destroy(ptr_);
        CAFFE_ENFORCE_EQ(rc, 0);
       */
    }
}

impl ZmqContext {
    
    pub fn new(io_threads: i32) -> Self {
        todo!();
        /*
            : ptr_(zmq_ctx_new()) 

        CAFFE_ENFORCE(ptr_ != nullptr, "Failed to create zmq context.");
        int rc = zmq_ctx_set(ptr_, ZMQ_IO_THREADS, io_threads);
        CAFFE_ENFORCE_EQ(rc, 0);
        rc = zmq_ctx_set(ptr_, ZMQ_MAX_SOCKETS, ZMQ_MAX_SOCKETS_DFLT);
        CAFFE_ENFORCE_EQ(rc, 0);
        */
    }
    
    #[inline] pub fn ptr(&mut self)  {
        
        todo!();
        /*
            return ptr_;
        */
    }
}

///-------------------------------------
pub struct ZmqMessage {
    msg: zmq_msg_t,
}

impl Default for ZmqMessage {
    
    fn default() -> Self {
        todo!();
        /*
            int rc = zmq_msg_init(&msg_);
        CAFFE_ENFORCE_EQ(rc, 0)
        */
    }
}

impl Drop for ZmqMessage {
    fn drop(&mut self) {
        todo!();
        /* 
        int rc = zmq_msg_close(&msg_);
        CAFFE_ENFORCE_EQ(rc, 0);
       */
    }
}

impl ZmqMessage {
    
    #[inline] pub fn msg(&mut self) -> *mut zmq_msg_t {
        
        todo!();
        /*
            return &msg_;
        */
    }
    
    #[inline] pub fn data(&mut self)  {
        
        todo!();
        /*
            return zmq_msg_data(&msg_);
        */
    }
    
    #[inline] pub fn size(&mut self) -> usize {
        
        todo!();
        /*
            return zmq_msg_size(&msg_);
        */
    }
}


///-----------------------------------

pub struct ZmqSocket {
    context: ZmqContext,
    ptr:     *mut c_void,
}

impl Drop for ZmqSocket {
    fn drop(&mut self) {
        todo!();
        /* 
        int rc = zmq_close(ptr_);
        CAFFE_ENFORCE_EQ(rc, 0);
       */
    }
}

impl ZmqSocket {
    
    pub fn new(ty: i32) -> Self {
        todo!();
        /*
            : context_(1), ptr_(zmq_socket(context_.ptr(), type)) 

        CAFFE_ENFORCE(ptr_ != nullptr, "Failed to create zmq socket.");
        */
    }
    
    #[inline] pub fn bind(&mut self, addr: &String)  {
        
        todo!();
        /*
            int rc = zmq_bind(ptr_, addr.c_str());
        CAFFE_ENFORCE_EQ(rc, 0);
        */
    }
    
    #[inline] pub fn unbind(&mut self, addr: &String)  {
        
        todo!();
        /*
            int rc = zmq_unbind(ptr_, addr.c_str());
        CAFFE_ENFORCE_EQ(rc, 0);
        */
    }
    
    #[inline] pub fn connect(&mut self, addr: &String)  {
        
        todo!();
        /*
            int rc = zmq_connect(ptr_, addr.c_str());
        CAFFE_ENFORCE_EQ(rc, 0);
        */
    }
    
    #[inline] pub fn disconnect(&mut self, addr: &String)  {
        
        todo!();
        /*
            int rc = zmq_disconnect(ptr_, addr.c_str());
        CAFFE_ENFORCE_EQ(rc, 0);
        */
    }
    
    #[inline] pub fn send(&mut self, msg: &String, flags: i32) -> i32 {
        
        todo!();
        /*
            int nbytes = zmq_send(ptr_, msg.c_str(), msg.size(), flags);
        if (nbytes) {
          return nbytes;
        } else if (zmq_errno() == EAGAIN) {
          return 0;
        } else {
          LOG(FATAL) << "Cannot send zmq message. Error number: "
                          << zmq_errno();
          return 0;
        }
        */
    }
    
    #[inline] pub fn send_till_success(&mut self, msg: &String, flags: i32) -> i32 {
        
        todo!();
        /*
            CAFFE_ENFORCE(msg.size(), "You cannot send an empty message.");
        int nbytes = 0;
        do {
          nbytes = Send(msg, flags);
        } while (nbytes == 0);
        return nbytes;
        */
    }
    
    #[inline] pub fn recv(&mut self, msg: *mut ZmqMessage) -> i32 {
        
        todo!();
        /*
            int nbytes = zmq_msg_recv(msg->msg(), ptr_, 0);
        if (nbytes >= 0) {
          return nbytes;
        } else if (zmq_errno() == EAGAIN || zmq_errno() == EINTR) {
          return 0;
        } else {
          LOG(FATAL) << "Cannot receive zmq message. Error number: "
                          << zmq_errno();
          return 0;
        }
        */
    }
    
    #[inline] pub fn recv_till_success(&mut self, msg: *mut ZmqMessage) -> i32 {
        
        todo!();
        /*
            int nbytes = 0;
        do {
          nbytes = Recv(msg);
        } while (nbytes == 0);
        return nbytes;
        */
    }
}
