crate::ix!();

pub struct TimerStat {
    /*TODO
    CAFFE_STAT_CTOR(TimerStat);
    CAFFE_AVG_EXPORTED_STAT(time_ns);
    */
}

pub struct TimerInstance {
    running: bool,
    start:   std::time::Instant,
    stat:    TimerStat,
}

impl TimerInstance {

    pub fn new(name: &String) -> Self {
    
        todo!();
        /*
            : running_(false), stat_(name)
        */
    }
    
    #[inline] pub fn begin(&mut self)  {
        
        todo!();
        /*
            CAFFE_ENFORCE(!running_, "Called TimerBegin on an already running timer.");
        running_ = true;
        start_ = std::chrono::high_resolution_clock::now();
        */
    }
    
    #[inline] pub fn end(&mut self)  {
        
        todo!();
        /*
            CAFFE_ENFORCE(running_, "Called TimerEnd on a stopped timer.");
        using namespace std::chrono;
        auto duration = high_resolution_clock::now() - start_;
        auto nanos = duration_cast<nanoseconds>(duration).count();
        CAFFE_EVENT(stat_, time_ns, nanos);
        running_ = false;
        */
    }
    
    #[inline] pub fn get_ns(&mut self) -> i64 {
        
        todo!();
        /*
            CAFFE_ENFORCE(running_, "Called TimerGet on a stopped timer.");
        using namespace std::chrono;
        auto duration = high_resolution_clock::now() - start_;
        auto nanos = duration_cast<nanoseconds>(duration).count();
        return nanos;
        */
    }
}

caffe_known_type!{*mut TimerInstance}
