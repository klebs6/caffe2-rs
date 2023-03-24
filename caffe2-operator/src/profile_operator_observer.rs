crate::ix!();

pub struct ProfileOperatorObserver {
    counter: ProfileCounter,
    base:    ObserverBase<OperatorStorage>,

    net_observer: *mut ProfileObserver,

    /**
      | Needed because this is not visible in
      | RNN Executor
      |
      */
    net_position: i32,

    rnn_order: i32, // default = OperatorStorage::kNoNetPositionSet;
}

impl ProfileOperatorObserver {

    pub fn new_from_subject_and_net_observer(
        subject:      *mut OperatorStorage, 
        net_observer: *mut ProfileObserver) -> Self {
        todo!();
        /*
            : ObserverBase<OperatorStorage>(subject), netObserver_(netObserver) 

        if (subject) {
          net_position_ = subject->net_position();
        }
        */
    }
    
    pub fn new(
        subject: *mut OperatorStorage,
        net_observer: *mut ProfileObserver,
        net_position: i32,
        rnn_order: i32) -> Self 
    {
        todo!();
        /*
            : ProfileOperatorObserver(subject, netObserver) 
        net_position_ = net_position;
        rnn_order_ = rnn_order;
        */
    }
    
    #[inline] pub fn get_id(&self) -> String {
        
        todo!();
        /*
            std::stringstream ss;
        ss << net_position_;
        if (rnn_order_ != OperatorStorage::kNoNetPositionSet) {
          ss << "-" << rnn_order_;
        }
        return ss.str();
        */
    }
    
    #[inline] pub fn dump(&self)  {
        
        todo!();
        /*
            static std::mutex loggingMutex;
      std::lock_guard<std::mutex> lock(loggingMutex);

      LOG(INFO) << "--------- Starting operator " << subject_->debug_def().type()
                << " op#" << getId() << " ---------";
      for (int i = 0; i < subject_->InputSize(); ++i) {
        if (subject_->InputIsTensorType(i, CPU)) {
          const auto& tensor = subject_->Input<Tensor>(i, CPU);
          const auto& name = subject_->debug_def().input(i);
          TensorPrinter printer(name);
          LOG(INFO) << "Input " << i << ": " << printer.MetaStr(tensor);
        } else if (subject_->InputIsTensorType(i, CUDA)) {
          const auto& tensor = subject_->Input<Tensor>(i, CUDA);
          const auto& name = subject_->debug_def().input(i);
          TensorPrinter printer(name);
          LOG(INFO) << "Input " << i << ": " << printer.MetaStr(tensor);
        }
      }

      int a = 0;
      for (const auto& arg : subject_->debug_def().arg()) {
        LOG(INFO) << "Argument " << a << ": " << arg.ShortDebugString();
        ++a;
      }

      for (int o = 0; o < subject_->OutputSize(); ++o) {
        if (subject_->OutputIsTensorType(o, CPU)) {
          auto* tensor = subject_->Output<Tensor>(o, CPU);
          const auto& name = subject_->debug_def().output(o);
          TensorPrinter printer(name);
          LOG(INFO) << "Output " << o << ": " << printer.MetaStr(*tensor);
        } else if (subject_->OutputIsTensorType(o, CUDA)) {
          auto* tensor = subject_->Output<Tensor>(o, CUDA);
          const auto& name = subject_->debug_def().output(o);
          TensorPrinter printer(name);
          LOG(INFO) << "Output " << o << ": " << printer.MetaStr(*tensor);
        }
      }

      LOG(INFO) << "--------- Finished operator " << subject_->debug_def().type()
                << " in " << run_time_ << " ms ---------";
        */
    }
    
    #[inline] pub fn start(&mut self)  {
        
        todo!();
        /*
            start_time_ = timer_.MilliSeconds();
        */
    }
    
    #[inline] pub fn stop(&mut self)  {
        
        todo!();
        /*
            run_time_ = timer_.MilliSeconds() - start_time_;
      Dump();
        */
    }
    
    #[inline] pub fn rnn_copy(
        &self, 
        subject: *mut OperatorStorage,
        rnn_order: i32) -> Box<ObserverBase<OperatorStorage>> 
    {
        todo!();
        /*
            return std::unique_ptr<ObserverBase<OperatorStorage>>(
          new ProfileOperatorObserver(
              subject, netObserver_, net_position_, rnn_order));
        */
    }
}
