crate::ix!();

use crate::{
    NetBase,
    ObserverBase,
};

/**
  | Thin class that attaches the observer
  | to all operators in the net
  |
  */
pub struct OperatorAttachingNetObserver<TOpObserver,TNetObserver> {
    base:               ObserverBase<NetBase>,
    operator_observers: Vec<*const TOpObserver>,
    phantom:            PhantomData<TNetObserver>,
}

impl<TOpObserver,TNetObserver> OperatorAttachingNetObserver<TOpObserver,TNetObserver> {
    
    pub fn new(subject: *mut NetBase, net_observer: *mut TNetObserver) -> Self {
        todo!();
        /*
            : ObserverBase<NetBase>(subject_) 

        const auto& operators = subject_->GetOperators();
        for (auto* op : operators) {
          auto observer = std::make_unique<TOpObserver>(op, netObserver);
          const auto* ob = observer.get();
          op->AttachObserver(std::move(observer));
          operator_observers_.push_back(ob);
        }
        */
    }
}
