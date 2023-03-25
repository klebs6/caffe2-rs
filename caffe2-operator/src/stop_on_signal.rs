crate::ix!();

#[derive(Clone)]
pub struct StopOnSignal {

    handler: Arc<SignalHandler>,
}

impl StopOnSignal {
    
    #[inline] pub fn invoke(&mut self, iter: i32) -> bool {
        
        todo!();
        /*
            return handler_->CheckForSignals() != SignalHandler::Action::STOP;
        */
    }
}

impl Default for StopOnSignal {
    
    fn default() -> Self {
        todo!();
        /*
            : handler_(std::make_shared<SignalHandler>(
                SignalHandler::Action::STOP,
                SignalHandler::Action::STOP)
        */
    }
}

pub type ShouldContinue = fn(i32) -> bool;

