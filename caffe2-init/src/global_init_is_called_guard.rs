crate::ix!();

pub struct GlobalInitIsCalledGuard;

impl GlobalInitIsCalledGuard {

    pub fn new() -> Self {
        todo!();
        /*
        if (!GlobalInitAlreadyRun()) {
            LOG(WARNING)
                << "Caffe2 GlobalInit should be run before any other API calls.";
        }
        */
    }
}
