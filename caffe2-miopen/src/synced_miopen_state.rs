crate::ix!();

pub type PerGPUMIOPENStates 
= [[SyncedMIOPENState; CAFFE2_COMPILE_TIME_MAX_MIOPEN_STATES]; COMPILE_TIME_MAX_GPUS];

pub struct SyncedMIOPENState
{
    mutex:  parking_lot::RawMutex,
    state:  Box<MIOpenState>,
}

