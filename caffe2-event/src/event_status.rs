crate::ix!();

pub enum EventStatus {
    EVENT_INITIALIZED = 0,
    EVENT_SCHEDULED   = 1,
    EVENT_SUCCESS     = 2,
    EVENT_FAILED      = 3,
}
