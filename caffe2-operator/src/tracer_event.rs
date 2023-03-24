crate::ix!();

pub struct TracerEvent {
    op_id:         i32,                   // default = -1
    task_id:       i32,                   // default = -1
    stream_id:     i32,                   // default = -1
    name:          *const u8,             // default = nullptr
    category:      *const u8,             // default = nullptr
    timestamp:     i64,                   // default = -1.0
    is_beginning:  bool,                  // default = false
    thread_label:  i64,                   // default = -1
    tid:           std::thread::ThreadId,
    iter:          i32,                   // default = -1
}
