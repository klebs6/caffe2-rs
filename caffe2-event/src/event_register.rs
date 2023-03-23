crate::ix!();

register_event_create_function![
    CPU,           
    EventCreateCPU
];

register_event_record_function![
    CPU,           
    EventRecordCPU
];

register_event_wait_function![
    CPU,             
    CPU, EventWaitCPUCPU
];

register_event_finish_function![
    CPU,           
    EventFinishCPU
];

register_event_query_function![
    CPU,            
    EventQueryCPU
];

register_event_error_message_function![
    CPU,    
    EventErrorMessageCPU
];

register_event_set_finished_function![
    CPU,     
    EventSetFinishedCPU
];

register_event_reset_function![
    CPU,            
    EventResetCPU
];

register_event_set_callback_function![
    CPU,     
    EventSetCallbackCPU
];
