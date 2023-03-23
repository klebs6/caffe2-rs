crate::ix!();

register_event_create_function![CUDA,          EventCreateCUDA];
register_event_record_function![CUDA,          EventRecordCUDA];
register_event_wait_function![CUDA,            CUDA, EventWaitCUDACUDA];
register_event_wait_function![CPU,             CUDA, EventWaitCPUCUDA];
register_event_wait_function![CUDA,            CPU, EventWaitCUDACPU];
register_event_finish_function![CUDA,          EventFinishCUDA];

register_event_query_function![CUDA,           EventQueryCUDA];
register_event_error_message_function![CUDA,   EventErrorMessageCUDA];
register_event_set_finished_function![CUDA,    EventSetFinishedCUDA];
register_event_reset_function![CUDA,           EventResetCUDA];

register_event_wait_function![MKLDNN,          CUDA, EventWaitCPUCUDA];
register_event_wait_function![CUDA,            MKLDNN, EventWaitCUDACPU];
