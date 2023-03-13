crate::ix!();

#[test] fn event_cpu_test_event_basics() {
    todo!();
    /*
      DeviceOption device_option;
      device_option.set_device_type(PROTO_CPU);
      Event event(device_option);
      CPUContext context;

      context.Record(&event);
      event.SetFinished();

      context.WaitEvent(event);
      event.Finish();

      event.Reset();
      event.Record(CPU, &context);
      event.SetFinished();
      event.Wait(CPU, &context);
      */
}

#[test] fn event_cpu_test_event_errors() {
    todo!();
    /*
      DeviceOption device_option;
      device_option.set_device_type(PROTO_CPU);
      Event event(device_option);

      event.SetFinished();
      ASSERT_THROW(event.SetFinished("error"), caffe2::EnforceNotMet);
      ASSERT_EQ(event.ErrorMessage(), "No error");

      event.Reset();
      event.SetFinished("error 1");
      event.SetFinished("error 2");
      ASSERT_EQ(event.ErrorMessage(), "error 1");
      */
}
