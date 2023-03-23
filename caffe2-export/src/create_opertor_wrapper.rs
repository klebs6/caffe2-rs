crate::ix!();

#[cfg(all(not(caffe2_is_xplat_build), not(c10_mobile)))]
#[inline] pub fn create_c10operator_wrapper<Context>(op_name: &OperatorName) 
-> fn(_u0: &OperatorDef, _u1: *mut Workspace) -> Box<OperatorStorage> {

    todo!();
    /*
        return [op_name](const OperatorDef& op_def, Workspace* ws) {
        auto op_handle =
            c10::Dispatcher::singleton().findSchema(op_name);
        AT_ASSERTM(
            op_handle.has_value(),
            "Tried to register c10 operator ",
            op_name.name,
            ".",
            op_name.overload_name,
            " with caffe2, but didn't find the c10 operator.");
        return std::make_unique<C10OperatorWrapper<Context>>(
            *op_handle, op_def, ws);
    */
}
