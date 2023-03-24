crate::ix!();

pub fn operator_logger_default(def: &OperatorDef) { }

lazy_static!{
    static ref operator_logger: fn(def: &OperatorDef) -> () = operator_logger_default;
}

/// Operator logging capabilities
pub fn set_operator_logger(tracer: fn(def: &OperatorDef) -> ()) {
    todo!();
    /*
       OperatorLogger = tracer;
       */
}

pub fn get_operator_logger() -> fn(def: &OperatorDef) -> () {
    todo!();
    /*
       return OperatorLogger;
       */
}

