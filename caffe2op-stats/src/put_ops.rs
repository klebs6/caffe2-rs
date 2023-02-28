crate::ix!();

pub struct TemplatePutOp<T> {
    storage:           OperatorStorage,
    context:           CPUContext,
    given_name:        String,
    magnitude_expand:  i64,
    bound:             bool,
    has_default:       bool,
    default_value:     f32,
    stat:              T,
}

impl<T> TemplatePutOp<T> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator(operator_def, ws),
            given_name_(GetSingleArgument<std::string>( "stat_name", operator_def.input().Get(0))),
            magnitude_expand_(GetSingleArgument<int64_t>("magnitude_expand", 1)),
            bound_(GetSingleArgument<bool>("bound", false)),
            has_default_(HasSingleArgumentOfType<float>("default_value")),
            default_value_(GetSingleArgument<float>("default_value", 0.0)),
            stat_(given_name_)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<
            int,
            float,
            uint8_t,
            int8_t,
            uint16_t,
            int16_t,
            int64_t,
            at::Half,
            double>>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<V>(&mut self) -> bool {
    
        todo!();
        /*
            V input = default_value_;

        // If we receive an empty tensor
        if (Input(0).template data<V>()) {
          input = *Input(0).template data<V>();
        } else if (!has_default_) {
          CAFFE_THROW(
              "Default value must be provided when receiving empty tensors for ",
              given_name_);
        }

        int64_t bound_value =
            int64_t::max / magnitude_expand_;

        int64_t int_value;
        if (bound_) {
          if (isNan(input)) {
            int_value = 0;
          } else if (input <= -bound_value) {
            int_value = int64_t::min;
          } else if (input >= bound_value) {
            int_value = int64_t::max;
          } else {
            int_value = input * magnitude_expand_;
          }
        } else {
          CAFFE_ENFORCE(
              std::abs(static_cast<int64_t>(input)) < bound_value,
              "Input value is too large for the given magnitude expansion!");
          CAFFE_ENFORCE(!isNan(input), "Input value cannot be NaN!");
          int_value = input * magnitude_expand_;
        }

        CAFFE_EVENT(stat_, stat_value, int_value);

        return true;
        */
    }
    
    #[inline] pub fn is_nan<V>(&mut self, input: V) -> bool {
    
        todo!();
        /*
            /*
        Checks if given number of is NaN, while being permissive with different
        implementations of the standard libraries between operating systems.

        Uses the preperties of NaN, defined by IEEE.
        https://www.gnu.org/software/libc/manual/html_node/Infinity-and-NaN.html
        */
        return input != input;
        */
    }
}

#[macro_export] macro_rules! register_templated_stat_put_op {
    ($OP_NAME:ident, $STAT_NAME:ident, $STAT_MACRO:ident) => {
        /*
        
          struct STAT_NAME {                                                   
            CAFFE_STAT_CTOR(STAT_NAME);                                        
            STAT_MACRO(stat_value);                                            
          };                                                                   
          REGISTER_CPU_OPERATOR(OP_NAME, TemplatePutOp<STAT_NAME>);
        */
    }
}

/**
  | Consume a value and pushes it to the global
  | stat registry as an average.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_put_ops.cc
  |
  */
register_templated_stat_put_op!{
    AveragePut,
    AveragePutStat,
    CAFFE_AVG_EXPORTED_STAT
}

num_inputs!{AveragePut, 1}

num_outputs!{AveragePut, 0}

inputs!{AveragePut, 
    0 => ("value", "(*Tensor`<number>`*): A scalar tensor, representing any numeric value")
}

args!{AveragePut, 
    0 => ("name", "(*str*): name of the stat. If not present, then uses name of input blob"),
    1 => ("magnitude_expand", "(*int64_t*): number to multiply input values by (used when inputting floats, as stats can only receive integers"),
    2 => ("bound", "(*boolean*): whether or not to clamp inputs to the max inputs allowed"),
    3 => ("default_value", "(*float*): Optionally provide a default value for receiving empty tensors")
}

/**
  | Consume a value and pushes it to the global
  | stat registry as an sum.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_put_ops.cc
  |
  */
register_templated_stat_put_op!{
    IncrementPut,
    IncrementPutStat,
    CAFFE_EXPORTED_STAT
}

num_inputs!{IncrementPut, 1}

num_outputs!{IncrementPut, 0}

inputs!{IncrementPut, 
    0 => ("value", "(*Tensor`<number>`*): A scalar tensor, representing any numeric value")
}

args!{IncrementPut, 
    0 => ("name", "(*str*): name of the stat. If not present, then uses name of input blob"),
    1 => ("magnitude_expand", "(*int64_t*): number to multiply input values by (used when inputting floats, as stats can only receive integers"),
    2 => ("bound", "(*boolean*): whether or not to clamp inputs to the max inputs allowed"),
    3 => ("default_value", "(*float*): Optionally provide a default value for receiving empty tensors")
}

/**
  | Consume a value and pushes it to the global
  | stat registry as an standard deviation.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_put_ops.cc
  |
  */
register_templated_stat_put_op!{
    StdDevPut,
    StdDevPutStat,
    CAFFE_STDDEV_EXPORTED_STAT
}

num_inputs!{StdDevPut, 1}

num_outputs!{StdDevPut, 0}

inputs!{StdDevPut, 
    0 => ("value", "(*Tensor`<number>`*): A scalar tensor, representing any numeric value")
}

args!{StdDevPut, 
    0 => ("name",             "(*str*): name of the stat. If not present, then uses name of input blob"),
    1 => ("magnitude_expand", "(*int64_t*): number to multiply input values by (used when inputting floats, as stats can only receive integers"),
    2 => ("bound",            "(*boolean*): whether or not to clamp inputs to the max inputs allowed"),
    3 => ("default_value",    "(*float*): Optionally provide a default value for receiving empty tensors")
}
