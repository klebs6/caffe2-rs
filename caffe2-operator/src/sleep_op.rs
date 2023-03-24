crate::ix!();

/**
  | SleepOp basically sleeps for a given
  | number of seconds.
  | 
  | We allow arbitrary inputs and at most
  | one output so that we can test scaffolding
  | of networks. If the output is 1, it will
  | be filled with vector<int64_t> with
  | two elements: start time and end time.
  |
  */
pub struct SleepOp {
    storage: OperatorStorage,
    context: CPUContext,
    ms:   i32,
}

impl SleepOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws),
            ms_(OperatorStorage::GetSingleArgument<int>("ms", 1000)) 

        DCHECK_GT(ms_, 0);
        DCHECK_LT(ms_, 3600 * 1000) << "Really? This long?";
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto start = std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(ms_));
        auto end = std::chrono::high_resolution_clock::now();
        if (OperatorStorage::OutputSize()) {
          vector<int64_t>* output = OperatorStorage::Output<vector<int64_t>>(0);
          output->resize(2);
          (*output)[0] = start.time_since_epoch().count();
          (*output)[1] = end.time_since_epoch().count();
        }
        return true;
        */
    }
}

num_inputs!{Sleep, (0,INT_MAX)}

num_outputs!{Sleep, (0,1)}

register_cpu_operator!{Sleep, SleepOp}

register_cuda_operator!{Sleep, SleepOp}

pub const kSleepNetDefString: &'static str = "
      name: \"sleepnet\"
      type: \"dag\"
      num_workers: 2
      op {
        output: \"sleep1\"
        name: \"sleep1\"
        type: \"Sleep\"
        arg {
          name: \"ms\"
          i: 100
        }
      }
      op {
        input: \"sleep1\"
        output: \"sleep2\"
        name: \"sleep2\"
        type: \"Sleep\"
        arg {
          name: \"ms\"
          i: 100
        }
      }
      op {
        output: \"sleep3\"
        name: \"sleep3\"
        type: \"Sleep\"
        arg {
          name: \"ms\"
          i: 150
        }
      }";

/**
  | This network has two operators reading
  | the same blob at the same time. This should
  | not change anything and the DAG should
  | still make sleep2 and sleep3 run in parallel.
  |
  */
pub const kSleepNetDefStringReadAfterRead: &'static str = "
      name: \"sleepnet\"
      type: \"dag\"
      num_workers: 2
      op {
        output: \"sleep1\"
        name: \"sleep1\"
        type: \"Sleep\"
        arg {
          name: \"ms\"
          i: 100
        }
      }
      op {
        input: \"sleep1\"
        output: \"sleep2\"
        name: \"sleep2\"
        type: \"Sleep\"
        arg {
          name: \"ms\"
          i: 100
        }
      }
      op {
        input: \"sleep1\"
        output: \"sleep3\"
        name: \"sleep3\"
        type: \"Sleep\"
        arg {
          name: \"ms\"
          i: 150
        }
      }";

/**
  | This network has two operators writing
  | out the sleep2 blob. As a result, the
  | operator sleep2-again creates a write
  | after write dependency and the whole
  | process should be sequential.
  |
  */
const kSleepNetDefStringWriteAfterWrite: &'static str = "
      name: \"sleepnet\"
      type: \"dag\"
      num_workers: 2
      op {
        output: \"sleep1\"
        name: \"sleep1\"
        type: \"Sleep\"
        arg {
          name: \"ms\"
          i: 100
        }
      }
      op {
        input: \"sleep1\"
        output: \"sleep2\"
        name: \"sleep2\"
        type: \"Sleep\"
        arg {
          name: \"ms\"
          i: 100
        }
      }
      op {
        output: \"sleep2\"
        name: \"sleep2-again\"
        type: \"Sleep\"
        arg {
          name: \"ms\"
          i: 150
        }
      }";

/**
  | This network has an operator writing
  | to sleep1 while another operator is
  | accessing it. As a result, the operator
  | sleep1-again creates a write after
  | read dependency and the whole process
  | should be sequential.
  |
  */
pub const kSleepNetDefStringWriteAfterRead: &'static str =
      "name: \"sleepnet\"
      type: \"dag\"
      num_workers: 2
      op {
        output: \"sleep1\"
        name: \"sleep1\"
        type: \"Sleep\"
        arg {
          name: \"ms\"
          i: 100
        }
      }
      op {
        input: \"sleep1\"
        output: \"sleep2\"
        name: \"sleep2\"
        type: \"Sleep\"
        arg {
          name: \"ms\"
          i: 100
        }
      }
      op {
        output: \"sleep1\"
        name: \"sleep1-again\"
        type: \"Sleep\"
        arg {
          name: \"ms\"
          i: 150
        }
      }";

/**
  | This network has an operator writing to sleep1
  | while another operator has a control dependency
  | on it. As a result, the operator sleep1-again
  | creates a write after read dependency and the
  | whole process should be sequential.
  */
pub const kSleepNetDefStringControlDependency: &'static str = "
  name: \"sleepnet\"
  type: \"dag\"
  num_workers: 2
  op {
    output: \"sleep1\"
    name: \"sleep1\"
    type: \"Sleep\"
    arg {
      name: \"ms\"
      i: 100
    }
  }
  op {
    control_input: \"sleep1\"
    output: \"sleep2\"
    name: \"sleep2\"
    type: \"Sleep\"
    arg {
      name: \"ms\"
      i: 100
    }
  }
  op {
    output: \"sleep1\"
    name: \"sleep1-again\"
    type: \"Sleep\"
    arg {
      name: \"ms\"
      i: 150
    }
  }";

