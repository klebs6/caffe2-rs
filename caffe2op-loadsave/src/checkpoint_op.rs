crate::ix!();

/**
  | CheckpointOp is a wrapper over a SaveFloatTensorOp
  | that basically allows flexible naming
  | over iterations.
  | 
  | The file pattern in db_name should be
  | a format string that can be passed into
  | sprintf with an int argument specifying
  | the current iteration. An example:
  | "/path/to/my/checkpoint/checkpoint_at_%d.pb"
  | 
  | The Checkpoint operator is similar
  | to the Save operator, but allows one
  | to save to db every few iterations, with
  | a db name that is appended with the iteration
  | count. It takes [1, infinity) number
  | of inputs and has no output. The first
  | input has to be a TensorCPU of type int
  | and has size 1 (i.e. the iteration counter).
  | This is determined whether we need to
  | do checkpointing.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct CheckpointOp<Context> {
    storage:     OperatorStorage,
    context:     Context,
    db_pattern:  String,
    every:       i32,
    ws:          *mut Workspace,
    save_op_def: OperatorDef,
}

num_inputs!{Checkpoint, (1,INT_MAX)}

num_outputs!{Checkpoint, 0}

args!{Checkpoint, 
    0 => ("absolute_path", "(int, default 0) if set, use the db path directly and do not prepend the current root folder of the workspace."),
    1 => ("db",            "(string) a template string that one can combine with the iteration to create the final db name. For example, /home/lonestarr/checkpoint_%08d.db"),
    2 => ("db_type",       "(string) the type of the db."),
    3 => ("every",         "(int, default 1) the checkpointing is carried out when (iter mod every) is zero.")
}

impl<Context> CheckpointOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            db_pattern_(this->template GetSingleArgument<string>("db", "")),
            every_(this->template GetSingleArgument<int>("every", 1)),
            ws_(ws),
            save_op_def_(operator_def) 

        CAFFE_ENFORCE_GT(
            db_pattern_.size(), 0, "Must specify a checkpoint file pattern.");
        CAFFE_ENFORCE_GT(every_, 0, "Checkpoint interval should be positive.");
        if (every_ == 1) {
          // Just issue a warning, but it's totally legal so we don't do anything.
          LOG(WARNING) << "It seems that we are checkpointting every iteration. "
                       << "Is that intended?";
        }
        save_op_def_.set_type("Save");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            int64_t iter =
            this->template Input<Tensor>(0, CPU).template data<int64_t>()[0];
        if (iter % every_ == 0) {
          GetMutableArgument("db", true, &save_op_def_)
              ->set_s(FormatString(db_pattern_, iter));
          SaveOp<Context> sub_op(save_op_def_, ws_);
          return sub_op.Run();
        } else {
          return true;
        }
        */
    }
}
