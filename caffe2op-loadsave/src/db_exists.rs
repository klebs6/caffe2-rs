crate::ix!();

/**
  | Checks if the db described by the arguments
  | exists.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/load_save_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct DBExistsOp<Context> {
    storage:       OperatorStorage,
    context:       Context,
    ws:            *mut Workspace,
    absolute_path: bool,
    db_name:       String,
    db_type:       String,
}

#[test] fn db_exists_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "DBExists",
        [],
        ["exists"],
        db_name="test_db",
        db_type="leveldb",
    )

    workspace.RunOperatorOnce(op)
    print("exists:", workspace.FetchBlob("exists"))
    */
}

num_inputs!{DBExists, 0}

num_outputs!{DBExists, 1}

outputs!{DBExists, 
    0 => ("exists",        "*(type: Tensor`<bool>`)* Scalar boolean output tensor. True if the db exists, else false.")
}

args!{DBExists, 
    0 => ("absolute_path", "*(type: int; default: 0)* If set to non-zero, save the db directly to the path specified by the `db` arg. If not set (default), prepend the path of the current root folder of the workspace to the path specified by the `db` arg."),
    1 => ("db_name",       "*(type: string)* Path to the db in question; see the `absolute_path` arg details for options regarding the current root folder of the workspace."),
    2 => ("db_type",       "*(type: string)* Type of db to save (options: lmdb, leveldb, minidb).")
}

impl<Context> DBExistsOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            ws_(ws),
            absolute_path_( this->template GetSingleArgument<int>("absolute_path", false)),
            db_name_(this->template GetSingleArgument<string>("db_name", "")),
            db_type_(this->template GetSingleArgument<string>("db_type", ""))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            string full_db_name =
            absolute_path_ ? db_name_ : (ws_->RootFolder() + "/" + db_name_);
        auto* output = Output(0);
        output->Resize();
        bool* exists = output->template mutable_data<bool>();

        *exists = caffe2::db::DBExists(db_type_, full_db_name);
        return true;
        */
    }
}
