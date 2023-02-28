crate::ix!();

use crate::{
    Cursor,
    BlobState,
    TensorShape,
    SerializationOptions,
    OperatorDef,
    Blob,
    CPUContext,
    Workspace,
    BlobSerializationOptions,
    BlobProto,
    CUDAContext,
    OperatorStorage
};

/**
  | Checks if the db described by the arguments
  | exists.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/load_save_op.cc
  |
  */
pub struct DBExistsOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
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

/**
  | The Load operator loads a set of serialized
  | blobs from a db or multiple dbs. It takes
  | $[0, \infty)$ number of inputs and $[0,
  | \infty)$ number of outputs, using the
  | db keys to match the db entries with the
  | outputs.
  | 
  | If at least one input is passed, then
  | it is assumed that that input blobs are
  | a set of DBReaders to load from. Otherwise
  | the `db` or `dbs` argument is used to
  | load blobs from one single db or multiple
  | dbs respectively. `db_type` argument
  | is used to specify the type of the input
  | db/dbs.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/load_save_op.cc
  |
  */
pub struct LoadOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage:           OperatorStorage,
    context:           Context,
    ws:                *mut Workspace,
    absolute_path:     bool,
    add_prefix:        String,
    strip_prefix:      String,
    db_name:           String,
    db_names:          Vec<String>,
    db_type:           String,
    keep_device:       bool,
    load_all:          bool,
    allow_incomplete:  bool,
    output_indices:    HashMap<String,i32>,
    key_to_dbid:       HashMap<String,i32>,
    blob_names:        Vec<String>,
    shape:             Vec<i64>,
}

#[test] fn load_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Load",
        [],
        ["X", "Y"],
        db="test_db",
        db_type="lmdb"
    )

    workspace.RunOperatorOnce(op)
    print("X:", workspace.FetchBlob("X"))
    print("Y:", workspace.FetchBlob("Y"))
    */
}

num_inputs!{Load, (0,INT_MAX)}

num_outputs!{Load, (0,INT_MAX)}

inputs!{Load, 
    0 => ("X, Y, ...",           "*(type: List(DBReader))* [OPTIONAL] List of DBReaders to load from. Can use this instead of the `db`/`dbs` args.")
}

args!{Load, 
    0 => ("absolute_path",       "*(type: int; default: 0)* If set to non-zero, save the db directly to the path specified by the `db` arg. If not set (default), prepend the path of the current root folder of the workspace to the path specified by the `db` arg."),
    1 => ("add_prefix",          "*(type: string, default: )* Blobs will be prefixed with this when loading. Useful for avoiding collisions with blobs existing in the workspace. The output blob names specified to this op should include this prefix."),
    2 => ("strip_prefix",        "*(type: string, default: )* Characters in the provided blob names that match `strip_prefix` will be removed prior to saving. Also, characters that precede `strip_prefix` will be removed. Useful for removing device scope from blob names."),
    3 => ("db",                  "*(type: string)* The output path of the db. See the `absolute_path` arg details for options regarding the current root folder of the workspace."),
    4 => ("dbs",                 "*(type: List(string))* List of paths to dbs to load blobs from. See the `absolute_path` arg details for options regarding the current root folder of the workspace."),
    5 => ("db_type",             "(type: string)* Type of db to save (options: lmdb, leveldb, minidb)."),
    6 => ("keep_device",         "*(type: int; default: 0)* If nonzero, the blobs are loaded into the device that is specified in the serialized `BlobProto`. Otherwise, the device will be set as the one that the `Load` operator is being run under."),
    7 => ("load_all",            "*(type: int; default: 0)* If nonzero, will load all blobs pointed to by the db to the workspace overwriting/creating blobs as needed."),
    8 => ("allow_incomplete",    "*(type: bool; default: False)* If True, will allow not loading all the output blobs specified in the outputs."),
    9 => ("source_blob_names",   "*(type: List(string))* If set, used instead of output blob names to specify which blobs in the db shall be loaded. Must be the same length as number of output blobs.")
}

tensor_inference_function!{Load, /* LoadTensorInference<> */}

impl<Context> LoadOp<Context> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            ws_(ws),
            absolute_path_( this->template GetSingleArgument<int>("absolute_path", false)),
            add_prefix_(this->template GetSingleArgument<string>("add_prefix", "")),
            strip_prefix_( this->template GetSingleArgument<string>("strip_prefix", "")),
            db_name_(this->template GetSingleArgument<string>("db", "")),
            db_names_(this->template GetRepeatedArgument<string>("dbs")),
            db_type_(this->template GetSingleArgument<string>("db_type", "")),
            keep_device_(this->template GetSingleArgument<int>("keep_device", 0)),
            load_all_(this->template GetSingleArgument<int>("load_all", 0)),
            allow_incomplete_( this->template GetSingleArgument<bool>("allow_incomplete", false)),
            blob_names_( this->template GetRepeatedArgument<string>("source_blob_names")),
            shape_(this->template GetRepeatedArgument<int64_t>("shape")) 

        if (InputSize() == 0) {
          CAFFE_ENFORCE_GT(db_type_.size(), 0, "Must specify a db type.");
          if (db_names_.empty()) {
            CAFFE_ENFORCE_GT(db_name_.size(), 0, "Must specify a db name.");
            db_names_.push_back(db_name_);
            db_name_ = "";
          } else {
            std::set<std::string> db_name_set;
            for (const string& db_name : db_names_) {
              CAFFE_ENFORCE_GT(db_name.size(), 0, "Db name should not be empty.");
              CAFFE_ENFORCE(
                  db_name_set.insert(db_name).second,
                  "Duplicated db name: ",
                  db_name);
            }
            db_name_ = "";
          }
        }
        CAFFE_ENFORCE(
            blob_names_.empty() || blob_names_.size() == OutputSize(),
            "Number of output blobs and source_blob_names mismatch.");
        CAFFE_ENFORCE(
            blob_names_.empty() || strip_prefix_.empty(),
            "strip_prefix and source_blob_names are mutually exclusive.");
        CAFFE_ENFORCE(
            blob_names_.empty() || !load_all_,
            "cannot load_all_ while using source_blob_names.");
        if (!load_all_) {
          // blob_names_ will be filled with ''source blob names'' in file/db
          // if argument source_blob_names is not given, then blob_names_ is
          // inferred from operator output
          if (blob_names_.empty()) {
            for (const string& name : operator_def.output()) {
              blob_names_.push_back(name);
            }
          }
          int idx = 0;
          std::set<std::string> name_set;
          for (const string& name : blob_names_) {
            CAFFE_ENFORCE(
                name_set.insert(name).second,
                "Duplicated source blob name: ",
                name);
            output_indices_[name] = idx++;
          }
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            int total_loaded_blobs = 0;
        std::unordered_map<string, load_save_op_util::BlobState> blob_states;
        if (InputSize() > 0) {
          for (int i = 0; i < InputSize(); ++i) {
            const db::DBReader& reader = this->template Input<db::DBReader>(i);
            extract(i, reader.cursor(), &blob_states, &total_loaded_blobs);
          }
        } else {
          for (int i = 0; i < db_names_.size(); ++i) {
            string full_db_name = absolute_path_
                ? db_names_[i]
                : (ws_->RootFolder() + "/" + db_names_[i]);
            std::unique_ptr<DB> in_db(
                caffe2::db::CreateDB(db_type_, full_db_name, caffe2::db::READ));
            CAFFE_ENFORCE(
                in_db.get(),
                "Cannot find db implementation of type ",
                db_type_,
                " (while trying to open ",
                full_db_name,
                ")");
            std::unique_ptr<Cursor> cursor(in_db->NewCursor());
            extract(i, cursor.get(), &blob_states, &total_loaded_blobs);
          }
        }

        load_save_op_util::validateBlobStates(blob_states);
        // Loaded all the needed blobs.
        if (!load_all_ && total_loaded_blobs == OutputSize()) {
          VLOG(1) << "Loaded " << total_loaded_blobs << " blobs fully from db(s)";
          return true;
        }

        if (load_all_) {
          for (const string& name : this->debug_def().output()) {
            CAFFE_ENFORCE(
                blob_states.count(name),
                "Output blob name ",
                name,
                " does not exist in the db(s).");
          }
          return true;
        }

        // Only loaded a subset of the blobs.
        if (allow_incomplete_) {
          VLOG(1) << "Loaded " << total_loaded_blobs << " blobs out of "
                  << OutputSize() << " blobs from db(s).";
        } else {
          for (const string& output_name : this->debug_def().output()) {
            if (blob_states.count(output_name) == 0) {
              LOG(ERROR) << "Failed to load blob: " << output_name;
            }
          }
          CAFFE_THROW(
              "Expected to load ",
              OutputSize(),
              " blobs, got ",
              total_loaded_blobs,
              " only.\n");
        }

        return true;
        */
    }
    
    #[inline] pub fn extract(
        &mut self, 
        db_id:              i32,
        cursor:             *mut dyn Cursor,
        blob_states:        *mut HashMap<String,BlobState>,
        total_loaded_blobs: *mut i32)  
    {
        todo!();
        /*
            if (load_all_) {
          extractAll(db_id, cursor, blob_states, total_loaded_blobs);
        } else {
          extractFrom(
              db_id,
              cursor,
              OperatorStorage::Outputs(),
              blob_states,
              total_loaded_blobs);
        }
        */
    }
    
    #[inline] pub fn extract_all(
        &mut self, 
        db_id:               i32,
        cursor:              *mut dyn Cursor,
        blob_states:         *mut HashMap<String,BlobState>,
        total_loaded_blobs:  *mut i32)  
    {
        todo!();
        /*
            CAFFE_ENFORCE(cursor, "cursor is not valid");
        int loaded_blobs = 0;
        for (; cursor->Valid(); cursor->Next()) {
          const auto key = load_save_op_util::buildBlobNameFromDbKey(
              cursor->key(), strip_prefix_, add_prefix_);
          if (key_to_dbid_.count(key) && key_to_dbid_[key] != db_id) {
            CAFFE_THROW("Duplicate Key ", key, " is found!\n");
          } else {
            key_to_dbid_[key] = db_id;
          }

          BlobProto proto;
          CAFFE_ENFORCE(
              proto.ParseFromString(cursor->value()), "Couldn't parse Proto");
          if (!keep_device_) {
            // If we are not keeping the device as the one specified in the
            // proto, we will set the current device.
            SetCurrentDevice(&proto);
          }
          Blob* blob = ws_->CreateBlob(key);
          load_save_op_util::ProcessBlob(
              blob, proto, blob_states, key, &loaded_blobs);
        }
        *total_loaded_blobs += loaded_blobs;
        */
    }
    
    #[inline] pub fn extract_from(
        &mut self, 
        db_id:                i32,
        cursor:               *mut dyn Cursor,
        outputs:              &Vec<*mut Blob>,
        blob_states:          *mut HashMap<String,BlobState>,
        total_loaded_blobs:   *mut i32)  
    {
        todo!();
        /*
            CAFFE_ENFORCE(cursor);
        int loaded_blobs = 0;
        for (; cursor->Valid(); cursor->Next()) {
          const auto key = load_save_op_util::buildBlobNameFromDbKey(
              cursor->key(), strip_prefix_, add_prefix_);
          if (!output_indices_.count(key)) {
            VLOG(1) << "Key " << key << " not used. Skipping.";
          } else {
            if (key_to_dbid_.count(key) && key_to_dbid_[key] != db_id) {
              CAFFE_THROW("Duplicate Key ", key, " is found!\n");
            } else {
              key_to_dbid_[key] = db_id;
            }

            VLOG(2) << "Deserializing blob " << key;
            BlobProto proto;
            CAFFE_ENFORCE(proto.ParseFromString(cursor->value()));
            if (!keep_device_) {
              // If we are not keeping the device as the one specified in the
              // proto, we will set the current device.
              SetCurrentDevice(&proto);
            }
            auto blobIndex = output_indices_[key];
            Blob* blob = outputs.at(blobIndex);
            load_save_op_util::ProcessBlob(
                blob, proto, blob_states, key, &loaded_blobs);

            if (*total_loaded_blobs + loaded_blobs == OutputSize()) {
              break;
            }
          }
        }

        *total_loaded_blobs += loaded_blobs;
        */
    }
}

/**
  | Saves a set of blobs to a db. 
  |
  | It takes $[1, \infty)$ 
  | number of inputs and has no
  | output. The contents of the inputs are
  | written into the db using the settings
  | specified by the arguments.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/load_save_op.cc
  |
  */
pub struct SaveOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage:      OperatorStorage,
    context:      Context,
    operator:     *mut OperatorStorage,
    strip_prefix: String,
    full_db_name: String,
    db_type:      String,
    blob_names:   Vec<String>,
    options:      SerializationOptions,
}

#[test] fn save_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Save",
        ["X", "Y", "Z"],
        [],
        db="test_db2",
        db_type="leveldb",
        blob_name_overrides=["x_scores", "y_scores", "z_scores"]
    )

    workspace.FeedBlob("X", np.random.randint(20, size=(5,5)))
    workspace.FeedBlob("Y", np.random.randint(20, size=(5,5)))
    workspace.FeedBlob("Z", np.random.randint(20, size=(5,5)))
    workspace.RunOperatorOnce(op)
    */
}

num_inputs!{Save, (1,INT_MAX)}

num_outputs!{Save, 0}

inputs!{Save, 
    0 => ("X", "*(type: Tensor)* Input tensor(s).")
}

args!{Save, 
    0 => ("absolute_path",       "*(type: int; default: 0)* If set to non-zero, save the db directly to the path specified by the `db` arg. If not set (default), prepend the path of the current root folder of the workspace to the path specified by the `db` arg."),
    1 => ("strip_prefix",        "*(type: string, default: )* Characters in the provided blob names that match `strip_prefix` will be removed prior to saving. Also, characters that precede `strip_prefix` will be removed. Useful for removing device scope from blob names."),
    2 => ("blob_name_overrides", "*(List(string))* If set, used as blob names instead of original blob names. Must be same length as number of blobs."),
    3 => ("db",                  "*(type: string)* The output path of the db. See the `absolute_path` arg details for options regarding the current root folder of the workspace."),
    4 => ("db_type",             "*(type: string)* Type of db to save (options: lmdb, leveldb, minidb)."),
    5 => ("chunk_size",          "*(type: string; default: kDefaultChunkSize)* The chunk size to split tensor data into. If not set, caffe2_tensor_chunk_size will be used")
}

impl<Context> SaveOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws), impl_(this, operator_def, ws)
        */

        /*
            : operator_(op),
          strip_prefix_(op->template GetSingleArgument<string>("strip_prefix", "")),
          db_type_(op->template GetSingleArgument<string>("db_type", "")),
          blob_names_( op->template GetRepeatedArgument<string>("blob_name_overrides")) 

      CAFFE_ENFORCE_GT(db_type_.size(), 0, "Must specify a db type.");
      CAFFE_ENFORCE(
          blob_names_.empty() || blob_names_.size() == op->Inputs().size(),
          "Number of blobs and blob_name_overrides mismatch.");
        CAFFE_ENFORCE(
            blob_names_.empty() || strip_prefix_.empty(),
            "strip_prefix and blob_name_overrides are mutually exclusive.");

      auto absolute_path =
          op->template GetSingleArgument<int>("absolute_path", false);
      auto db_name = op->template GetSingleArgument<string>("db", "");
      CAFFE_ENFORCE_GT(db_name.size(), 0, "Must specify a db name.");
      full_db_name_ = absolute_path ? db_name : (ws->RootFolder() + "/" + db_name);

      auto options_data = op->template GetSingleArgument<string>("options", "");
      if (!options_data.empty()) {
        if (!options_.ParseFromString(options_data)) {
          CAFFE_ENFORCE(false, "unable to parse serialization options");
        }
      }
      if (op->template HasSingleArgumentOfType<int>("chunk_size")) {
        // The chunk size argument pre-dates the options argument.
        // If it was passed in, add it to the options list as a final default
        // setting.
        auto chunk_size_argument =
            op->template GetSingleArgument<int>("chunk_size", kDefaultChunkSize);
        // The chunk_size argument used 0 to mean "no chunking", and -1 to mean
        // "default chunk size".  This is backwards from the behavior of the
        // chunk_size field in the BlobSerializationOptions, so swap these values if
        // we see them.  (BlobSerializationOptions uses 0 to mean "default chunk
        // size" since protobuf v3 does not support custom default values, and so we
        // need to use 0 to mean the default behavior.)
        constexpr int kOldDefaultChunkSize = -1;
        constexpr int kOldNoChunking = 0;
        if (chunk_size_argument == kOldDefaultChunkSize) {
          chunk_size_argument = kDefaultChunkSize;
        } else if (chunk_size_argument == kOldNoChunking) {
          chunk_size_argument = kNoChunking;
        }
        options_.mutable_options()->Add()->set_chunk_size(chunk_size_argument);
      }

      if (blob_names_.empty()) {
        std::set<std::string> input_names;
        blob_names_.resize(op->Inputs().size());
        for (int i = 0; i < blob_names_.size(); ++i) {
          std::string name;
          if (strip_prefix_.empty()) {
            name = operator_def.input(i);
          } else {
            auto match_pos = operator_def.input(i).find(strip_prefix_);
            if (match_pos == string::npos) {
              name = operator_def.input(i);
            } else {
              name = operator_def.input(i).substr(
                  match_pos + strip_prefix_.size(), string::npos);
            }
          }
          CAFFE_ENFORCE(
              input_names.insert(name).second, "Duplicated input: ", name);
          blob_names_[i] = name;
        }
      }
        */

    }

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            std::unique_ptr<DB> out_db(
          caffe2::db::CreateDB(db_type_, full_db_name_, caffe2::db::NEW));
      CAFFE_ENFORCE(
          out_db.get(),
          "Cannot find db implementation of type ",
          db_type_,
          " (while trying to open ",
          full_db_name_,
          ")");

      BlobSerializerBase::SerializationAcceptor acceptor =
          [&](const std::string& blobName, const std::string& data) {
            // transaction should take care of locking
            VLOG(2) << "Sending " << blobName << " blob's data of size "
                    << data.size() << " to db";
            auto transaction = out_db->NewTransaction();
            transaction->Put(blobName, data);
            transaction->Commit();
          };

      const vector<const Blob*>& inputs = operator_->OperatorStorage::Inputs();
      VLOG(0) << "Saving " << inputs.size() << " inputs to " << db_type_ << ": "
              << full_db_name_;
      BlobSerializationOptions default_options;
      for (int i = 0; i < inputs.size(); ++i) {
        SerializeBlob(
            *inputs[i],
            blob_names_[i],
            acceptor,
            GetBlobOptions(blob_names_[i], options_, default_options));
      }
      out_db->Close();
      return true;
        */
    }
}

#[inline] pub fn format_string<Ts>( pattern: &String, values: Ts) -> String 
{
    todo!();
    /*
        // Start with an initial buffer size that is probably enough most of the time.
      std::string buffer(256, '\0');
      auto bytes_written =
          snprintf(&buffer[0], buffer.size(), pattern.c_str(), values...);
      if (bytes_written < 0) {
        throw std::runtime_error("FormatString failed");
      }
      if (bytes_written > buffer.size()) {
        // Our initial buffer size wasn't enough, resize and run again.
        buffer.resize(bytes_written + 1);
        bytes_written =
            snprintf(&buffer[0], buffer.size(), pattern.c_str(), values...);
        if (bytes_written < 0) {
          throw std::runtime_error("FormatString failed");
        }
      }
      // Truncate the string to the correct size to trim off the nul terminator.
      buffer.resize(bytes_written);
      return buffer;
    */
}

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
pub struct CheckpointOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

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

impl LoadOp<CPUContext> {

    #[inline] pub fn set_current_device(&mut self, proto: *mut BlobProto)  {
        
        todo!();
        /*
            if (proto->has_tensor()) {
            proto->mutable_tensor()->clear_device_detail();
            proto->mutable_tensor()->mutable_device_detail()->set_device_type(
                PROTO_CPU);
          }
        */
    }

}

#[inline] pub fn load_tensor_inference<const VALUE_TYPE: i32>(
    def:    &OperatorDef,
    unused: &Vec<TensorShape>) -> Vec<TensorShape> 
{
    todo!();
    /*
        ArgumentHelper helper(def);
      auto shape = helper.GetRepeatedArgument<int64_t>("shape");
      vector<TensorShape> out;
      // Currently load op supports only shape.
      // TODO: We have to extend it to support shapes vector.
      // Since it support just one shape, we return
      // the right shape information only when there is just one blob loaded.
      // Otherwise, we return unknown TensorShapes.
      if (def.output_size() == 1 && shape.size() > 0) {
        TensorShape ts;
        ts.set_data_type(static_cast<TensorProto_DataType>(
            helper.GetSingleArgument<int>("dtype", VALUE_TYPE)));
        for (auto d : shape) {
          ts.add_dims(d);
        }
        out.push_back(ts);
      } else {
        for (int i = 0; i < def.output_size(); i++) {
          TensorShape ts;
          ts.set_unknown_shape(true);
          out.push_back(ts);
        }
      }
      return out;
    */
}

#[inline] pub fn get_blob_options<'a>(
    blob_name:       &str,
    options_list:    &SerializationOptions,
    default_options: &'a BlobSerializationOptions) -> &'a BlobSerializationOptions 
{
    todo!();
    /*
        for (const auto& options : options_list.options()) {
        const auto& name_regex = options.blob_name_regex();
        if (name_regex.empty()) {
          return options;
        }

    #if CAFFE2_HAVE_RE2
        // If we have re2, prefer it over std::regex.
        re2::RE2 regex(name_regex);
        if (re2::RE2::FullMatch(
            re2::StringPiece(blob_name.data(), blob_name.size()), regex)) {
          return options;
        }
    #else
        // std::regex should be avoided if at all possible, but use it as a fallback
        // if we don't have re2 (e.g., for some issues with it see
        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=61582)
        if (std::regex_match(
                blob_name.begin(), blob_name.end(), std::regex(name_regex))) {
          return options;
        }
    #endif
      }
      return default_options;
    */
}

register_cpu_operator!{DBExists,   DBExistsOp<CPUContext>}
register_cpu_operator!{Load,       LoadOp<CPUContext>}
register_cpu_operator!{Save,       SaveOp<CPUContext>}
register_cpu_operator!{Checkpoint, CheckpointOp<CPUContext>}

/**
  | CPU Operator old name: do NOT use, we
  | may deprecate this later.
  |
  */
register_cpu_operator!{Snapshot,   CheckpointOp<CPUContext>}

no_gradient!{Load}

should_not_do_gradient!{DBExists}
should_not_do_gradient!{Save}
should_not_do_gradient!{Checkpoint}
should_not_do_gradient!{Snapshot}

impl LoadOp<CUDAContext> {

    #[inline] pub fn set_current_device(&mut self, proto: *mut BlobProto)  {
        
        todo!();
        /*
            if (proto->has_tensor()) {
        proto->mutable_tensor()->clear_device_detail();
        auto* device_detail = proto->mutable_tensor()->mutable_device_detail();
        device_detail->set_device_type(PROTO_CUDA);
        device_detail->set_device_id(CaffeCudaGetDevice());
      }
        */
    }
}

register_cuda_operator!{Load,       LoadOp<CUDAContext>}
register_cuda_operator!{Save,       SaveOp<CUDAContext>}
register_cuda_operator!{Checkpoint, CheckpointOp<CUDAContext>}
