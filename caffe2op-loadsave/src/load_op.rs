crate::ix!();

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
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LoadOp<Context> {
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
