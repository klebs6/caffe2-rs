crate::ix!();

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
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SaveOp<Context> {
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
