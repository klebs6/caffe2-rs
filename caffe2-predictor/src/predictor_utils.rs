crate::ix!();

#[inline] pub fn get_blobs<'a>(def: &'a MetaNetDef, name: &String) 
-> &'a RepeatedPtrField<String> 
{
    todo!(); /* */
}

#[inline] pub fn get_net<'a>(def: &'a MetaNetDef, name: &String) -> &'a NetDef {
    
    todo!();
    /*
        for (const auto& n : def.nets()) {
        if (n.key() == name) {
          return n.value();
        }
      }
      CAFFE_THROW("Net not found: ", name);
    */
}

#[inline] pub fn extract_meta_net_def(
    cursor: *mut dyn Cursor, 
    key:    &String) -> Box<MetaNetDef> {
    
    todo!();
    /*
        CAFFE_ENFORCE(cursor);
      if (cursor->SupportsSeek()) {
        cursor->Seek(key);
      }
      for (; cursor->Valid(); cursor->Next()) {
        if (cursor->key() != key) {
          continue;
        }
        // We've found a match. Parse it out.
        BlobProto proto;
        CAFFE_ENFORCE(proto.ParseFromString(cursor->value()));
        Blob blob;
        DeserializeBlob(proto, &blob);
        CAFFE_ENFORCE(blob.template IsType<string>());
        auto def = std::make_unique<MetaNetDef>();
        CAFFE_ENFORCE(def->ParseFromString(blob.template Get<string>()));
        return def;
      }
      CAFFE_THROW("Failed to find in db the key: ", key);
    */
}

/**
  | Extract the MetaNetDef from `db`, and
  | run the global init net on the `master`
  | workspace.
  |
  */
#[inline] pub fn run_global_initialization(db: Box<DBReader>, master: *mut Workspace) -> Box<MetaNetDef> {
    
    todo!();
    /*
        CAFFE_ENFORCE(db.get());
      auto* cursor = db->cursor();

      auto metaNetDef = extractMetaNetDef(
          cursor, PredictorConsts::default_instance().meta_net_def());
      if (metaNetDef->has_modelinfo()) {
        CAFFE_ENFORCE(
            metaNetDef->modelinfo().predictortype() ==
                PredictorConsts::default_instance().single_predictor(),
            "Can only load single predictor");
      }
      VLOG(1) << "Extracted meta net def";

      const auto globalInitNet = getNet(
          *metaNetDef, PredictorConsts::default_instance().global_init_net_type());
      VLOG(1) << "Global init net: " << ProtoDebugString(globalInitNet);

      // Now, pass away ownership of the DB into the master workspace for
      // use by the globalInitNet.
      master->CreateBlob(PredictorConsts::default_instance().predictor_dbreader())
          ->Reset(db.release());

      // Now, with the DBReader set, we can run globalInitNet.
      CAFFE_ENFORCE(
          master->RunNetOnce(globalInitNet),
          "Failed running the globalInitNet: ",
          ProtoDebugString(globalInitNet));

      return metaNetDef;
    */
}
