crate::ix!();

impl DistributeFpnProposalsOp<CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            int num_roi_lvls = roi_max_level_ - roi_min_level_ + 1;
      CAFFE_ENFORCE_EQ(OutputSize(), num_roi_lvls + 1);

      // Load Input(0) to rois
      const auto& rois_in = Input(0);
      const int num_rois = rois_in.size(0);
      const int dim_rois = rois_in.size(1);
      CAFFE_ENFORCE(dim_rois == 4 || dim_rois == 5);
      Eigen::Map<const ERArrXXf> rois_4or5(
          rois_in.data<float>(), num_rois, dim_rois);
      ERArrXXf rois = ERArrXXf::Zero(num_rois, 5);
      rois.rightCols(dim_rois) = rois_4or5;

      // Distribute
      // equivalent to python code
      //   lvl_min = cfg.FPN.ROI_MIN_LEVEL
      //   lvl_max = cfg.FPN.ROI_MAX_LEVEL
      //   lvls = fpn.map_rois_to_fpn_levels(rois[:, 1:5], lvl_min, lvl_max)
      const int lvl_min = roi_min_level_;
      const int lvl_max = roi_max_level_;
      const int canon_scale = roi_canonical_scale_;
      const int canon_level = roi_canonical_level_;
      auto rois_block = rois.block(0, 1, rois.rows(), 4);
      auto lvls = utils::MapRoIsToFpnLevels(
          rois_block, lvl_min, lvl_max, canon_scale, canon_level, legacy_plus_one_);

      // Create new roi blobs for each FPN level
      // (See: modeling.FPN.add_multilevel_roi_blobs which is similar but annoying
      // to generalize to support this particular case.)
      //
      // equivalent to python code
      //   rois_idx_order = np.empty((0, ))
      //   for (output_idx, lvl in enumerate(range(lvl_min, lvl_max + 1)))
      //       idx_lvl = np.where(lvls == lvl)[0]
      //       blob_roi_level = rois[idx_lvl, :]
      //       outputs[output_idx + 1].reshape(blob_roi_level.shape)
      //       outputs[output_idx + 1].data[...] = blob_roi_level
      //       rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
      //   rois_idx_restore = np.argsort(rois_idx_order)
      //   blob_utils.py_op_copy_blob(rois_idx_restore.astype(np.int32),
      //   outputs[-1])
      EArrXi rois_idx_restore;
      for (int i = 0, lvl = lvl_min; i < num_roi_lvls; i++, lvl++) {
        ERArrXXf blob_roi_level;
        EArrXi idx_lvl;
        utils::RowsWhereRoILevelEquals(rois, lvls, lvl, &blob_roi_level, &idx_lvl);

        // Output blob_roi_level

        auto* roi_out = Output(
            i + 0,
            {blob_roi_level.rows(), blob_roi_level.cols()},
            at::dtype<float>());
        Eigen::Map<ERArrXXf> roi_out_mat(
            roi_out->template mutable_data<float>(),
            blob_roi_level.rows(),
            blob_roi_level.cols());
        roi_out_mat = blob_roi_level;

        // Append indices from idx_lvl to rois_idx_restore
        rois_idx_restore.conservativeResize(
            rois_idx_restore.size() + idx_lvl.size());
        rois_idx_restore.tail(idx_lvl.size()) = idx_lvl;
      }
      utils::ArgSort(rois_idx_restore);

      auto* rois_idx_restore_out =
          Output(OutputSize() - 1, {rois_idx_restore.size()}, at::dtype<int>());
      Eigen::Map<EArrXi> rois_idx_restore_out_mat(
          rois_idx_restore_out->template mutable_data<int>(),
          rois_idx_restore.size());
      rois_idx_restore_out_mat = rois_idx_restore;

      return true;
        */
    }
}
