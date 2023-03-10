crate::ix!();

impl<Context> CollectRpnProposalsOp<Context> {

    #[inline] pub fn run_on_cpu_device(&mut self) -> bool {
        
        todo!();
        /*
            int num_rpn_lvls = rpn_max_level_ - rpn_min_level_ + 1;
      CAFFE_ENFORCE_EQ(InputSize(), 2 * num_rpn_lvls);

      // Collect rois and scores in Eigen
      // rois are in [[batch_idx, x0, y0, x1, y2], ...] format
      // Combine predictions across all levels and retain the top scoring
      //
      // equivalent to python code
      //   roi_inputs = inputs[:num_rpn_lvls]
      //   score_inputs = inputs[num_rpn_lvls:]
      //   rois = np.concatenate([blob.data for blob in roi_inputs])
      //   scores = np.concatenate([blob.data for blob in score_inputs]).squeeze()
      int proposal_num = 0;
      for (int i = 0; i < num_rpn_lvls; i++) {
        const auto& roi_in = Input(i);
        proposal_num += roi_in.size(0);
      }
      ERArrXXf rois(proposal_num, 5);
      EArrXf scores(proposal_num);
      int len = 0;
      for (int i = 0; i < num_rpn_lvls; i++) {
        const auto& roi_in = Input(i);
        const int n = roi_in.size(0);

        Eigen::Map<const ERArrXXf> roi(roi_in.data<float>(), n, 5);
        rois.block(len, 0, n, 5) = roi;

        const auto& score_in = Input(num_rpn_lvls + i);
        CAFFE_ENFORCE_EQ(score_in.size(0), n);

        // No need to squeeze, since we are reshaping when converting to Eigen
        // https://docs.scipy.org/doc/numpy/reference/generated/numpy.squeeze.html
        Eigen::Map<const EArrXf> score(score_in.data<float>(), n);
        scores.segment(len, n) = score;

        len += n;
      }

      // Grab only top rpn_post_nms_topN rois
      // equivalent to python code
      //   inds = np.argsort(-scores)[:rpn_post_nms_topN]
      //   rois = rois[inds, :]
      utils::SortAndLimitRoIsByScores(scores, rpn_post_nms_topN_, rois);

      // equivalent to python code
      //   outputs[0].reshape(rois.shape)
      //   outputs[0].data[...] = rois

      auto* rois_out = Output(0, {rois.rows(), rois.cols()}, at::dtype<float>());
      Eigen::Map<ERArrXXf> rois_out_mat(
          rois_out->template mutable_data<float>(), rois.rows(), rois.cols());
      rois_out_mat = rois;

      return true;
        */
    }
}
