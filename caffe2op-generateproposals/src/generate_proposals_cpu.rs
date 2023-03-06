crate::ix!();

impl GenerateProposalsOp<CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& scores = Input(0);
      const auto& bbox_deltas = Input(1);
      const auto& im_info_tensor = Input(2);
      const auto& anchors_tensor = Input(3);

      CAFFE_ENFORCE_EQ(scores.dim(), 4, scores.dim());
      CAFFE_ENFORCE(scores.template IsType<float>(), scores.dtype().name());
      const auto num_images = scores.size(0);
      const auto A = scores.size(1);
      const auto height = scores.size(2);
      const auto width = scores.size(3);
      const auto box_dim = anchors_tensor.size(1);
      CAFFE_ENFORCE(box_dim == 4 || box_dim == 5);

      // bbox_deltas: (num_images, A * box_dim, H, W)
      CAFFE_ENFORCE_EQ(
          bbox_deltas.sizes(),
          (at::ArrayRef<int64_t>{num_images, box_dim * A, height, width}));

      // im_info_tensor: (num_images, 3), format [height, width, scale; ...]
      CAFFE_ENFORCE_EQ(im_info_tensor.sizes(), (vector<int64_t>{num_images, 3}));
      CAFFE_ENFORCE(
          im_info_tensor.template IsType<float>(), im_info_tensor.dtype().name());

      // anchors: (A, box_dim)
      CAFFE_ENFORCE_EQ(anchors_tensor.sizes(), (vector<int64_t>{A, box_dim}));
      CAFFE_ENFORCE(
          anchors_tensor.template IsType<float>(), anchors_tensor.dtype().name());

      Eigen::Map<const ERArrXXf> im_info(
          im_info_tensor.data<float>(),
          im_info_tensor.size(0),
          im_info_tensor.size(1));

      Eigen::Map<const ERArrXXf> anchors(
          anchors_tensor.data<float>(),
          anchors_tensor.size(0),
          anchors_tensor.size(1));

      std::vector<ERArrXXf> im_boxes(num_images);
      std::vector<EArrXf> im_probs(num_images);
      for (int i = 0; i < num_images; i++) {
        auto cur_im_info = im_info.row(i);
        auto cur_bbox_deltas = GetSubTensorView<float>(bbox_deltas, i);
        auto cur_scores = GetSubTensorView<float>(scores, i);

        ERArrXXf& im_i_boxes = im_boxes[i];
        EArrXf& im_i_probs = im_probs[i];
        ProposalsForOneImage(
            cur_im_info,
            anchors,
            cur_bbox_deltas,
            cur_scores,
            &im_i_boxes,
            &im_i_probs);
      }

      int roi_counts = 0;
      for (int i = 0; i < num_images; i++) {
        roi_counts += im_boxes[i].rows();
      }
      const int roi_col_count = box_dim + 1;
      auto* out_rois = Output(0, {roi_counts, roi_col_count}, at::dtype<float>());
      auto* out_rois_probs = Output(1, {roi_counts}, at::dtype<float>());
      float* out_rois_ptr = out_rois->template mutable_data<float>();
      float* out_rois_probs_ptr = out_rois_probs->template mutable_data<float>();
      for (int i = 0; i < num_images; i++) {
        const ERArrXXf& im_i_boxes = im_boxes[i];
        const EArrXf& im_i_probs = im_probs[i];
        int csz = im_i_boxes.rows();

        // write rois
        Eigen::Map<ERArrXXf> cur_rois(out_rois_ptr, csz, roi_col_count);
        cur_rois.col(0).setConstant(i);
        cur_rois.block(0, 1, csz, box_dim) = im_i_boxes;

        // write rois_probs
        Eigen::Map<EArrXf>(out_rois_probs_ptr, csz) = im_i_probs;

        out_rois_ptr += csz * roi_col_count;
        out_rois_probs_ptr += csz;
      }

      return true;
        */
    }
}
