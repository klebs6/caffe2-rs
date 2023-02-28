crate::ix!();

use crate::{
    CPUContext,
    OperatorStorage,
};

/**
  | Apply NMS to each class (except background)
  | and limit the number of returned boxes.
  | 
  | C++ implementation of function insert_box_results_with_nms_and_limit()
  |
  */
pub struct BoxWithNMSLimitOp<Context> {

    // USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    /// TEST.SCORE_THRESH
    score_thres: f32, // 0.05

    /// TEST.NMS
    nms_thres: f32, //0.3

    /// TEST.DETECTIONS_PER_IM
    detections_per_im: i32, //100

    /// TEST.SOFT_NMS.ENABLED
    soft_nms_enabled: bool, //false

    /// TEST.SOFT_NMS.METHOD
    soft_nms_method_str: String, // "linear"
    soft_nms_method: u32, //1, linear

    /// TEST.SOFT_NMS.SIGMA
    soft_nms_sigma: f32, //0.5

    /**
      | Lower-bound on updated scores to discard
      | boxes
      |
      */
    soft_nms_min_score_thres: f32, //0.001

    /**
      | Set for RRPN case to handle rotated boxes.
      | Inputs should be in format [ctr_x, ctr_y,
      | width, height, angle (in degrees)].
      |
      */
    rotated: bool, //false

    /// MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
    cls_agnostic_bbox_reg: bool, //false

    /**
      | Whether input `boxes` includes background
      | class. If true, boxes will have shape of
      | (N, (num_fg_class+1) * 4or5), otherwise
      | (N, num_fg_class * 4or5)
      */
    input_boxes_include_bg_cls: bool, //true

    /**
      | Whether output `classes` includes
      | background class. If true, index 0 will
      | represent background, and valid outputs
      | start from 1.
      */
    output_classes_include_bg_cls: bool, //true

    /**
      | The index where foreground starts in
      | scoures. Eg. if 0 represents background
      | class then foreground class starts
      | with 1.
      |
      */
    input_scores_fg_cls_starting_id: i32, //1

    /**
      | The infamous "+ 1" for box width and height
      | dating back to the DPM days
      |
      */
    legacy_plus_one: bool, //true
}

impl<Context> BoxWithNMSLimitOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            score_thres_(
                this->template GetSingleArgument<float>("score_thresh", 0.05)),
            nms_thres_(this->template GetSingleArgument<float>("nms", 0.3)),
            detections_per_im_(
                this->template GetSingleArgument<int>("detections_per_im", 100)),
            soft_nms_enabled_(
                this->template GetSingleArgument<bool>("soft_nms_enabled", false)),
            soft_nms_method_str_(this->template GetSingleArgument<std::string>(
                "soft_nms_method",
                "linear")),
            soft_nms_sigma_(
                this->template GetSingleArgument<float>("soft_nms_sigma", 0.5)),
            soft_nms_min_score_thres_(this->template GetSingleArgument<float>(
                "soft_nms_min_score_thres",
                0.001)),
            rotated_(this->template GetSingleArgument<bool>("rotated", false)),
            cls_agnostic_bbox_reg_(this->template GetSingleArgument<bool>(
                "cls_agnostic_bbox_reg",
                false)),
            input_boxes_include_bg_cls_(this->template GetSingleArgument<bool>(
                "input_boxes_include_bg_cls",
                true)),
            output_classes_include_bg_cls_(this->template GetSingleArgument<bool>(
                "output_classes_include_bg_cls",
                true)),
            legacy_plus_one_(
                this->template GetSingleArgument<bool>("legacy_plus_one", true)) 

        CAFFE_ENFORCE(
            soft_nms_method_str_ == "linear" || soft_nms_method_str_ == "gaussian",
            "Unexpected soft_nms_method");
        soft_nms_method_ = (soft_nms_method_str_ == "linear") ? 1 : 2;

        // When input `boxes` doesn't include background class, the score will skip
        // background class and start with foreground classes directly, and put the
        // background class in the end, i.e. score[:, 0:NUM_CLASSES-1] represents
        // foreground classes and score[:,NUM_CLASSES] represents background class.
        input_scores_fg_cls_starting_id_ = (int)input_boxes_include_bg_cls_;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (InputSize() > 2) {
          return DispatchHelper<TensorTypes<int, float>>::call(this, Input(2));
        } else {
          return DoRunWithType<float>();
        }
        */
    }

    /**
      | Map a class id (starting with background
      | and then foreground) from (0, 1, ...,
      | NUM_FG_CLASSES) to it's matching value in
      | box
      */
    #[inline] pub fn get_box_cls_index(&mut self, bg_fg_cls_id: i32) -> i32 {
        
        todo!();
        /*
            if (cls_agnostic_bbox_reg_) {
          return 0;
        } else if (!input_boxes_include_bg_cls_) {
          return bg_fg_cls_id - 1;
        } else {
          return bg_fg_cls_id;
        }
        */
    }

    /**
      | Map a class id (starting with background
      | and then foreground) from (0, 1, ...,
      | NUM_FG_CLASSES) to it's matching value in
      | score
      */
    #[inline] pub fn get_score_cls_index(&mut self, bg_fg_cls_id: i32) -> i32 {
        
        todo!();
        /*
            return bg_fg_cls_id - 1 + input_scores_fg_cls_starting_id_;
        */
    }
}

impl BoxWithNMSLimitOp<CPUContext> {

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto& tscores = Input(0);
          const auto& tboxes = Input(1);

          const int box_dim = rotated_ ? 5 : 4;

          // tscores: (num_boxes, num_classes), 0 for background
          if (tscores.dim() == 4) {
            CAFFE_ENFORCE_EQ(tscores.size(2), 1);
            CAFFE_ENFORCE_EQ(tscores.size(3), 1);
          } else {
            CAFFE_ENFORCE_EQ(tscores.dim(), 2);
          }
          CAFFE_ENFORCE(tscores.template IsType<float>(), tscores.dtype().name());
          // tboxes: (num_boxes, num_classes * box_dim)
          if (tboxes.dim() == 4) {
            CAFFE_ENFORCE_EQ(tboxes.size(2), 1);
            CAFFE_ENFORCE_EQ(tboxes.size(3), 1);
          } else {
            CAFFE_ENFORCE_EQ(tboxes.dim(), 2);
          }
          CAFFE_ENFORCE(tboxes.template IsType<float>(), tboxes.dtype().name());

          int N = tscores.size(0);
          int num_classes = tscores.size(1);

          CAFFE_ENFORCE_EQ(N, tboxes.size(0));
          int num_boxes_classes = get_box_cls_index(num_classes - 1) + 1;
          CAFFE_ENFORCE_EQ(num_boxes_classes * box_dim, tboxes.size(1));

          // Default value for batch_size and batch_splits
          int batch_size = 1;
          vector<T> batch_splits_default(1, tscores.size(0));
          const T* batch_splits_data = batch_splits_default.data();
          if (InputSize() > 2) {
            // tscores and tboxes have items from multiple images in a batch. Get the
            // corresponding batch splits from input.
            const auto& tbatch_splits = Input(2);
            CAFFE_ENFORCE_EQ(tbatch_splits.dim(), 1);
            batch_size = tbatch_splits.size(0);
            batch_splits_data = tbatch_splits.data<T>();
          }
          Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> batch_splits(batch_splits_data, batch_size);
          CAFFE_ENFORCE_EQ(batch_splits.sum(), N);

          auto* out_scores = Output(0, {0}, at::dtype<float>());
          auto* out_boxes = Output(1, {0, box_dim}, at::dtype<float>());
          auto* out_classes = Output(2, {0}, at::dtype<float>());

          Tensor* out_keeps = nullptr;
          Tensor* out_keeps_size = nullptr;
          if (OutputSize() > 4) {
            out_keeps = Output(4);
            out_keeps_size = Output(5);
            out_keeps->Resize(0);
            out_keeps_size->Resize(batch_size, num_classes);
          }

          vector<int> total_keep_per_batch(batch_size);
          int offset = 0;
          for (int b = 0; b < batch_splits.size(); ++b) {
            int num_boxes = batch_splits[b];
            Eigen::Map<const ERArrXXf> scores(
                tscores.data<float>() + offset * tscores.size(1),
                num_boxes,
                tscores.size(1));
            Eigen::Map<const ERArrXXf> boxes(
                tboxes.data<float>() + offset * tboxes.size(1),
                num_boxes,
                tboxes.size(1));

            // To store updated scores if SoftNMS is used
            ERArrXXf soft_nms_scores(num_boxes, tscores.size(1));
            vector<vector<int>> keeps(num_classes);

            // Perform nms to each class
            // skip j = 0, because it's the background class
            int total_keep_count = 0;
            for (int j = 1; j < num_classes; j++) {
              auto cur_scores = scores.col(get_score_cls_index(j));
              auto inds = utils::GetArrayIndices(cur_scores > score_thres_);
              auto cur_boxes =
                  boxes.block(0, get_box_cls_index(j) * box_dim, boxes.rows(), box_dim);

              if (soft_nms_enabled_) {
                auto cur_soft_nms_scores = soft_nms_scores.col(get_score_cls_index(j));
                keeps[j] = utils::soft_nms_cpu(
                    &cur_soft_nms_scores,
                    cur_boxes,
                    cur_scores,
                    inds,
                    soft_nms_sigma_,
                    nms_thres_,
                    soft_nms_min_score_thres_,
                    soft_nms_method_,
                    -1, /* topN */
                    legacy_plus_one_);
              } else {
                std::sort(
                    inds.data(),
                    inds.data() + inds.size(),
                    [&cur_scores](int lhs, int rhs) {
                      return cur_scores(lhs) > cur_scores(rhs);
                    });
                int keep_max = detections_per_im_ > 0 ? detections_per_im_ : -1;
                keeps[j] = utils::nms_cpu(
                    cur_boxes,
                    cur_scores,
                    inds,
                    nms_thres_,
                    keep_max,
                    legacy_plus_one_);
              }
              total_keep_count += keeps[j].size();
            }

            if (soft_nms_enabled_) {
              // Re-map scores to the updated SoftNMS scores
              new (&scores) Eigen::Map<const ERArrXXf>(
                  soft_nms_scores.data(),
                  soft_nms_scores.rows(),
                  soft_nms_scores.cols());
            }

            // Limit to max_per_image detections *over all classes*
            if (detections_per_im_ > 0 && total_keep_count > detections_per_im_) {
              // merge all scores (represented by indices) together and sort
              auto get_all_scores_sorted = [&]() {
                // flatten keeps[i][j] to [pair(i, keeps[i][j]), ...]
                // first: class index (1 ~ keeps.size() - 1),
                // second: values in keeps[first]
                using KeepIndex = std::pair<int, int>;
                vector<KeepIndex> ret(total_keep_count);

                int ret_idx = 0;
                for (int j = 1; j < num_classes; j++) {
                  auto& cur_keep = keeps[j];
                  for (auto& ckv : cur_keep) {
                    ret[ret_idx++] = {j, ckv};
                  }
                }

                std::sort(
                    ret.data(),
                    ret.data() + ret.size(),
                    [this, &scores](const KeepIndex& lhs, const KeepIndex& rhs) {
                      return scores(lhs.second, this->get_score_cls_index(lhs.first)) >
                          scores(rhs.second, this->get_score_cls_index(rhs.first));
                    });

                return ret;
              };

              // Pick the first `detections_per_im_` boxes with highest scores
              auto all_scores_sorted = get_all_scores_sorted();
              DCHECK_GT(all_scores_sorted.size(), detections_per_im_);

              // Reconstruct keeps from `all_scores_sorted`
              for (auto& cur_keep : keeps) {
                cur_keep.clear();
              }
              for (int i = 0; i < detections_per_im_; i++) {
                DCHECK_GT(all_scores_sorted.size(), i);
                auto& cur = all_scores_sorted[i];
                keeps[cur.first].push_back(cur.second);
              }
              total_keep_count = detections_per_im_;
            }
            total_keep_per_batch[b] = total_keep_count;

            // Write results
            int cur_start_idx = out_scores->size(0);
            out_scores->Extend(total_keep_count, 50);
            out_boxes->Extend(total_keep_count, 50);
            out_classes->Extend(total_keep_count, 50);

            int cur_out_idx = 0;
            for (int j = 1; j < num_classes; j++) {
              auto cur_scores = scores.col(get_score_cls_index(j));
              auto cur_boxes =
                  boxes.block(0, get_box_cls_index(j) * box_dim, boxes.rows(), box_dim);
              auto& cur_keep = keeps[j];
              Eigen::Map<EArrXf> cur_out_scores(
                  out_scores->template mutable_data<float>() + cur_start_idx +
                      cur_out_idx,
                  cur_keep.size());
              Eigen::Map<ERArrXXf> cur_out_boxes(
                  out_boxes->mutable_data<float>() +
                      (cur_start_idx + cur_out_idx) * box_dim,
                  cur_keep.size(),
                  box_dim);
              Eigen::Map<EArrXf> cur_out_classes(
                  out_classes->template mutable_data<float>() + cur_start_idx +
                      cur_out_idx,
                  cur_keep.size());

              utils::GetSubArray(
                  cur_scores, utils::AsEArrXt(cur_keep), &cur_out_scores);
              utils::GetSubArrayRows(
                  cur_boxes, utils::AsEArrXt(cur_keep), &cur_out_boxes);
              for (int k = 0; k < cur_keep.size(); k++) {
                cur_out_classes[k] =
                    static_cast<float>(j - !output_classes_include_bg_cls_);
              }

              cur_out_idx += cur_keep.size();
            }

            if (out_keeps) {
              out_keeps->Extend(total_keep_count, 50);

              Eigen::Map<EArrXi> out_keeps_arr(
                  out_keeps->template mutable_data<int>() + cur_start_idx,
                  total_keep_count);
              Eigen::Map<EArrXi> cur_out_keeps_size(
                  out_keeps_size->template mutable_data<int>() + b * num_classes,
                  num_classes);

              cur_out_idx = 0;
              for (int j = 0; j < num_classes; j++) {
                out_keeps_arr.segment(cur_out_idx, keeps[j].size()) =
                    utils::AsEArrXt(keeps[j]);
                cur_out_keeps_size[j] = keeps[j].size();
                cur_out_idx += keeps[j].size();
              }
            }

            offset += num_boxes;
          }

          if (OutputSize() > 3) {
            auto* batch_splits_out = Output(3, {batch_size}, at::dtype<float>());
            Eigen::Map<EArrXf> batch_splits_out_map(
                batch_splits_out->template mutable_data<float>(), batch_size);
            batch_splits_out_map =
                Eigen::Map<const EArrXi>(total_keep_per_batch.data(), batch_size)
                    .cast<float>();
          }

          return true;
        */
    }
}

register_cpu_operator!{
    BoxWithNMSLimit, 
    BoxWithNMSLimitOp<CPUContext>
}

num_inputs!{BoxWithNMSLimit, (2,3)}

num_outputs!{BoxWithNMSLimit, (3,6)}

inputs!{BoxWithNMSLimit, 
    0 => ("scores",                        "Scores, size (count, num_classes)"),
    1 => ("boxes",                         "Bounding box for each class, size (count, num_classes * 4). For rotated boxes, this would have an additional angle (in degrees) in the format [<optionaal_batch_id>, ctr_x, ctr_y, w, h, angle]. Size: (count, num_classes * 5)."),
    2 => ("batch_splits",                  "Tensor of shape (batch_size) with each element denoting the number of RoIs/boxes belonging to the corresponding image in batch. Sum should add up to total count of scores/boxes.")
}

outputs!{BoxWithNMSLimit, 
    0 => ("scores",                        "Filtered scores, size (n)"),
    1 => ("boxes",                         "Filtered boxes, size (n, 4). For rotated boxes, size (n, 5), format [ctr_x, ctr_y, w, h, angle]."),
    2 => ("classes",                       "Class id for each filtered score/box, size (n)"),
    3 => ("batch_splits",                  "Output batch splits for scores/boxes after applying NMS"),
    4 => ("keeps",                         "Optional filtered indices, size (n)"),
    5 => ("keeps_size",                    "Optional number of filtered indices per class, size (num_classes)")
}

args!{BoxWithNMSLimit, 
    0 => ("score_thresh",                  "(float) TEST.SCORE_THRESH"),
    1 => ("nms",                           "(float) TEST.NMS"),
    2 => ("detections_per_im",             "(int) TEST.DEECTIONS_PER_IM"),
    3 => ("soft_nms_enabled",              "(bool) TEST.SOFT_NMS.ENABLED"),
    4 => ("soft_nms_method",               "(string) TEST.SOFT_NMS.METHOD"),
    5 => ("soft_nms_sigma",                "(float) TEST.SOFT_NMS.SIGMA"),
    6 => ("soft_nms_min_score_thres",      "(float) Lower bound on updated scores to discard boxes"),
    7 => ("rotated",                       "bool (default false). If true, then boxes (rois and deltas) include angle info to handle rotation. The format will be [ctr_x, ctr_y, width, height, angle (in degrees)].")
}

should_not_do_gradient!{BoxWithNMSLimit}
