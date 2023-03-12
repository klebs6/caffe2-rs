crate::ix!();

/**
  | @params Xmin initial solution passed
  | and potentiall better solution returns
  | 
  | @params Xmax initial solution passed
  | and potentiall better solution returns
  |
  */
#[inline] pub fn param_search_greedy(
    x:          *const f32,
    n:          i32,
    n_bins:     Option<i32>,
    ratio:      Option<f32>,
    xmin:       &mut f32,
    xmax:       &mut f32,
    bit_rate:   i32)  
{
    let n_bins: i32 = n_bins.unwrap_or(200);
    let ratio: f32 = ratio.unwrap_or(0.16);

    todo!();
    /*
        float stepsize = (Xmax - Xmin) / n_bins;
      int min_bins = n_bins * (1 - ratio);

      vector<float> Xq(N);

      float loss =
          compress_uniform_simplified_(X, N, Xmin, Xmax, Xq.data(), bit_rate);
      float best_loss = loss;

      float cur_min = Xmin;
      float cur_max = Xmax;
      float cur_loss = loss;

      float thr = min_bins * stepsize;
      while (cur_min + thr < cur_max) {
        // move left
        float loss1 = compress_uniform_simplified_(
            X, N, cur_min + stepsize, cur_max, Xq.data(), bit_rate);
        // move right
        float loss2 = compress_uniform_simplified_(
            X, N, cur_min, cur_max - stepsize, Xq.data(), bit_rate);
        if (cur_loss < loss1 && cur_loss < loss2 && cur_loss < best_loss) {
          // found a local optima
          best_loss = cur_loss;
          Xmin = cur_min;
          Xmax = cur_max;
        }
        if (loss1 < loss2) {
          cur_min = cur_min + stepsize;
          cur_loss = loss1;
        } else {
          cur_max = cur_max - stepsize;
          cur_loss = loss2;
        }
      }
    */
}
