crate::ix!();

#[test] fn gather_ranges_to_dense_example1() {

    todo!();

    /*
    Example 1:
      DATA  = [1, 2, 3, 4, 5, 6, 7, 8]
      RANGES = [
        [
          [2, 4],
          [0, 2],
        ],
        [
          [0, 0],
          [6, 2],
        ]
      ]
      lengths = [4, 2]
      OUTPUT[0] = [[3, 4, 5, 6], [0, 0, 0, 0]]
      OUTPUT[1] = [[1, 2], [7, 8]]
    */
}

/**
  | Contrast Example 2 with Example 1. For
  | each data point per feature, the values
  | are sorted by the corresponding KEY.
  |
  */
#[test] fn gather_ranges_to_dense_example2() {

    todo!();

    /*
    Example 2 (with KEY):
    DATA  = [1, 2, 3, 4, 5, 6, 7, 8]
    KEY   = [0, 1, 3, 2, 1, 0, 1, 0]
    RANGES = [
      [
        [2, 4],
        [0, 2],
      ],
      [
        [0, 0],
        [6, 2],
      ]
    ]
    lengths = [4, 2]
    OUTPUT[0] = [[6, 5, 4, 3], [0, 0, 0, 0]]
    OUTPUT[1] = [[1, 2], [8, 7]]
    */
}
