crate::ix!();

#[test] fn gather_ranges_op_example() {

    todo!();

    /*
    RANGES dimentions description:
    1: represents list of examples within a batch
    2: represents list features
    3: two values which are start and length or a range (to be applied on DATA)

    Another output LENGTHS represents each example length within OUTPUT

    Example:
      DATA  = [1, 2, 3, 4, 5, 6]
      RANGES = [
        [
          [0, 1],
          [2, 2],
        ],
        [
          [4, 1],
          [5, 1],
        ]
      ]
      OUTPUT = [1, 3, 4, 5, 6]
      LENGTHS = [3, 2]
    */
}
