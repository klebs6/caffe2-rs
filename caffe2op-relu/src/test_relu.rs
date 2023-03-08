crate::ix!();

#[test] fn relu_functor_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
      "Relu",
      ["X"],
      ["Y"]
      )

    workspace.FeedBlob("X", np.random.randn(4, 4).astype(np.float32)) // NCHW
    print("X:\n", workspace.FetchBlob("X"), "\n")

    workspace.RunOperatorOnce(op)
    print("Y:\n", workspace.FetchBlob("Y"))

    X:
     [[-1.4655551   0.64575136  0.7921748   0.4150579 ]
     [ 0.41085166 -0.2837964   0.9881425  -1.9300346 ]
     [ 0.39705405  0.44639114  0.9940703   0.2926532 ]
     [-0.6726489   0.01330667  1.101319    0.33858967]]

    Y:
     [[0.         0.64575136 0.7921748  0.4150579 ]
     [0.41085166 0.         0.9881425  0.        ]
     [0.39705405 0.44639114 0.9940703  0.2926532 ]
     [0.         0.01330667 1.101319   0.33858967]]
    */
}
