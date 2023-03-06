crate::ix!();

#[test] fn max_op_example() {

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Max",
        ["X", "Y", "Z"],
        ["X"],
    )

    workspace.FeedBlob("X", (np.random.rand(3,3)).astype(np.float32))
    workspace.FeedBlob("Y", (np.random.rand(3,3)).astype(np.float32))
    workspace.FeedBlob("Z", (np.random.rand(3,3)).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    print("Y:", workspace.FetchBlob("Y"))
    print("Z:", workspace.FetchBlob("Z"))
    workspace.RunOperatorOnce(op)
    print("Max:", workspace.FetchBlob("X"))

    X:
    [[0.4496477  0.07061381 0.7139333 ]
     [0.83203    0.05970785 0.72786295]
     [0.75988126 0.04601283 0.32820013]]
    Y:
    [[0.05683139 0.16872478 0.671098  ]
     [0.70739156 0.09878621 0.03416285]
     [0.34087983 0.94986707 0.67263436]]
    Z:
    [[0.48051122 0.07141234 0.85264146]
     [0.77086854 0.22082241 0.13154659]
     [0.42401117 0.995431   0.4263775 ]]
    Max:
    [[0.48051122 0.16872478 0.85264146]
     [0.83203    0.22082241 0.72786295]
     [0.75988126 0.995431   0.67263436]]

    */
}
