crate::ix!();

#[test] fn prelu_op_example() {

    todo!();

    /*

    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "PRelu",
        ["X","Slope"],
        ["Y"],
    )

    workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32))
    print("X:\n", workspace.FetchBlob("X"), "\n")

    workspace.FeedBlob("Slope", np.array([0.1]).astype(np.float32))
    print("Slope:\n", workspace.FetchBlob("Slope"), "\n")

    workspace.RunOperatorOnce(op)
    print("Y:\n", workspace.FetchBlob("Y"))

    X:
     [[ 0.3957382  -0.19725518 -0.26991343]
     [ 1.5513182  -0.27427664 -0.14584002]
     [-0.4121164   0.9292345   0.96426094]]

    Slope:
     [0.1]

    Y:
     [[ 0.3957382  -0.01972552 -0.02699134]
     [ 1.5513182  -0.02742766 -0.014584  ]
     [-0.04121164  0.9292345   0.96426094]]

    */
}
