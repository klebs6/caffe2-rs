# caffe2-rs

This library is based on the flexible Caffe2 C++ project. 

It provides a high performance and modular rust environment for deep learning.

As of July 1, 2023, the codebase is under
construction: it is ready for interface
experimentation and development work, but not yet
for production.

If you are interested in this project and would
like to contribute, your contribution is both
welcome and appreciated!

The current bottleneck is in the translation of
c++ statements into rust. It is possible to do
this manually. However, it is better to do it
automatically. Some work is being done in this
domain, however there are several challenges to be
overcome.

For more information, see `chomper`:
https://github.com/klebs6/chomper

## Note from original translator (klebs6)

Greetings! I have a few preliminary work-items to
attend for a few months. Once these are done,
I plan to productionize the `chomper`. 

This will include integrations with the
rust-ecosystem compiler tools. 

During the productionization, we will be able to
enable name resolution and type inference during
transpilation (not to mention several performance
wins).

This work will provide a *massive* speedup for the translation. 
Subsequently, caffe2-rs development will be free to progress.

If there are developers out there who would like to contribute to the c++ to rust transpiler development track, I would love to hear from you.
Please feel free to reach out. I am sure the rust ecosystem will be grateful. 
C++ to Rust transpilation is a major bottleneck for many projects. 

## Acknowledgments

This project owes its DNA to the original authors
of the Caffe2 C++ library, without whom, this work
would not exist.

The author of this project would like to thank the
original authors of the Caffe2 C++ library for
their work, as well as the Rust community for
their support.
