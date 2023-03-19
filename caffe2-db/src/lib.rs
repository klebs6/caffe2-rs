#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{create}
x!{cursor}
x!{db}
x!{db_reader}
x!{deser}
x!{minidb}
x!{minidb_cursor}
x!{minidb_txn}
x!{register}
x!{txn}
