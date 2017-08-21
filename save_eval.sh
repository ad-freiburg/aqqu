#!/bin/sh

cp eval_out.log eval_logs/eval_out_$1_`git rev-parse HEAD`.log
