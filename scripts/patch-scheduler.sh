#!/bin/bash
git apply ./patches/scheduler-csi-capacity-1.patch
git apply ./patches/scheduler-csi-capacity-2.patch

# diff -u eventhandlers.go eventhandlers-new.go > changes.patch
git apply ./patches/scheduler-csi-capacity-3.patch
git apply ./patches/scheduler-pdb-1.patch

# diff -u original_file.go modified_file.go > changes.patch
git apply ./patches/scheduler-pdb-2.patch

# change `findNodesThatFitPod` to public method for scheduler simulation 
git apply ./patches/scheduler-sched-one.patch