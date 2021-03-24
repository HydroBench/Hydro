#!/bin/sh

echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
 echo 0 | sudo tee /proc/sys/kernel/kptr_restrict
 echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid
 echo 0 | sudo tee /proc/sys/dev/i915/perf_stream_paranoid
 vtune-self-checker.sh
 
