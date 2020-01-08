#!/bin/bash
git diff --name-only HEAD | egrep '.*\.c|.*\.h|.*\.*pp'| xargs indent
#EOF
