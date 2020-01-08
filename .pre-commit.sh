#!/bin/bash
git diff --name-only HEAD | egrep '.*\.C|.*\.c|.*\.h|.*\.*pp'
git diff --name-only HEAD | egrep '.*\.C|.*\.c|.*\.h|.*\.*pp'| xargs indent
#EOF
