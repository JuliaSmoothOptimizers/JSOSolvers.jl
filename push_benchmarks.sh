#!/bin/bash

julia benchmark/send_comment_to_pr.jl -o $org -r $repo -p $pullrequest -c '**Starting benchmarks!**'

LOCAL_BRANCH_NAME="temp_bmark"
git checkout $LOCAL_BRANCH_NAME -- || true

julia benchmark/$1 $repo

if [ "$?" -eq "0" ] ; then
    julia benchmark/send_comment_to_pr.jl -o $org -r $repo -p $pullrequest -c "Benchmark results" -g "gist.json"
else
    ERROR_LOGS="/home/jenkins/benchmarks/$org/$repo/${pullrequest}_${BUILD_NUMBER}_bmark_error.log"
    julia benchmark/send_comment_to_pr.jl -o $org -r $repo -p $pullrequest -c "**An error occured while running $1**" -g $ERROR_LOGS
fi
