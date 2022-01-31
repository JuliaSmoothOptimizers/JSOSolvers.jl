#!/bin/bash

set -e
export tuningFile="$1"
export org="$2"
export repo="$3"
export pullrequest="$4"
julia --project=tuning tuning/${tuningFile} &> "$org"_"$repo"_"$pullrequest".txt
