#! /bin/bash

en_file="data.en"
hi_file="data.hi"

rm -f $en_file; touch $en_file
rm -f $hi_file; touch $hi_file

awk 'BEGIN { FS="\t" } { print $1 }' hin.txt > $en_file
awk 'BEGIN { FS="\t" } { print $2 }' hin.txt > $hi_file