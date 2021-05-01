#! /bin/bash

NUM_DOCS=30000

tail -n $NUM_DOCS parallel/IITB.en-hi.en | \
    sed 's/"//g' | \
    sed "s/'//g" | \
    sed 's/\t/ /g' \
    > data.en
tail -n $NUM_DOCS parallel/IITB.en-hi.hi | \
    sed 's/"//g' | \
    sed "s/'//g" | \
    sed 's/\t/ /g' \
    > data.hi