#! /bin/bash

SRC=$(grep "SRC=" ../../.config | cut -d '=' -f 2)
TGT=$(grep "TGT=" ../../.config | cut -d '=' -f 2)

DOC_LIMIT=50000     # Stops concatenation when these many documents are reached.

root_path="finalrepo"
train_path="$root_path/train"
dev_path="$root_path/dev"
test_path="$root_path/test"

touch train.$SRC train.$TGT
touch dev.$SRC dev.$TGT
touch test.$SRC test.$TGT

echo "Concatenating $SRC-$TGT datasets in $root_path..."

num_docs=0          # Number of docs in concat file.
for repo in $train_path/*/$SRC-$TGT; do
    num_docs_curr=$(wc -l $repo/train.$SRC | sed 's/^[[:space:]]\{1,\}//g' | cut -d ' ' -f 1)   # Number of docs in current file.
    num_docs=$(( $num_docs + $num_docs_curr ))
    # Not including this file if total number of documents in concatted file would exceed DOC_LIMIT.
    if [ $num_docs -gt $DOC_LIMIT ]; then
        printf "\tCould not add $repo because of document limit of $DOC_LIMIT.\n"
        num_docs=$(( $num_docs - $num_docs_curr ))
        continue
    fi

    # First sed replaces \t with space (clashes with sep of pd.read_csv).
    # Second sed deletes double quotes (clashes with Python ").
    # Third sed remove empty lines (Pandas skips empty lines).
    cat "$repo/train.$SRC" | sed 's/\t\t*/ /g' | sed 's/"//g' | sed '/^[[:space:]]*$/d' >> train.$SRC
    cat "$repo/train.$TGT" | sed 's/\t\t*/ /g' | sed 's/"//g' | sed '/^[[:space:]]*$/d' >> train.$TGT
    printf "\tAdded $repo/train.[$SRC,$TGT] to train.[$SRC,$TGT].\n"
done

cat "$dev_path/dev.$SRC" | sed 's/\t\t*/ /g' | sed 's/"//g' | sed '/^[[:space:]]*$/d' >> dev.$SRC
cat "$dev_path/dev.$TGT" | sed 's/\t\t*/ /g' | sed 's/"//g' | sed '/^[[:space:]]*$/d' >> dev.$TGT
printf "\tAdded $dev_path/dev.[$SRC,$TGT] to dev.[$SRC,$TGT].\n"

cat "$test_path/test.$SRC" | sed 's/\t\t*/ /g' | sed 's/"//g' | sed '/^[[:space:]]*$/d' >> test.$SRC
cat "$test_path/test.$TGT" | sed 's/\t\t*/ /g' | sed 's/"//g' | sed '/^[[:space:]]*$/d' >> test.$TGT
printf "\tAdded $test_path/test.[$SRC,$TGT] to test.[$SRC,$TGT].\n"