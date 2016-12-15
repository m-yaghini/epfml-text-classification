#!/bin/bash
cat "$1/data/train_pos.txt" "$1/data/train_neg.txt" | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c | sed "s/^\s\+//g" | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 > "$1/outputs/train/vocab.txt"
cat ../data/cleaned/test_data.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c | sed "s/^\s\+//g" | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 > "$1/outputs/test/vocab.txt"