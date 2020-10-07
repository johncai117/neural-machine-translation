#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "sh run.sh [vocab|train|decode] (options)"
		exit 1
fi

if [ "$1" = "train" ]; then
	python -u main.py --mode train --train_src ./data/train.fr --train_tgt ./data/train.en --dev_src ./data/dev.fr --dev_tgt ./data/dev.en --vocab_file vocab.json
elif [ "$1" = "decode" ]; then
    if [ "$#" -lt 2 ]; then
		echo "sh run.sh decode [dev|test]"
		exit 1
	fi
    if [ "$2" = "dev" ]; then
        python main.py --mode decode --model_path model.bin --test_src ./data/dev.fr --test_tgt ./data/dev.en --output_file dev.out
    elif [ "$2" = "test" ]; then
        python main.py --mode decode --model_path model.bin --test_src ./data/test.fr --output_file test.out
    else
        echo "Invalid option selected"
    fi
elif [ "$1" = "vocab" ]; then
	python vocab.py --train_src ./data/train.fr --train_tgt ./data/train.en --vocab_file vocab.json
else
	echo "Invalid option selected"
fi
