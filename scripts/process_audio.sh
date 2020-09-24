#!/bin/bash
for i in {0..9}
do
    python scripts/process_audio.py $i
done