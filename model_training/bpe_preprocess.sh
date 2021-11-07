TASK=/path/to/data/files/

for SPLIT in train val
do
  for LANG in source target
  do
    python3 multiprocessing_bpe_encoder.py \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$TASK/$SPLIT.$LANG" \
    --outputs "$TASK/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done
