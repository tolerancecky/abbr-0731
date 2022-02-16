# abbr-0731
The code is based on [fairseq]. You just need to replace the corresponding code in [fairseq] with our code.

# For train:
fairseq-train \
data/data-bin/fn2abbr \
    --source-lang fzh --target-lang azh \
    --save-dir checkpoints/fn2abbr \
    --task translation_lev \
    --criterion nat_loss \
    --arch abbr_transformer_small \
    --noise random_mask \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.0 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --log-format 'simple' --log-interval 1000 \
    --fixed-validation-seed 7 \
    --max-tokens 8000 \
    --save-interval-updates 1000 \
    --max-update 200000 \
    --keep-last-epochs 5 \
    --batch-size 8

# For inference
fairseq-generate \
   data/data-bin/fn2abbr \
    --gen-subset test \
    --task translation_lev \
    --path checkpoints/fn2abbr/checkpoint_best.pt \
    --iter-decode-max-iter 2 \
    --iter-decode-eos-penalty 0 \
    --beam 1  \
    --print-step \
    --batch-size 1 \
    --results-path data/data-bin/result/fn2abbr \
