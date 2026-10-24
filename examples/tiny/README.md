# Tiny example tasks

Two small DNA classification tasks in the GUE layout (`train.csv`, `dev.csv` and `test.csv` with the columns
`sequence,label`), written by `make_tiny.py`. The script needs only the standard library, and `python make_tiny.py`
rewrites the files byte for byte (see `SHA256SUMS`).

* `clean/`: 600 training, 100 development and 200 test sequences of 200 random bases. Positives carry the motif
  TATAAA at a random position with probability 0.6. There is no leak, and the expected verdict is PASS.
* `leaky/`: the same task, except that 15% of the test items are copies of training items (half of them reverse
  complemented) and 10% are near copies (3% substitutions), each with the label of its training source. The expected
  verdict is FAIL (overlap.exact, overlap.near, overlap.windows).

```
seqleak audit --train clean/train.csv --dev clean/dev.csv --test clean/test.csv --out clean_report.json --fail-on fail
seqleak audit --train leaky/train.csv --dev leaky/dev.csv --test leaky/test.csv --out leaky_report.json --fail-on fail
```

The first command exits with status 0 and the second with status 1 (`--fail-on fail`). Each takes about a second.
