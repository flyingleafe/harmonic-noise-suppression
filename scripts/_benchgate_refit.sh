set -e
P=omnirun-outputs
R=artifacts/stochastic-fit/clips/results
python -m experiments.stochastic_fit.run getr2 --dest $P \
  --key $R/refit5dregon/P2.json=refit-dregon.json \
  --key $R/refit5michaels/P3.json=refit-michaels.json
python -m experiments.stochastic_fit.run popbench --summary $P/refit-dregon.json \
  --output $P/refit-dregon-bench.json --bench-output $P/bench-dregon-refit.json
python -m experiments.stochastic_fit.run poprawgate --summary $P/refit-dregon-bench.json \
  --policy conf/online_mix/rig_fm_5050.yaml --source-index 0 --train-groups dregon_room2 \
  --test-groups dregon_room2 --k-max 64 --bootstrap 200 --seed 670 --draws-per-clip 4 \
  --compact --output /tmp/trainrender.json
python -m experiments.stochastic_fit.run popcalibrate --summary $P/refit-dregon-bench.json \
  --raw-npz /tmp/trainrender.npz --reference-order 2 --output $P/refit-dregon-bench-cal.json
python -m experiments.stochastic_fit.run poprawgate --summary $P/refit-dregon-bench-cal.json \
  --policy conf/online_mix/rig_fm_5050.yaml --source-index 0 --train-groups dregon_room2 \
  --test-groups dregon_room1 --k-max 64 --bootstrap 1000 --seed 670 --draws-per-clip 4 \
  --compact --output $P/gate-refit-dregon.json
python -m experiments.stochastic_fit.run poprawgate --summary $P/refit-michaels.json \
  --policy conf/online_mix/rig_fm_5050.yaml --source-index 1 --train-groups fly125 \
  --test-groups fly124 --k-max 64 --bootstrap 1000 --seed 670 --draws-per-clip 4 \
  --compact --output $P/gate-refit-michaels.json
python -m experiments.stochastic_fit.run putr2 --prefix $R/benchgate \
  --path $P/refit-dregon-bench.json --path $P/refit-dregon-bench-cal.json \
  --path $P/bench-dregon-refit.json --path $P/gate-refit-dregon.json \
  --path $P/gate-refit-michaels.json
