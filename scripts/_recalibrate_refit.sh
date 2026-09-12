set -e
P=omnirun-outputs
R=artifacts/stochastic-fit/clips/results
python -m experiments.stochastic_fit.run getr2 --dest $P \
  --key $R/refit5dregon/P2.json=refit-dregon.json \
  --key $R/refit5michaels/P3.json=refit-michaels.json \
  --key $R/accepted/pop-dregon-final-w.json=pop-dregon-final-w.json \
  --key $R/accepted/pop-michaels-final-w.json=pop-michaels-final-w.json
for r in dregon michaels; do
  python -m experiments.stochastic_fit.run popaugment --summary $P/refit-$r.json \
    --carry $P/pop-$r-final-w.json --output $P/refit-$r-aug.json
done
# DREGON: bench merge first (the bench measures what 4 s crops cannot), then
# the render-matched calibration, then the held-out gate.
python -m experiments.stochastic_fit.run popbench --summary $P/refit-dregon-aug.json \
  --output $P/refit-dregon-bench.json --bench-output $P/bench-dregon-refit.json
python -m experiments.stochastic_fit.run poprawgate --summary $P/refit-dregon-bench.json \
  --policy conf/online_mix/rig_fm_5050.yaml --source-index 0 --train-recordings DREGON-frames:free-flight_nosource_room2 DREGON-frames:hovering_nosource_room2 DREGON-frames:updown_nosource_room2 DREGON-frames:rectangle_nosource_room2 DREGON-frames:spinning_nosource_room2 \
  --test-recordings DREGON-frames:free-flight_nosource_room2 DREGON-frames:hovering_nosource_room2 DREGON-frames:updown_nosource_room2 DREGON-frames:rectangle_nosource_room2 DREGON-frames:spinning_nosource_room2 --k-max 64 --bootstrap 200 --seed 670 --draws-per-clip 4 \
  --compact --output /tmp/d-train.json
python -m experiments.stochastic_fit.run popcalibrate --summary $P/refit-dregon-bench.json \
  --raw-npz /tmp/d-train.npz --reference-order 2 --output $P/refit-dregon-final.json
python -m experiments.stochastic_fit.run poprawgate --summary $P/refit-dregon-final.json \
  --policy conf/online_mix/rig_fm_5050.yaml --source-index 0 --train-recordings DREGON-frames:free-flight_nosource_room2 DREGON-frames:hovering_nosource_room2 DREGON-frames:updown_nosource_room2 DREGON-frames:rectangle_nosource_room2 DREGON-frames:spinning_nosource_room2 \
  --test-recordings DREGON-frames:free-flight_nosource_room1 --k-max 64 --bootstrap 1000 --seed 670 --draws-per-clip 4 \
  --compact --output $P/gate-refit-dregon.json
# Michael's: calibration then gate.
python -m experiments.stochastic_fit.run poprawgate --summary $P/refit-michaels-aug.json \
  --policy conf/online_mix/rig_fm_5050.yaml --source-index 1 --train-recordings michaels-frames:FLY125 \
  --test-recordings michaels-frames:FLY125 --k-max 64 --bootstrap 200 --seed 570 --draws-per-clip 4 \
  --compact --output /tmp/m-train.json
python -m experiments.stochastic_fit.run popcalibrate --summary $P/refit-michaels-aug.json \
  --raw-npz /tmp/m-train.npz --reference-order 2 --output $P/refit-michaels-final.json
python -m experiments.stochastic_fit.run poprawgate --summary $P/refit-michaels-final.json \
  --policy conf/online_mix/rig_fm_5050.yaml --source-index 1 --train-recordings michaels-frames:FLY125 \
  --test-recordings michaels-frames:FLY124 --k-max 64 --bootstrap 1000 --seed 570 --draws-per-clip 4 \
  --compact --output $P/gate-refit-michaels.json
python -m experiments.stochastic_fit.run putr2 --prefix $R/recalib \
  --path $P/refit-dregon-final.json --path $P/refit-michaels-final.json \
  --path $P/bench-dregon-refit.json --path $P/gate-refit-dregon.json \
  --path $P/gate-refit-michaels.json
