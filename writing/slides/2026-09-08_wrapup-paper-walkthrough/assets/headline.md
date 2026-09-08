|Family|Method|zero|<30|ramp|cruise DR|cruise FL|all|
|:--|:--|--:|--:|--:|--:|--:|--:|
|Classical (greedy multi-rotor)|pYIN|90.8|75.1|46.4|13.0|62.7|43.0|
||cepstrum|95.5|80.2|38.2|21.1|27.8|36.7|
||HPS|63.9|45.4|20.6|24.3|16.7|27.5|
||matched comb filter|87.2|73.8|42.4|28.0|41.8|42.8|
|Classical (multi-pitch)|NMF harmonic dictionary|83.7|63.3|27.6|11.6|17.8|26.4|
||OT inverse harmonic clustering|69.2|49.1|18.7|14.4|18.6|24.2|
|Blind tracker (ours)|no refusal gates|79.4|60.1|15.3|**0.92**|9.19|17.1|
||gates + refusal → 0|**0.01**|23.4|56.8|28.2|70.4|39.8|
|Multi-pitch L0, R4 (warm-up)|HPPNet|17.5|17.5|21.0|3.81|1.63|7.77|
||HarmoF0|14.8|18.9|20.4|11.5|1.92|10.8|
||LateDeep|52.6|35.6|16.8|2.96|4.46|12.6|
||Basic Pitch|33.9|38.7|24.4|26.4|25.6|27.3|
|Regression, R3|SimpleConv|4.37|**8.06**|4.36|2.27|1.30|**2.77**|
||Conv+BiGRU|4.46|9.07|3.88|2.15|2.27|2.96|
||Transformer|4.91|14.9|4.58|2.42|1.48|3.23|
||causal GRU|6.82|16.3|5.91|2.15|2.28|3.80|
|Regression, R4 (warm-up)|SimpleConv†|3.92|8.68|4.88|2.93|2.00|3.28|
||Conv+BiGRU|3.30|10.8|**3.63**|2.62|1.25|2.77|
||Transformer|4.65|13.1|4.53|2.79|**1.21**|3.21|
||causal GRU|6.05|13.8|7.77|2.13|3.64|4.22|
|Multi-pitch L2, R4 (warm-up)|Basic Pitch|0.50|18.0|39.1|43.9|9.47|27.6|
||LateDeep|4.98|15.8|8.61|2.32|1.69|3.83|
||HarmoF0|**0.09**|**11.5**|**3.21**|2.98|1.07|2.45|
||HPPNet|1.07|14.6|3.63|**2.07**|**0.77**|**2.27**|
