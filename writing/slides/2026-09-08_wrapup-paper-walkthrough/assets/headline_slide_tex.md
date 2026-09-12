|Family|Method|zero|<30|ramp|cruise DR|cruise FL|all|
|:--|:--|--:|--:|--:|--:|--:|--:|
|Classical (multi-pitch)|NMF harmonic dictionary|\textcolor[HTML]{b02020}{83.7}|\textcolor[HTML]{b02020}{63.3}|\textcolor[HTML]{b02020}{27.6}|\textcolor[HTML]{c05a00}{11.6}|\textcolor[HTML]{b02020}{17.8}|\textcolor[HTML]{b02020}{26.4}|
||OT inverse harmonic clustering|\textcolor[HTML]{b02020}{69.2}|\textcolor[HTML]{b02020}{49.1}|\textcolor[HTML]{b02020}{18.7}|\textcolor[HTML]{c05a00}{14.4}|\textcolor[HTML]{b02020}{18.6}|\textcolor[HTML]{b02020}{24.2}|
|Blind tracker (ours)|no refusal gates|\textcolor[HTML]{b02020}{79.4}|\textcolor[HTML]{b02020}{60.1}|\textcolor[HTML]{b02020}{15.3}|\textcolor[HTML]{1a7f37}{\textbf{0.92}}|\textcolor[HTML]{c05a00}{9.19}|\textcolor[HTML]{b02020}{17.1}|
||gates + refusal → 0|\textcolor[HTML]{1a7f37}{\textbf{0.01}}|\textcolor[HTML]{b02020}{23.4}|\textcolor[HTML]{b02020}{56.8}|\textcolor[HTML]{b02020}{28.2}|\textcolor[HTML]{b02020}{70.4}|\textcolor[HTML]{b02020}{39.8}|
|Multi-pitch L0, R4 (warm-up)|HPPNet|\textcolor[HTML]{b02020}{17.5}|\textcolor[HTML]{b02020}{17.5}|\textcolor[HTML]{b02020}{21.0}|\textcolor[HTML]{8a7a00}{3.81}|\textcolor[HTML]{1a7f37}{1.63}|\textcolor[HTML]{c05a00}{7.77}|
||HarmoF0|\textcolor[HTML]{c05a00}{14.8}|\textcolor[HTML]{b02020}{18.9}|\textcolor[HTML]{b02020}{20.4}|\textcolor[HTML]{c05a00}{11.5}|\textcolor[HTML]{1a7f37}{1.92}|\textcolor[HTML]{c05a00}{10.8}|
||LateDeep|\textcolor[HTML]{b02020}{52.6}|\textcolor[HTML]{b02020}{35.6}|\textcolor[HTML]{b02020}{16.8}|\textcolor[HTML]{1a7f37}{2.96}|\textcolor[HTML]{8a7a00}{4.46}|\textcolor[HTML]{c05a00}{12.6}|
||Basic Pitch|\textcolor[HTML]{b02020}{33.9}|\textcolor[HTML]{b02020}{38.7}|\textcolor[HTML]{b02020}{24.4}|\textcolor[HTML]{b02020}{26.4}|\textcolor[HTML]{b02020}{25.6}|\textcolor[HTML]{b02020}{27.3}|
|Regression, R3|SimpleConv|\textcolor[HTML]{8a7a00}{4.37}|\textcolor[HTML]{c05a00}{\textbf{8.06}}|\textcolor[HTML]{8a7a00}{4.36}|\textcolor[HTML]{1a7f37}{2.27}|\textcolor[HTML]{1a7f37}{1.30}|\textcolor[HTML]{1a7f37}{\textbf{2.77}}|
||Conv+BiGRU|\textcolor[HTML]{8a7a00}{4.46}|\textcolor[HTML]{c05a00}{9.07}|\textcolor[HTML]{8a7a00}{3.88}|\textcolor[HTML]{1a7f37}{2.15}|\textcolor[HTML]{1a7f37}{2.27}|\textcolor[HTML]{1a7f37}{2.96}|
||Transformer|\textcolor[HTML]{8a7a00}{4.91}|\textcolor[HTML]{c05a00}{14.9}|\textcolor[HTML]{8a7a00}{4.58}|\textcolor[HTML]{1a7f37}{2.42}|\textcolor[HTML]{1a7f37}{1.48}|\textcolor[HTML]{8a7a00}{3.23}|
||causal GRU|\textcolor[HTML]{c05a00}{6.82}|\textcolor[HTML]{b02020}{16.3}|\textcolor[HTML]{8a7a00}{5.91}|\textcolor[HTML]{1a7f37}{2.15}|\textcolor[HTML]{1a7f37}{2.28}|\textcolor[HTML]{8a7a00}{3.80}|
|Multi-pitch L2, R4 (warm-up)|Basic Pitch|\textcolor[HTML]{1a7f37}{0.50}|\textcolor[HTML]{b02020}{18.0}|\textcolor[HTML]{b02020}{39.1}|\textcolor[HTML]{b02020}{43.9}|\textcolor[HTML]{c05a00}{9.47}|\textcolor[HTML]{b02020}{27.6}|
||LateDeep|\textcolor[HTML]{8a7a00}{4.98}|\textcolor[HTML]{b02020}{15.8}|\textcolor[HTML]{c05a00}{8.61}|\textcolor[HTML]{1a7f37}{2.32}|\textcolor[HTML]{1a7f37}{1.69}|\textcolor[HTML]{8a7a00}{3.83}|
||HarmoF0|\textcolor[HTML]{1a7f37}{\textbf{0.09}}|\textcolor[HTML]{c05a00}{\textbf{11.5}}|\textcolor[HTML]{8a7a00}{\textbf{3.21}}|\textcolor[HTML]{1a7f37}{2.98}|\textcolor[HTML]{1a7f37}{1.07}|\textcolor[HTML]{1a7f37}{2.45}|
||HPPNet|\textcolor[HTML]{1a7f37}{1.07}|\textcolor[HTML]{c05a00}{14.6}|\textcolor[HTML]{8a7a00}{3.63}|\textcolor[HTML]{1a7f37}{\textbf{2.07}}|\textcolor[HTML]{1a7f37}{\textbf{0.77}}|\textcolor[HTML]{1a7f37}{\textbf{2.27}}|
