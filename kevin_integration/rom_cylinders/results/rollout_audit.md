# ROM Rollout Audit

- Checkpoint: `kevin_integration/rom_cylinders/checkpoints/baselines/ridge_core_seed42/best_checkpoint.pth`
- Split: `val`
- Split scheme: `original`
- Rollout mode: `predicted_pressure`
- Verified feature indices: `Extension=8:12`, `dP=4:8`
- Verdict: **verified model instability or distribution-shift sensitivity under this rollout mode**

- Update rule: raw `next = current + denormalized_delta`; normalization is applied only before model inference.

# Sequence run_id=1

## Step 1

- run ID: `1`
- source sample index: `1`
- current raw Extension_1..4: `[ 0.387255,  0.61136,  0.680683,  0.622723]`
- predicted raw DeltaExtension_1..4: `[ 0.00636576,  0.00248091, -0.0182501,  0.00235862]`
- updated raw Extension_1..4: `[ 0.393621,  0.613841,  0.662433,  0.625081]`
- current raw dP_1..4: `[ 9.23958e+07,  7.47664e+07,  6.88897e+07,  1.10072e+08]`
- predicted raw DeltaDP_1..4: `[-6.61184e+06, -5.93072e+07, -5.2091e+07, -3.02629e+06]`
- updated raw dP_1..4: `[ 8.5784e+07,  1.54592e+07,  1.67987e+07,  1.07046e+08]`
- normalized Extension values: `[ 1.68469,  0.808536,  1.90877,  1.78126]`
- normalized dP values: `[ 1.85147,  0.947178,  1.35411,  1.48927]`
- model normalized outputs: `[ 0.983867,  0.323799, -0.736984,  0.14642, -0.347389, -3.19985, -2.13498, -0.106895]`
- denormalized outputs: `[ 0.00636576,  0.00248091, -0.0182501,  0.00235862, -6.61184e+06, -5.93072e+07, -5.2091e+07, -3.02629e+06]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `False`

## Step 2

- run ID: `1`
- source sample index: `2`
- current raw Extension_1..4: `[ 0.393621,  0.613841,  0.662433,  0.625081]`
- predicted raw DeltaExtension_1..4: `[ 0.00634409,  0.00244356, -0.0183496,  0.00234748]`
- updated raw Extension_1..4: `[ 0.399965,  0.616285,  0.644083,  0.627429]`
- current raw dP_1..4: `[ 8.5784e+07,  1.54592e+07,  1.67987e+07,  1.07046e+08]`
- predicted raw DeltaDP_1..4: `[ 9.6935e+06,  5.56117e+07,  1.31238e+07, -1.09684e+07]`
- updated raw dP_1..4: `[ 9.54775e+07,  7.10709e+07,  2.99225e+07,  9.60771e+07]`
- normalized Extension values: `[ 1.75976,  0.837589,  1.6639,  1.82402]`
- normalized dP values: `[ 1.75423,  0.0366826,  0.318631,  1.44589]`
- model normalized outputs: `[ 0.980607,  0.318948, -0.741029,  0.145772,  0.516056,  3.00051,  0.540609, -0.405199]`
- denormalized outputs: `[ 0.00634409,  0.00244356, -0.0183496,  0.00234748,  9.6935e+06,  5.56117e+07,  1.31238e+07, -1.09684e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `False`

## Step 3

- run ID: `1`
- source sample index: `3`
- current raw Extension_1..4: `[ 0.399965,  0.616285,  0.644083,  0.627429]`
- predicted raw DeltaExtension_1..4: `[ 0.00636146,  0.00256147, -0.016665,  0.00236156]`
- updated raw Extension_1..4: `[ 0.406327,  0.618846,  0.627418,  0.62979]`
- current raw dP_1..4: `[ 9.54775e+07,  7.10709e+07,  2.99225e+07,  9.60771e+07]`
- predicted raw DeltaDP_1..4: `[-1.24872e+06, -2.46153e+07, -1.27753e+07, -3.05002e+07]`
- updated raw dP_1..4: `[ 9.42288e+07,  4.64556e+07,  1.71472e+07,  6.55769e+07]`
- normalized Extension values: `[ 1.83457,  0.866205,  1.41769,  1.86658]`
- normalized dP values: `[ 1.89679,  0.890444,  0.57951,  1.28868]`
- model normalized outputs: `[ 0.98322,  0.33426, -0.672557,  0.146591, -0.0633864, -1.32807, -0.521963, -1.13881]`
- denormalized outputs: `[ 0.00636146,  0.00256147, -0.016665,  0.00236156, -1.24872e+06, -2.46153e+07, -1.27753e+07, -3.05002e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `False`

## Step 4

- run ID: `1`
- source sample index: `4`
- current raw Extension_1..4: `[ 0.406327,  0.618846,  0.627418,  0.62979]`
- predicted raw DeltaExtension_1..4: `[ 0.00635364,  0.00262217, -0.0159432,  0.00259172]`
- updated raw Extension_1..4: `[ 0.41268,  0.621468,  0.611475,  0.632382]`
- current raw dP_1..4: `[ 9.42288e+07,  4.64556e+07,  1.71472e+07,  6.55769e+07]`
- predicted raw DeltaDP_1..4: `[-419847,  9.91695e+06,  2.63464e+06, -4.97732e+07]`
- updated raw dP_1..4: `[ 9.38089e+07,  5.63726e+07,  1.97818e+07,  1.58037e+07]`
- normalized Extension values: `[ 1.90959,  0.896201,  1.19409,  1.90939]`
- normalized dP values: `[ 1.87842,  0.512546,  0.325558,  0.851511]`
- model normalized outputs: `[ 0.982044,  0.342144, -0.643217,  0.159984, -0.0194935,  0.535087,  0.110267, -1.8627]`
- denormalized outputs: `[ 0.00635364,  0.00262217, -0.0159432,  0.00259172, -419847,  9.91695e+06,  2.63464e+06, -4.97732e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `False`

## Step 5

- run ID: `1`
- source sample index: `5`
- current raw Extension_1..4: `[ 0.41268,  0.621468,  0.611475,  0.632382]`
- predicted raw DeltaExtension_1..4: `[ 0.00634976,  0.00265159, -0.0150491,  0.00250273]`
- updated raw Extension_1..4: `[ 0.41903,  0.62412,  0.596426,  0.634885]`
- current raw dP_1..4: `[ 9.38089e+07,  5.63726e+07,  1.97818e+07,  1.58037e+07]`
- predicted raw DeltaDP_1..4: `[ 1.48816e+06, -7.46178e+06, -987899, -3.19869e+07]`
- updated raw dP_1..4: `[ 9.52971e+07,  4.89108e+07,  1.87939e+07, -1.61832e+07]`
- normalized Extension values: `[ 1.98452,  0.926909,  0.980174,  1.95638]`
- normalized dP values: `[ 1.87225,  0.664793,  0.377931,  0.138096]`
- model normalized outputs: `[ 0.98146,  0.345964, -0.606876,  0.154806,  0.0815445, -0.40257, -0.0383566, -1.19465]`
- denormalized outputs: `[ 0.00634976,  0.00265159, -0.0150491,  0.00250273,  1.48816e+06, -7.46178e+06, -987899, -3.19869e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `False`

## Step 6

- run ID: `1`
- source sample index: `6`
- current raw Extension_1..4: `[ 0.41903,  0.62412,  0.596426,  0.634885]`
- predicted raw DeltaExtension_1..4: `[ 0.00636318,  0.00267391, -0.014278,  0.00263231]`
- updated raw Extension_1..4: `[ 0.425393,  0.626794,  0.582148,  0.637517]`
- current raw dP_1..4: `[ 9.52971e+07,  4.89108e+07,  1.87939e+07, -1.61832e+07]`
- predicted raw DeltaDP_1..4: `[-2.07992e+06,  4.85552e+06,  1.34462e+07, -2.96435e+07]`
- updated raw dP_1..4: `[ 9.32172e+07,  5.37663e+07,  3.22402e+07, -4.58267e+07]`
- normalized Extension values: `[ 2.0594,  0.95796,  0.778253,  2.00175]`
- normalized dP values: `[ 1.89413,  0.550238,  0.358293, -0.320382]`
- model normalized outputs: `[ 0.983479,  0.348862, -0.575533,  0.162346, -0.107402,  0.262001,  0.55384, -1.10663]`
- denormalized outputs: `[ 0.00636318,  0.00267391, -0.014278,  0.00263231, -2.07992e+06,  4.85552e+06,  1.34462e+07, -2.96435e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `False`

## Step 7

- run ID: `1`
- source sample index: `7`
- current raw Extension_1..4: `[ 0.425393,  0.626794,  0.582148,  0.637517]`
- predicted raw DeltaExtension_1..4: `[ 0.00635895,  0.00270162, -0.0133438,  0.00269249]`
- updated raw Extension_1..4: `[ 0.431752,  0.629495,  0.568804,  0.64021]`
- current raw dP_1..4: `[ 9.32172e+07,  5.37663e+07,  3.22402e+07, -4.58267e+07]`
- predicted raw DeltaDP_1..4: `[-1.83435e+06, -3.77311e+06,  1.07137e+07, -1.68481e+07]`
- updated raw dP_1..4: `[ 9.13828e+07,  4.99932e+07,  4.29539e+07, -6.26747e+07]`
- normalized Extension values: `[ 2.13444,  0.989274,  0.586678,  2.04948]`
- normalized dP values: `[ 1.86355,  0.624781,  0.625582, -0.745272]`
- model normalized outputs: `[ 0.982842,  0.352462, -0.537561,  0.165848, -0.094398, -0.203551,  0.441729, -0.626038]`
- denormalized outputs: `[ 0.00635895,  0.00270162, -0.0133438,  0.00269249, -1.83435e+06, -3.77311e+06,  1.07137e+07, -1.68481e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `False`

## Step 8

- run ID: `1`
- source sample index: `8`
- current raw Extension_1..4: `[ 0.431752,  0.629495,  0.568804,  0.64021]`
- predicted raw DeltaExtension_1..4: `[ 0.00635344,  0.00271362, -0.0124306,  0.00282675]`
- updated raw Extension_1..4: `[ 0.438105,  0.632209,  0.556373,  0.643036]`
- current raw dP_1..4: `[ 9.13828e+07,  4.99932e+07,  4.29539e+07, -6.26747e+07]`
- predicted raw DeltaDP_1..4: `[-2.69459e+06,  3.77526e+06,  1.93991e+07, -1.21913e+07]`
- updated raw dP_1..4: `[ 8.86882e+07,  5.37685e+07,  6.2353e+07, -7.4866e+07]`
- normalized Extension values: `[ 2.20942,  1.02091,  0.407639,  2.09829]`
- normalized dP values: `[ 1.83657,  0.566855,  0.838553, -0.98676]`
- model normalized outputs: `[ 0.982014,  0.354019, -0.50044,  0.173661, -0.139952,  0.203716,  0.79807, -0.451131]`
- denormalized outputs: `[ 0.00635344,  0.00271362, -0.0124306,  0.00282675, -2.69459e+06,  3.77526e+06,  1.93991e+07, -1.21913e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `False`

## Step 9

- run ID: `1`
- source sample index: `9`
- current raw Extension_1..4: `[ 0.438105,  0.632209,  0.556373,  0.643036]`
- predicted raw DeltaExtension_1..4: `[ 0.00634126,  0.00273722, -0.0113205,  0.00289666]`
- updated raw Extension_1..4: `[ 0.444447,  0.634946,  0.545053,  0.645933]`
- current raw dP_1..4: `[ 8.86882e+07,  5.37685e+07,  6.2353e+07, -7.4866e+07]`
- predicted raw DeltaDP_1..4: `[-3.97445e+06,  3.70172e+06,  5.23617e+07, -3.81307e+06]`
- updated raw dP_1..4: `[ 8.47138e+07,  5.74702e+07,  1.14715e+08, -7.86791e+07]`
- normalized Extension values: `[ 2.28435,  1.05269,  0.240853,  2.14954]`
- normalized dP values: `[ 1.79694,  0.624814,  1.22418, -1.1615]`
- model normalized outputs: `[ 0.980181,  0.357085, -0.45532,  0.177728, -0.207726,  0.199748,  2.15044, -0.136446]`
- denormalized outputs: `[ 0.00634126,  0.00273722, -0.0113205,  0.00289666, -3.97445e+06,  3.70172e+06,  5.23617e+07, -3.81307e+06]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `False`

## Step 10

- run ID: `1`
- source sample index: `10`
- current raw Extension_1..4: `[ 0.444447,  0.634946,  0.545053,  0.645933]`
- predicted raw DeltaExtension_1..4: `[ 0.00631824,  0.00277288, -0.0105461,  0.00283774]`
- updated raw Extension_1..4: `[ 0.450765,  0.637719,  0.534507,  0.648771]`
- current raw dP_1..4: `[ 8.47138e+07,  5.74702e+07,  1.14715e+08, -7.86791e+07]`
- predicted raw DeltaDP_1..4: `[ 2.18882e+07, -2.7511e+07, -3.60895e+07,  3.67722e+07]`
- updated raw dP_1..4: `[ 1.06602e+08,  2.99592e+07,  7.86252e+07, -4.19069e+07]`
- normalized Extension values: `[ 2.35913,  1.08474,  0.0889615,  2.20206]`
- normalized dP values: `[ 1.7385,  0.681643,  2.26504, -1.21616]`
- model normalized outputs: `[ 0.97672,  0.361716, -0.423843,  0.1743,  1.16182, -1.48431, -1.47849,  1.38793]`
- denormalized outputs: `[ 0.00631824,  0.00277288, -0.0105461,  0.00283774,  2.18882e+07, -2.7511e+07, -3.60895e+07,  3.67722e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `False`

## Step 11

- run ID: `1`
- source sample index: `11`
- current raw Extension_1..4: `[ 0.450765,  0.637719,  0.534507,  0.648771]`
- predicted raw DeltaExtension_1..4: `[ 0.00652243,  0.00269656, -0.0133085,  0.00262132]`
- updated raw Extension_1..4: `[ 0.457287,  0.640416,  0.521198,  0.651392]`
- current raw dP_1..4: `[ 1.06602e+08,  2.99592e+07,  7.86252e+07, -4.19069e+07]`
- predicted raw DeltaDP_1..4: `[ 2.1356e+06,  2.18211e+07, -1.21413e+07,  9.64573e+07]`
- updated raw dP_1..4: `[ 1.08738e+08,  5.17803e+07,  6.64839e+07,  5.45504e+07]`
- normalized Extension values: `[ 2.43364,  1.11722, -0.0525397,  2.25351]`
- normalized dP values: `[ 2.06039,  0.259288,  1.54764, -0.689089]`
- model normalized outputs: `[ 1.00743,  0.351804, -0.536126,  0.161706,  0.115829,  1.17737, -0.495954,  3.62969]`
- denormalized outputs: `[ 0.00652243,  0.00269656, -0.0133085,  0.00262132,  2.1356e+06,  2.18211e+07, -1.21413e+07,  9.64573e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `False`

## Step 12

- run ID: `1`
- source sample index: `12`
- current raw Extension_1..4: `[ 0.457287,  0.640416,  0.521198,  0.651392]`
- predicted raw DeltaExtension_1..4: `[ 0.00659493,  0.00258711, -0.014855,  0.00264011]`
- updated raw Extension_1..4: `[ 0.463882,  0.643003,  0.506343,  0.654032]`
- current raw dP_1..4: `[ 1.08738e+08,  5.17803e+07,  6.64839e+07,  5.45504e+07]`
- predicted raw DeltaDP_1..4: `[-5.89914e+06, -1.53059e+07, -1.82641e+07,  1.8805e+08]`
- updated raw dP_1..4: `[ 1.02839e+08,  3.64743e+07,  4.82198e+07,  2.426e+08]`
- normalized Extension values: `[ 2.51055,  1.1488, -0.231106,  2.30103]`
- normalized dP values: `[ 2.09179,  0.59429,  1.30629,  0.693465]`
- model normalized outputs: `[ 1.01833,  0.33759, -0.598986,  0.1628, -0.309648, -0.825797, -0.747154,  7.06988]`
- denormalized outputs: `[ 0.00659493,  0.00258711, -0.014855,  0.00264011, -5.89914e+06, -1.53059e+07, -1.82641e+07,  1.8805e+08]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 13

- run ID: `1`
- source sample index: `13`
- current raw Extension_1..4: `[ 0.463882,  0.643003,  0.506343,  0.654032]`
- predicted raw DeltaExtension_1..4: `[ 0.00661473,  0.00258191, -0.0167401,  0.00284787]`
- updated raw Extension_1..4: `[ 0.470497,  0.645585,  0.489603,  0.65688]`
- current raw dP_1..4: `[ 1.02839e+08,  3.64743e+07,  4.82198e+07,  2.426e+08]`
- predicted raw DeltaDP_1..4: `[-350890,  2.61219e+07, -3.7134e+07,  1.87365e+08]`
- updated raw dP_1..4: `[ 1.02488e+08,  6.25962e+07,  1.10858e+07,  4.29965e+08]`
- normalized Extension values: `[ 2.58832,  1.17909, -0.430422,  2.3489]`
- normalized dP values: `[ 2.00504,  0.359311,  0.943231,  3.38884]`
- model normalized outputs: `[ 1.02131,  0.336915, -0.67561,  0.17489, -0.0158419,  1.40941, -1.52134,  7.04416]`
- denormalized outputs: `[ 0.00661473,  0.00258191, -0.0167401,  0.00284787, -350890,  2.61219e+07, -3.7134e+07,  1.87365e+08]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 14

- run ID: `1`
- source sample index: `14`
- current raw Extension_1..4: `[ 0.470497,  0.645585,  0.489603,  0.65688]`
- predicted raw DeltaExtension_1..4: `[ 0.00661764,  0.00256042, -0.0185662,  0.00265324]`
- updated raw Extension_1..4: `[ 0.477115,  0.648145,  0.471037,  0.659533]`
- current raw dP_1..4: `[ 1.02488e+08,  6.25962e+07,  1.10858e+07,  4.29965e+08]`
- predicted raw DeltaDP_1..4: `[ 7.99082e+06, -8.17685e+06, -7.34368e+07,  2.15221e+08]`
- updated raw dP_1..4: `[ 1.10478e+08,  5.44193e+07, -6.2351e+07,  6.45185e+08]`
- normalized Extension values: `[ 2.66633,  1.20933, -0.655031,  2.40053]`
- normalized dP values: `[ 1.99988,  0.760339,  0.205068,  6.0744]`
- model normalized outputs: `[ 1.02175,  0.334124, -0.749832,  0.163564,  0.425891, -0.441152, -3.01075,  8.09041]`
- denormalized outputs: `[ 0.00661764,  0.00256042, -0.0185662,  0.00265324,  7.99082e+06, -8.17685e+06, -7.34368e+07,  2.15221e+08]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 15

- run ID: `1`
- source sample index: `15`
- current raw Extension_1..4: `[ 0.477115,  0.648145,  0.471037,  0.659533]`
- predicted raw DeltaExtension_1..4: `[ 0.00664722,  0.00255245, -0.0210466,  0.00263506]`
- updated raw Extension_1..4: `[ 0.483762,  0.650698,  0.44999,  0.662168]`
- current raw dP_1..4: `[ 1.10478e+08,  5.44193e+07, -6.2351e+07,  6.45185e+08]`
- predicted raw DeltaDP_1..4: `[ 6.36626e+06,  1.39807e+07, -8.46657e+07,  1.95547e+08]`
- updated raw dP_1..4: `[ 1.16845e+08,  6.84001e+07, -1.47017e+08,  8.40732e+08]`
- normalized Extension values: `[ 2.74437,  1.23931, -0.904141,  2.44863]`
- normalized dP values: `[ 2.11739,  0.634806, -1.25473,  9.15922]`
- model normalized outputs: `[ 1.0262,  0.333089, -0.850655,  0.162506,  0.339862,  0.754344, -3.47145,  7.35148]`
- denormalized outputs: `[ 0.00664722,  0.00255245, -0.0210466,  0.00263506,  6.36626e+06,  1.39807e+07, -8.46657e+07,  1.95547e+08]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 16

- run ID: `1`
- source sample index: `16`
- current raw Extension_1..4: `[ 0.483762,  0.650698,  0.44999,  0.662168]`
- predicted raw DeltaExtension_1..4: `[ 0.00663254,  0.00252706, -0.0236313,  0.00239085]`
- updated raw Extension_1..4: `[ 0.490395,  0.653225,  0.426359,  0.664559]`
- current raw dP_1..4: `[ 1.16845e+08,  6.84001e+07, -1.47017e+08,  8.40732e+08]`
- predicted raw DeltaDP_1..4: `[ 1.90432e+07, -835667, -1.14906e+08,  1.95179e+08]`
- updated raw dP_1..4: `[ 1.35888e+08,  6.75644e+07, -2.61923e+08,  1.03591e+09]`
- normalized Extension values: `[ 2.82276,  1.2692, -1.18653,  2.49641]`
- normalized dP values: `[ 2.21102,  0.849441, -2.93775,  11.9621]`
- model normalized outputs: `[ 1.02399,  0.329792, -0.955712,  0.148295,  1.01116, -0.0450632, -4.71212,  7.33765]`
- denormalized outputs: `[ 0.00663254,  0.00252706, -0.0236313,  0.00239085,  1.90432e+07, -835667, -1.14906e+08,  1.95179e+08]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 17

- run ID: `1`
- source sample index: `17`
- current raw Extension_1..4: `[ 0.490395,  0.653225,  0.426359,  0.664559]`
- predicted raw DeltaExtension_1..4: `[ 0.00666549,  0.00252263, -0.0266871,  0.00226952]`
- updated raw Extension_1..4: `[ 0.49706,  0.655747,  0.399672,  0.666829]`
- current raw dP_1..4: `[ 1.35888e+08,  6.75644e+07, -2.61923e+08,  1.03591e+09]`
- predicted raw DeltaDP_1..4: `[ 9.25698e+06,  1.02268e+07, -1.02429e+08,  1.52426e+08]`
- updated raw dP_1..4: `[ 1.45145e+08,  7.77912e+07, -3.64351e+08,  1.18834e+09]`
- normalized Extension values: `[ 2.90097,  1.2988, -1.5036,  2.53975]`
- normalized dP values: `[ 2.49107,  0.836612, -5.22189,  14.7596]`
- model normalized outputs: `[ 1.02895,  0.329216, -1.07992,  0.141235,  0.49294,  0.551805, -4.20021,  5.73185]`
- denormalized outputs: `[ 0.00666549,  0.00252263, -0.0266871,  0.00226952,  9.25698e+06,  1.02268e+07, -1.02429e+08,  1.52426e+08]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 18

- run ID: `1`
- source sample index: `18`
- current raw Extension_1..4: `[ 0.49706,  0.655747,  0.399672,  0.666829]`
- predicted raw DeltaExtension_1..4: `[ 0.00666735,  0.0025153, -0.0297045,  0.00210406]`
- updated raw Extension_1..4: `[ 0.503727,  0.658263,  0.369967,  0.668933]`
- current raw dP_1..4: `[ 1.45145e+08,  7.77912e+07, -3.64351e+08,  1.18834e+09]`
- predicted raw DeltaDP_1..4: `[ 8.48073e+06, -7.04892e+06, -7.79452e+07,  9.13469e+07]`
- updated raw dP_1..4: `[ 1.53626e+08,  7.07423e+07, -4.42296e+08,  1.27968e+09]`
- normalized Extension values: `[ 2.97958,  1.32834, -1.86168,  2.5809]`
- normalized dP values: `[ 2.6272,  0.993616, -7.258,  16.9444]`
- model normalized outputs: `[ 1.02923,  0.328264, -1.20257,  0.131607,  0.451834, -0.380295, -3.19572,  3.43774]`
- denormalized outputs: `[ 0.00666735,  0.0025153, -0.0297045,  0.00210406,  8.48073e+06, -7.04892e+06, -7.79452e+07,  9.13469e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 19

- run ID: `1`
- source sample index: `19`
- current raw Extension_1..4: `[ 0.503727,  0.658263,  0.369967,  0.668933]`
- predicted raw DeltaExtension_1..4: `[ 0.00669653,  0.00257641, -0.0320694,  0.00234105]`
- updated raw Extension_1..4: `[ 0.510424,  0.660839,  0.337898,  0.671274]`
- current raw dP_1..4: `[ 1.53626e+08,  7.07423e+07, -4.42296e+08,  1.27968e+09]`
- predicted raw DeltaDP_1..4: `[-1.03536e+07,  3.62328e+06,  6.97467e+06,  5.67729e+07]`
- updated raw dP_1..4: `[ 1.43272e+08,  7.43656e+07, -4.35322e+08,  1.33646e+09]`
- normalized Extension values: `[ 3.0582,  1.3578, -2.26023,  2.61905]`
- normalized dP values: `[ 2.75192,  0.885399, -8.80743,  18.2537]`
- model normalized outputs: `[ 1.03362,  0.336201, -1.29869,  0.145397, -0.545531,  0.195516,  0.288327,  2.13915]`
- denormalized outputs: `[ 0.00669653,  0.00257641, -0.0320694,  0.00234105, -1.03536e+07,  3.62328e+06,  6.97467e+06,  5.67729e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 20

- run ID: `1`
- source sample index: `20`
- current raw Extension_1..4: `[ 0.510424,  0.660839,  0.337898,  0.671274]`
- predicted raw DeltaExtension_1..4: `[ 0.00667739,  0.00269067, -0.0318896,  0.00282893]`
- updated raw Extension_1..4: `[ 0.517101,  0.66353,  0.306008,  0.674103]`
- current raw dP_1..4: `[ 1.43272e+08,  7.43656e+07, -4.35322e+08,  1.33646e+09]`
- predicted raw DeltaDP_1..4: `[ 2.83032e+07,  1.02483e+07, -2.11512e+07, -7.68732e+07]`
- updated raw dP_1..4: `[ 1.71575e+08,  8.46139e+07, -4.56473e+08,  1.25958e+09]`
- normalized Extension values: `[ 3.13717,  1.38797, -2.69052,  2.66149]`
- normalized dP values: `[ 2.59966,  0.941025, -8.66878,  19.0674]`
- model normalized outputs: `[ 1.03074,  0.351039, -1.29139,  0.173787,  1.50152,  0.552966, -0.865607, -2.88057]`
- denormalized outputs: `[ 0.00667739,  0.00269067, -0.0318896,  0.00282893,  2.83032e+07,  1.02483e+07, -2.11512e+07, -7.68732e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

# Sequence run_id=2

## Step 1

- run ID: `2`
- source sample index: `432`
- current raw Extension_1..4: `[ 0.394588,  0.609185,  0.704189,  0.545073]`
- predicted raw DeltaExtension_1..4: `[-0.000323337,  0.00153306, -0.0205093, -0.020401]`
- updated raw Extension_1..4: `[ 0.394265,  0.610718,  0.683679,  0.524672]`
- current raw dP_1..4: `[ 7.19132e+07,  6.72634e+07,  7.66912e+07,  8.99924e+07]`
- predicted raw DeltaDP_1..4: `[-6.55368e+07, -5.79066e+07, -5.48799e+07, -4.53076e+07]`
- updated raw dP_1..4: `[ 6.37645e+06,  9.35683e+06,  2.18113e+07,  4.46849e+07]`
- normalized Extension values: `[ 1.77116,  0.783064,  2.22415,  0.373452]`
- normalized dP values: `[ 1.55025,  0.831991,  1.5092,  1.20147]`
- model normalized outputs: `[-0.0222022,  0.200705, -0.828814, -1.17797, -3.46774, -3.12428, -2.24941, -1.69497]`
- denormalized outputs: `[-0.000323337,  0.00153306, -0.0205093, -0.020401, -6.55368e+07, -5.79066e+07, -5.48799e+07, -4.53076e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `False`

## Step 2

- run ID: `2`
- source sample index: `433`
- current raw Extension_1..4: `[ 0.394265,  0.610718,  0.683679,  0.524672]`
- predicted raw DeltaExtension_1..4: `[-0.000355281,  0.00149321, -0.0206258, -0.0203342]`
- updated raw Extension_1..4: `[ 0.393909,  0.612211,  0.663053,  0.504337]`
- current raw dP_1..4: `[ 6.37645e+06,  9.35683e+06,  2.18113e+07,  4.46849e+07]`
- predicted raw DeltaDP_1..4: `[ 5.06591e+07,  5.4663e+07,  4.0248e+06, -4.06959e+07]`
- updated raw dP_1..4: `[ 5.70355e+07,  6.40198e+07,  2.58361e+07,  3.98898e+06]`
- normalized Extension values: `[ 1.76735,  0.801016,  1.94897,  0.00357908]`
- normalized dP values: `[ 0.586457, -0.0570027,  0.418274,  0.552059]`
- model normalized outputs: `[-0.0270067,  0.195531, -0.83355, -1.17408,  2.68537,  2.94933,  0.167302, -1.52176]`
- denormalized outputs: `[-0.000355281,  0.00149321, -0.0206258, -0.0203342,  5.06591e+07,  5.4663e+07,  4.0248e+06, -4.06959e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `False`

## Step 3

- run ID: `2`
- source sample index: `434`
- current raw Extension_1..4: `[ 0.393909,  0.612211,  0.663053,  0.504337]`
- predicted raw DeltaExtension_1..4: `[-0.000157375,  0.00160701, -0.0190889, -0.0203224]`
- updated raw Extension_1..4: `[ 0.393752,  0.613818,  0.643965,  0.484015]`
- current raw dP_1..4: `[ 5.70355e+07,  6.40198e+07,  2.58361e+07,  3.98898e+06]`
- predicted raw DeltaDP_1..4: `[-3.01392e+07, -2.60868e+07, -2.16723e+07, -4.90019e+07]`
- updated raw dP_1..4: `[ 2.68963e+07,  3.7933e+07,  4.16381e+06, -4.50129e+07]`
- normalized Extension values: `[ 1.76316,  0.818503,  1.67223, -0.365083]`
- normalized dP values: `[ 1.33146,  0.782194,  0.49828, -0.0312482]`
- model normalized outputs: `[ 0.00275923,  0.21031, -0.771077, -1.17339, -1.59327, -1.40747, -0.886986, -1.83373]`
- denormalized outputs: `[-0.000157375,  0.00160701, -0.0190889, -0.0203224, -3.01392e+07, -2.60868e+07, -2.16723e+07, -4.90019e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `False`

## Step 4

- run ID: `2`
- source sample index: `435`
- current raw Extension_1..4: `[ 0.393752,  0.613818,  0.643965,  0.484015]`
- predicted raw DeltaExtension_1..4: `[-0.000176601,  0.00165557, -0.0186204, -0.0201377]`
- updated raw Extension_1..4: `[ 0.393575,  0.615474,  0.625344,  0.463877]`
- current raw dP_1..4: `[ 2.68963e+07,  3.7933e+07,  4.16381e+06, -4.50129e+07]`
- predicted raw DeltaDP_1..4: `[-3.21994e+06,  1.01994e+07, -6.87859e+06, -7.247e+07]`
- updated raw dP_1..4: `[ 2.36764e+07,  4.81324e+07, -2.71478e+06, -1.17483e+08]`
- normalized Extension values: `[ 1.7613,  0.837322,  1.4161, -0.73353]`
- normalized dP values: `[ 0.888225,  0.381705,  0.0674699, -0.733608]`
- model normalized outputs: `[-0.000132531,  0.216615, -0.752036, -1.16265, -0.167772,  0.550324, -0.280037, -2.71519]`
- denormalized outputs: `[-0.000176601,  0.00165557, -0.0186204, -0.0201377, -3.21994e+06,  1.01994e+07, -6.87859e+06, -7.247e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `False`

## Step 5

- run ID: `2`
- source sample index: `436`
- current raw Extension_1..4: `[ 0.393575,  0.615474,  0.625344,  0.463877]`
- predicted raw DeltaExtension_1..4: `[-0.00013316,  0.00167229, -0.0179866, -0.020349]`
- updated raw Extension_1..4: `[ 0.393442,  0.617146,  0.607358,  0.443528]`
- current raw dP_1..4: `[ 2.36764e+07,  4.81324e+07, -2.71478e+06, -1.17483e+08]`
- predicted raw DeltaDP_1..4: `[-1.00789e+07, -9.08278e+06, -1.22832e+07, -6.66325e+07]`
- updated raw dP_1..4: `[ 1.35975e+07,  3.90496e+07, -1.4998e+07, -1.84115e+08]`
- normalized Extension values: `[ 1.75922,  0.85671,  1.16627, -1.09863]`
- normalized dP values: `[ 0.840872,  0.538288, -0.0692652, -1.77234]`
- model normalized outputs: `[ 0.00640118,  0.218787, -0.726274, -1.17494, -0.530985, -0.49003, -0.501775, -2.49593]`
- denormalized outputs: `[-0.00013316,  0.00167229, -0.0179866, -0.020349, -1.00789e+07, -9.08278e+06, -1.22832e+07, -6.66325e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `False`

## Step 6

- run ID: `2`
- source sample index: `437`
- current raw Extension_1..4: `[ 0.393442,  0.617146,  0.607358,  0.443528]`
- predicted raw DeltaExtension_1..4: `[-0.000127175,  0.0016792, -0.0175213, -0.0203729]`
- updated raw Extension_1..4: `[ 0.393315,  0.618825,  0.589836,  0.423155]`
- current raw dP_1..4: `[ 1.35975e+07,  3.90496e+07, -1.4998e+07, -1.84115e+08]`
- predicted raw DeltaDP_1..4: `[-4.30359e+06,  3.4892e+06,  1.93176e+06, -8.03622e+07]`
- updated raw dP_1..4: `[ 9.29389e+06,  4.25388e+07, -1.30662e+07, -2.64478e+08]`
- normalized Extension values: `[ 1.75765,  0.876293,  0.924932, -1.46756]`
- normalized dP values: `[ 0.69265,  0.398847, -0.313435, -2.72741]`
- model normalized outputs: `[ 0.00730133,  0.219685, -0.707363, -1.17633, -0.225156,  0.188282,  0.0814297, -3.01162]`
- denormalized outputs: `[-0.000127175,  0.0016792, -0.0175213, -0.0203729, -4.30359e+06,  3.4892e+06,  1.93176e+06, -8.03622e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 7

- run ID: `2`
- source sample index: `438`
- current raw Extension_1..4: `[ 0.393315,  0.618825,  0.589836,  0.423155]`
- predicted raw DeltaExtension_1..4: `[-0.000102052,  0.00168894, -0.0169494, -0.0205189]`
- updated raw Extension_1..4: `[ 0.393213,  0.620514,  0.572887,  0.402637]`
- current raw dP_1..4: `[ 9.29389e+06,  4.25388e+07, -1.30662e+07, -2.64478e+08]`
- predicted raw DeltaDP_1..4: `[-1.00885e+07, -5.45866e+06, -1.89762e+06, -8.19173e+07]`
- updated raw dP_1..4: `[-794623,  3.70802e+07, -1.49638e+07, -3.46395e+08]`
- normalized Extension values: `[ 1.75615,  0.895958,  0.689841, -1.83692]`
- normalized dP values: `[ 0.629361,  0.452414, -0.275035, -3.87927]`
- model normalized outputs: `[ 0.0110801,  0.22095, -0.684116, -1.18483, -0.531495, -0.294493, -0.0756799, -3.07002]`
- denormalized outputs: `[-0.000102052,  0.00168894, -0.0169494, -0.0205189, -1.00885e+07, -5.45866e+06, -1.89762e+06, -8.19173e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 8

- run ID: `2`
- source sample index: `439`
- current raw Extension_1..4: `[ 0.393213,  0.620514,  0.572887,  0.402637]`
- predicted raw DeltaExtension_1..4: `[-0.000130021,  0.00167814, -0.0165041, -0.0206273]`
- updated raw Extension_1..4: `[ 0.393083,  0.622192,  0.556383,  0.382009]`
- current raw dP_1..4: `[-794623,  3.70802e+07, -1.49638e+07, -3.46395e+08]`
- predicted raw DeltaDP_1..4: `[-3.70752e+06,  1.90584e+06,  7.37976e+06, -9.23361e+07]`
- updated raw dP_1..4: `[-4.50214e+06,  3.8986e+07, -7.58409e+06, -4.38731e+08]`
- normalized Extension values: `[ 1.75495,  0.915736,  0.462423, -2.20893]`
- normalized dP values: `[ 0.480999,  0.368612, -0.312757, -5.05341]`
- model normalized outputs: `[ 0.00687338,  0.219547, -0.666015, -1.19113, -0.193591,  0.102853,  0.304947, -3.46135]`
- denormalized outputs: `[-0.000130021,  0.00167814, -0.0165041, -0.0206273, -3.70752e+06,  1.90584e+06,  7.37976e+06, -9.23361e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 9

- run ID: `2`
- source sample index: `440`
- current raw Extension_1..4: `[ 0.393083,  0.622192,  0.556383,  0.382009]`
- predicted raw DeltaExtension_1..4: `[-0.000108112,  0.00167654, -0.0159939, -0.0208271]`
- updated raw Extension_1..4: `[ 0.392975,  0.623869,  0.540389,  0.361182]`
- current raw dP_1..4: `[-4.50214e+06,  3.8986e+07, -7.58409e+06, -4.38731e+08]`
- predicted raw DeltaDP_1..4: `[ 1.0917e+06,  2.16408e+06,  3.8578e+07, -7.3096e+07]`
- updated raw dP_1..4: `[-3.41044e+06,  4.11501e+07,  3.09939e+07, -5.11827e+08]`
- normalized Extension values: `[ 1.75341,  0.935389,  0.240981, -2.58291]`
- normalized dP values: `[ 0.426475,  0.397871, -0.166059, -6.37689]`
- model normalized outputs: `[ 0.0101686,  0.219339, -0.645277, -1.20276,  0.0605498,  0.116786,  1.58493, -2.7387]`
- denormalized outputs: `[-0.000108112,  0.00167654, -0.0159939, -0.0208271,  1.0917e+06,  2.16408e+06,  3.8578e+07, -7.3096e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 10

- run ID: `2`
- source sample index: `441`
- current raw Extension_1..4: `[ 0.392975,  0.623869,  0.540389,  0.361182]`
- predicted raw DeltaExtension_1..4: `[-0.000106049,  0.00162332, -0.0157889, -0.0213584]`
- updated raw Extension_1..4: `[ 0.392869,  0.625492,  0.5246,  0.339824]`
- current raw dP_1..4: `[-3.41044e+06,  4.11501e+07,  3.09939e+07, -5.11827e+08]`
- predicted raw DeltaDP_1..4: `[-1.73252e+07, -1.70614e+07, -4.7048e+07, -1.22903e+08]`
- updated raw dP_1..4: `[-2.07356e+07,  2.40887e+07, -1.60541e+07, -6.3473e+08]`
- normalized Extension values: `[ 1.75214,  0.955023,  0.0263842, -2.96051]`
- normalized dP values: `[ 0.44253,  0.431094,  0.600808, -7.4246]`
- model normalized outputs: `[ 0.0104788,  0.212427, -0.636944, -1.23368, -0.91471, -0.920509, -1.92808, -4.60942]`
- denormalized outputs: `[-0.000106049,  0.00162332, -0.0157889, -0.0213584, -1.73252e+07, -1.70614e+07, -4.7048e+07, -1.22903e+08]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 11

- run ID: `2`
- source sample index: `442`
- current raw Extension_1..4: `[ 0.392869,  0.625492,  0.5246,  0.339824]`
- predicted raw DeltaExtension_1..4: `[-2.35572e-05,  0.00154285, -0.0188503, -0.0226571]`
- updated raw Extension_1..4: `[ 0.392845,  0.627035,  0.50575,  0.317167]`
- current raw dP_1..4: `[-2.07356e+07,  2.40887e+07, -1.60541e+07, -6.3473e+08]`
- predicted raw DeltaDP_1..4: `[ 1.14435e+07,  1.3092e+07, -4.16619e+07, -1.05098e+08]`
- updated raw dP_1..4: `[-9.29215e+06,  3.71808e+07, -5.7716e+07, -7.39828e+08]`
- normalized Extension values: `[ 1.75089,  0.974033, -0.185461, -3.34774]`
- normalized dP values: `[ 0.187744,  0.169165, -0.334429, -9.1862]`
- model normalized outputs: `[ 0.0228859,  0.201978, -0.761379, -1.30925,  0.608725,  0.706395, -1.70711, -3.94069]`
- denormalized outputs: `[-2.35572e-05,  0.00154285, -0.0188503, -0.0226571,  1.14435e+07,  1.3092e+07, -4.16619e+07, -1.05098e+08]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 12

- run ID: `2`
- source sample index: `443`
- current raw Extension_1..4: `[ 0.392845,  0.627035,  0.50575,  0.317167]`
- predicted raw DeltaExtension_1..4: `[ 0.000133971,  0.00143649, -0.0208028, -0.0232257]`
- updated raw Extension_1..4: `[ 0.392979,  0.628472,  0.484947,  0.293941]`
- current raw dP_1..4: `[-9.29215e+06,  3.71808e+07, -5.7716e+07, -7.39828e+08]`
- predicted raw DeltaDP_1..4: `[-519042, -1.98646e+07, -2.81909e+07, -6.36471e+07]`
- updated raw dP_1..4: `[-9.8112e+06,  1.73161e+07, -8.59069e+07, -8.03475e+08]`
- normalized Extension values: `[ 1.75061,  0.992101, -0.438383, -3.75851]`
- normalized dP values: `[ 0.356033,  0.370156, -1.1626, -10.6926]`
- model normalized outputs: `[ 0.0465789,  0.188165, -0.840743, -1.34234, -0.0247464, -1.07176, -1.15442, -2.3838]`
- denormalized outputs: `[ 0.000133971,  0.00143649, -0.0208028, -0.0232257, -519042, -1.98646e+07, -2.81909e+07, -6.36471e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 13

- run ID: `2`
- source sample index: `444`
- current raw Extension_1..4: `[ 0.392979,  0.628472,  0.484947,  0.293941]`
- predicted raw DeltaExtension_1..4: `[ 0.000238232,  0.00142054, -0.0229894, -0.0237436]`
- updated raw Extension_1..4: `[ 0.393217,  0.629892,  0.461957,  0.270197]`
- current raw dP_1..4: `[-9.8112e+06,  1.73161e+07, -8.59069e+07, -8.03475e+08]`
- predicted raw DeltaDP_1..4: `[ 2.45496e+06,  1.77838e+07, -4.78301e+07, -6.17887e+07]`
- updated raw dP_1..4: `[-7.35624e+06,  3.51e+07, -1.33737e+08, -8.65263e+08]`
- normalized Extension values: `[ 1.75219,  1.00892, -0.717503, -4.1796]`
- normalized dP values: `[ 0.3484,  0.0651904, -1.72299, -11.6049]`
- model normalized outputs: `[ 0.0622602,  0.186093, -0.929623, -1.37247,  0.132741,  0.95954, -1.96017, -2.314]`
- denormalized outputs: `[ 0.000238232,  0.00142054, -0.0229894, -0.0237436,  2.45496e+06,  1.77838e+07, -4.78301e+07, -6.17887e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 14

- run ID: `2`
- source sample index: `445`
- current raw Extension_1..4: `[ 0.393217,  0.629892,  0.461957,  0.270197]`
- predicted raw DeltaExtension_1..4: `[ 0.000358778,  0.00137656, -0.0251974, -0.0246598]`
- updated raw Extension_1..4: `[ 0.393576,  0.631269,  0.43676,  0.245537]`
- current raw dP_1..4: `[-7.35624e+06,  3.51e+07, -1.33737e+08, -8.65263e+08]`
- predicted raw DeltaDP_1..4: `[ 8.28063e+06, -1.16336e+07, -7.45817e+07, -1.01308e+07]`
- updated raw dP_1..4: `[ 924396,  2.34664e+07, -2.08319e+08, -8.75394e+08]`
- normalized Extension values: `[ 1.755,  1.02556, -1.02596, -4.61007]`
- normalized dP values: `[ 0.384503,  0.338212, -2.67377, -12.4905]`
- model normalized outputs: `[ 0.0803908,  0.180382, -1.01937, -1.42579,  0.441238, -0.627657, -3.05772, -0.373739]`
- denormalized outputs: `[ 0.000358778,  0.00137656, -0.0251974, -0.0246598,  8.28063e+06, -1.16336e+07, -7.45817e+07, -1.01308e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 15

- run ID: `2`
- source sample index: `446`
- current raw Extension_1..4: `[ 0.393576,  0.631269,  0.43676,  0.245537]`
- predicted raw DeltaExtension_1..4: `[ 0.00050681,  0.00135652, -0.028061, -0.0254201]`
- updated raw Extension_1..4: `[ 0.394083,  0.632625,  0.408699,  0.220117]`
- current raw dP_1..4: `[ 924396,  2.34664e+07, -2.08319e+08, -8.75394e+08]`
- predicted raw DeltaDP_1..4: `[ 6.05892e+06,  8.37149e+06, -8.50905e+07,  6.7465e+06]`
- updated raw dP_1..4: `[ 6.98332e+06,  3.18379e+07, -2.93409e+08, -8.68648e+08]`
- normalized Extension values: `[ 1.75923,  1.04168, -1.36405, -5.05716]`
- normalized dP values: `[ 0.506279,  0.159611, -4.15633, -12.6357]`
- model normalized outputs: `[ 0.102655,  0.17778, -1.13576, -1.47003,  0.323587,  0.451702, -3.48887,  0.260169]`
- denormalized outputs: `[ 0.00050681,  0.00135652, -0.028061, -0.0254201,  6.05892e+06,  8.37149e+06, -8.50905e+07,  6.7465e+06]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 16

- run ID: `2`
- source sample index: `447`
- current raw Extension_1..4: `[ 0.394083,  0.632625,  0.408699,  0.220117]`
- predicted raw DeltaExtension_1..4: `[ 0.000647666,  0.00132161, -0.0310136, -0.0263864]`
- updated raw Extension_1..4: `[ 0.394731,  0.633947,  0.377686,  0.193731]`
- current raw dP_1..4: `[ 6.98332e+06,  3.18379e+07, -2.93409e+08, -8.68648e+08]`
- predicted raw DeltaDP_1..4: `[ 1.3199e+07, -2.13344e+06, -1.09569e+08,  3.84026e+07]`
- updated raw dP_1..4: `[ 2.01824e+07,  2.97045e+07, -4.02978e+08, -8.30245e+08]`
- normalized Extension values: `[ 1.76521,  1.05757, -1.74055, -5.51803]`
- normalized dP values: `[ 0.595382,  0.288131, -5.84779, -12.539]`
- model normalized outputs: `[ 0.123841,  0.173246, -1.25578, -1.52626,  0.70169, -0.115084, -4.49315,  1.44916]`
- denormalized outputs: `[ 0.000647666,  0.00132161, -0.0310136, -0.0263864,  1.3199e+07, -2.13344e+06, -1.09569e+08,  3.84026e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 17

- run ID: `2`
- source sample index: `448`
- current raw Extension_1..4: `[ 0.394731,  0.633947,  0.377686,  0.193731]`
- predicted raw DeltaExtension_1..4: `[ 0.00081153,  0.00131781, -0.0343017, -0.0272995]`
- updated raw Extension_1..4: `[ 0.395542,  0.635265,  0.343384,  0.166431]`
- current raw dP_1..4: `[ 2.01824e+07,  2.97045e+07, -4.02978e+08, -8.30245e+08]`
- predicted raw DeltaDP_1..4: `[ 7.7707e+06,  3.78177e+06, -1.03995e+08,  2.07293e+07]`
- updated raw dP_1..4: `[ 2.79531e+07,  3.34862e+07, -5.06973e+08, -8.09516e+08]`
- normalized Extension values: `[ 1.77284,  1.07304, -2.15667, -5.99641]`
- normalized dP values: `[ 0.789488,  0.255378, -8.02584, -11.9886]`
- model normalized outputs: `[ 0.148487,  0.172753, -1.38943, -1.57939,  0.414234,  0.204067, -4.2645,  0.78536]`
- denormalized outputs: `[ 0.00081153,  0.00131781, -0.0343017, -0.0272995,  7.7707e+06,  3.78177e+06, -1.03995e+08,  2.07293e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 18

- run ID: `2`
- source sample index: `449`
- current raw Extension_1..4: `[ 0.395542,  0.635265,  0.343384,  0.166431]`
- predicted raw DeltaExtension_1..4: `[ 0.000952272,  0.00129378, -0.0373982, -0.028285]`
- updated raw Extension_1..4: `[ 0.396494,  0.636558,  0.305986,  0.138146]`
- current raw dP_1..4: `[ 2.79531e+07,  3.34862e+07, -5.06973e+08, -8.09516e+08]`
- predicted raw DeltaDP_1..4: `[-8.56626e+06, -8.39364e+06, -7.44023e+07,  1.04263e+07]`
- updated raw dP_1..4: `[ 1.93868e+07,  2.50926e+07, -5.81376e+08, -7.99089e+08]`
- normalized Extension values: `[ 1.78241,  1.08847, -2.61691, -6.49136]`
- normalized dP values: `[ 0.903765,  0.313437, -10.0931, -11.6915]`
- model normalized outputs: `[ 0.169655,  0.169632, -1.51529, -1.63674, -0.450884, -0.452849, -3.05036,  0.398382]`
- denormalized outputs: `[ 0.000952272,  0.00129378, -0.0373982, -0.028285, -8.56626e+06, -8.39364e+06, -7.44023e+07,  1.04263e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 19

- run ID: `2`
- source sample index: `450`
- current raw Extension_1..4: `[ 0.396494,  0.636558,  0.305986,  0.138146]`
- predicted raw DeltaExtension_1..4: `[ 0.00104447,  0.00131201, -0.03985, -0.0287384]`
- updated raw Extension_1..4: `[ 0.397539,  0.63787,  0.266136,  0.109408]`
- current raw dP_1..4: `[ 1.93868e+07,  2.50926e+07, -5.81376e+08, -7.99089e+08]`
- predicted raw DeltaDP_1..4: `[-5.27325e+06,  8.66405e+06,  4.41323e+07,  1.40195e+08]`
- updated raw dP_1..4: `[ 1.41135e+07,  3.37566e+07, -5.37243e+08, -6.58895e+08]`
- normalized Extension values: `[ 1.79364,  1.10363, -3.1187, -7.00417]`
- normalized dP values: `[ 0.777789,  0.184576, -11.5721, -11.542]`
- model normalized outputs: `[ 0.183521,  0.172, -1.61495, -1.66312, -0.276504,  0.467488,  1.81281,  5.27246]`
- denormalized outputs: `[ 0.00104447,  0.00131201, -0.03985, -0.0287384, -5.27325e+06,  8.66405e+06,  4.41323e+07,  1.40195e+08]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 20

- run ID: `2`
- source sample index: `451`
- current raw Extension_1..4: `[ 0.397539,  0.63787,  0.266136,  0.109408]`
- predicted raw DeltaExtension_1..4: `[ 0.00101129,  0.00142127, -0.0403934, -0.0288836]`
- updated raw Extension_1..4: `[ 0.39855,  0.639292,  0.225742,  0.0805244]`
- current raw dP_1..4: `[ 1.41135e+07,  3.37566e+07, -5.37243e+08, -6.58895e+08]`
- predicted raw DeltaDP_1..4: `[ 1.11473e+07,  1.2806e+07,  8.11168e+06,  1.098e+08]`
- updated raw dP_1..4: `[ 2.52608e+07,  4.65626e+07, -5.29132e+08, -5.49095e+08]`
- normalized Extension values: `[ 1.80596,  1.11899, -3.65338, -7.5252]`
- normalized dP values: `[ 0.70024,  0.317588, -10.6948, -9.53257]`
- model normalized outputs: `[ 0.178531,  0.186188, -1.63704, -1.67157,  0.593039,  0.690965,  0.334976,  4.13083]`
- denormalized outputs: `[ 0.00101129,  0.00142127, -0.0403934, -0.0288836,  1.11473e+07,  1.2806e+07,  8.11168e+06,  1.098e+08]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

# Sequence run_id=3

## Step 1

- run ID: `3`
- source sample index: `863`
- current raw Extension_1..4: `[ 0.256738,  0.589645,  0.74574,  0.636782]`
- predicted raw DeltaExtension_1..4: `[-0.00593332,  0.00752207, -0.0137241, -0.019995]`
- updated raw Extension_1..4: `[ 0.250805,  0.597167,  0.732016,  0.616787]`
- current raw dP_1..4: `[-8.01009e+07,  9.15183e+07,  5.82381e+07,  7.72881e+07]`
- predicted raw DeltaDP_1..4: `[-840994, -952996, -5.2058e+07, -6.00945e+07]`
- updated raw dP_1..4: `[-8.09419e+07,  9.05653e+07,  6.18016e+06,  1.71937e+07]`
- normalized Extension values: `[ 0.145555,  0.554233,  2.78166,  2.03616]`
- normalized dP values: `[-0.685288,  1.20436,  1.14238,  1.01937]`
- model normalized outputs: `[-0.865967,  0.978469, -0.553017, -1.15434, -0.0417952, -0.0513936, -2.13363, -2.25036]`
- denormalized outputs: `[-0.00593332,  0.00752207, -0.0137241, -0.019995, -840994, -952996, -5.2058e+07, -6.00945e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `False`

## Step 2

- run ID: `3`
- source sample index: `864`
- current raw Extension_1..4: `[ 0.250805,  0.597167,  0.732016,  0.616787]`
- predicted raw DeltaExtension_1..4: `[-0.00591636,  0.00751337, -0.0137437, -0.0198083]`
- updated raw Extension_1..4: `[ 0.244889,  0.60468,  0.718272,  0.596979]`
- current raw dP_1..4: `[-8.09419e+07,  9.05653e+07,  6.18016e+06,  1.71937e+07]`
- predicted raw DeltaDP_1..4: `[-3.0702e+06,  7.4471e+06,  2.313e+07, -1.38837e+07]`
- updated raw dP_1..4: `[-8.40121e+07,  9.80124e+07,  2.93102e+07,  3.30994e+06]`
- normalized Extension values: `[ 0.0755858,  0.642321,  2.59752,  1.67365]`
- normalized dP values: `[-0.697656,  1.18973,  0.107552,  0.158019]`
- model normalized outputs: `[-0.863417,  0.977338, -0.553815, -1.14348, -0.159842,  0.401828,  0.951141, -0.514698]`
- denormalized outputs: `[-0.00591636,  0.00751337, -0.0137437, -0.0198083, -3.0702e+06,  7.4471e+06,  2.313e+07, -1.38837e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `False`

## Step 3

- run ID: `3`
- source sample index: `865`
- current raw Extension_1..4: `[ 0.244889,  0.60468,  0.718272,  0.596979]`
- predicted raw DeltaExtension_1..4: `[-0.00594108,  0.00756998, -0.0121935, -0.0193931]`
- updated raw Extension_1..4: `[ 0.238948,  0.61225,  0.706078,  0.577586]`
- current raw dP_1..4: `[-8.40121e+07,  9.80124e+07,  2.93102e+07,  3.30994e+06]`
- predicted raw DeltaDP_1..4: `[-4.93142e+06,  2.45453e+06,  6.58174e+06, -3.92844e+07]`
- updated raw dP_1..4: `[-8.89435e+07,  1.00467e+08,  3.5892e+07, -3.59744e+07]`
- normalized Extension values: `[ 0.00581645,  0.730307,  2.41311,  1.31452]`
- normalized dP values: `[-0.742807,  1.30406,  0.567339, -0.0409811]`
- model normalized outputs: `[-0.867134,  0.98469, -0.490806, -1.11931, -0.258402,  0.132457,  0.272207, -1.46874]`
- denormalized outputs: `[-0.00594108,  0.00756998, -0.0121935, -0.0193931, -4.93142e+06,  2.45453e+06,  6.58174e+06, -3.92844e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `False`

## Step 4

- run ID: `3`
- source sample index: `866`
- current raw Extension_1..4: `[ 0.238948,  0.61225,  0.706078,  0.577586]`
- predicted raw DeltaExtension_1..4: `[-0.00595647,  0.007598, -0.0111064, -0.0193515]`
- updated raw Extension_1..4: `[ 0.232991,  0.619848,  0.694972,  0.558235]`
- current raw dP_1..4: `[-8.89435e+07,  1.00467e+08,  3.5892e+07, -3.59744e+07]`
- predicted raw DeltaDP_1..4: `[-3.46728e+06,  756019,  8.9907e+06, -4.09189e+07]`
- updated raw dP_1..4: `[-9.24108e+07,  1.01223e+08,  4.48826e+07, -7.68933e+07]`
- normalized Extension values: `[-0.0642444,  0.818957,  2.24951,  0.962923]`
- normalized dP values: `[-0.815329,  1.34174,  0.698173, -0.604056]`
- model normalized outputs: `[-0.86945,  0.988329, -0.446617, -1.1169, -0.18087,  0.0408152,  0.37104, -1.53013]`
- denormalized outputs: `[-0.00595647,  0.007598, -0.0111064, -0.0193515, -3.46728e+06,  756019,  8.9907e+06, -4.09189e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `False`

## Step 5

- run ID: `3`
- source sample index: `867`
- current raw Extension_1..4: `[ 0.232991,  0.619848,  0.694972,  0.558235]`
- predicted raw DeltaExtension_1..4: `[-0.00596756,  0.0076193, -0.0101298, -0.019257]`
- updated raw Extension_1..4: `[ 0.227024,  0.627467,  0.684842,  0.538978]`
- current raw dP_1..4: `[-9.24108e+07,  1.01223e+08,  4.48826e+07, -7.68933e+07]`
- predicted raw DeltaDP_1..4: `[-4.67567e+06,  2.6028e+06,  1.64773e+07, -4.37964e+07]`
- updated raw dP_1..4: `[-9.70864e+07,  1.03826e+08,  6.13599e+07, -1.2069e+08]`
- normalized Extension values: `[-0.134487,  0.907934,  2.10049,  0.612077]`
- normalized dP values: `[-0.866319,  1.35334,  0.876894, -1.19056]`
- model normalized outputs: `[-0.871117,  0.991095, -0.406923, -1.1114, -0.244859,  0.140457,  0.678195, -1.63821]`
- denormalized outputs: `[-0.00596756,  0.0076193, -0.0101298, -0.019257, -4.67567e+06,  2.6028e+06,  1.64773e+07, -4.37964e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `False`

## Step 6

- run ID: `3`
- source sample index: `868`
- current raw Extension_1..4: `[ 0.227024,  0.627467,  0.684842,  0.538978]`
- predicted raw DeltaExtension_1..4: `[-0.00598519,  0.00764119, -0.00902053, -0.0191329]`
- updated raw Extension_1..4: `[ 0.221038,  0.635108,  0.675822,  0.519845]`
- current raw dP_1..4: `[-9.70864e+07,  1.03826e+08,  6.13599e+07, -1.2069e+08]`
- predicted raw DeltaDP_1..4: `[-4.98397e+06,  2.40166e+06,  2.64585e+07, -4.4334e+07]`
- updated raw dP_1..4: `[-1.0207e+08,  1.06227e+08,  8.78184e+07, -1.65024e+08]`
- normalized Extension values: `[-0.20486,  0.997161,  1.96457,  0.262945]`
- normalized dP values: `[-0.93508,  1.3933,  1.20443, -1.81831]`
- model normalized outputs: `[-0.873769,  0.993938, -0.361834, -1.10418, -0.261185,  0.129605,  1.0877, -1.6584]`
- denormalized outputs: `[-0.00598519,  0.00764119, -0.00902053, -0.0191329, -4.98397e+06,  2.40166e+06,  2.64585e+07, -4.4334e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `False`

## Step 7

- run ID: `3`
- source sample index: `869`
- current raw Extension_1..4: `[ 0.221038,  0.635108,  0.675822,  0.519845]`
- predicted raw DeltaExtension_1..4: `[-0.00600682,  0.00766527, -0.00778599, -0.0189745]`
- updated raw Extension_1..4: `[ 0.215032,  0.642774,  0.668036,  0.50087]`
- current raw dP_1..4: `[-1.0207e+08,  1.06227e+08,  8.78184e+07, -1.65024e+08]`
- predicted raw DeltaDP_1..4: `[-5.96728e+06,  1.78558e+06,  3.07474e+07, -4.62286e+07]`
- updated raw dP_1..4: `[-1.08038e+08,  1.08013e+08,  1.18566e+08, -2.11252e+08]`
- normalized Extension values: `[-0.275441,  1.08664,  1.84354, -0.0839365]`
- normalized dP values: `[-1.00837,  1.43017,  1.73039, -2.45376]`
- model normalized outputs: `[-0.877022,  0.997065, -0.311654, -1.09496, -0.313256,  0.0963643,  1.26366, -1.72956]`
- denormalized outputs: `[-0.00600682,  0.00766527, -0.00778599, -0.0189745, -5.96728e+06,  1.78558e+06,  3.07474e+07, -4.62286e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 8

- run ID: `3`
- source sample index: `870`
- current raw Extension_1..4: `[ 0.215032,  0.642774,  0.668036,  0.50087]`
- predicted raw DeltaExtension_1..4: `[-0.00603306,  0.00768098, -0.00643997, -0.0188184]`
- updated raw Extension_1..4: `[ 0.208998,  0.650455,  0.661596,  0.482052]`
- current raw dP_1..4: `[-1.08038e+08,  1.08013e+08,  1.18566e+08, -2.11252e+08]`
- predicted raw DeltaDP_1..4: `[-8.56105e+06,  1.91362e+06,  3.65642e+07, -5.13326e+07]`
- updated raw dP_1..4: `[-1.16599e+08,  1.09927e+08,  1.5513e+08, -2.62585e+08]`
- normalized Extension values: `[-0.346277,  1.17641,  1.73907, -0.427946]`
- normalized dP values: `[-1.09613,  1.45759,  2.34159, -3.11637]`
- model normalized outputs: `[-0.880969,  0.999105, -0.256943, -1.08587, -0.450608,  0.103273,  1.50231, -1.92127]`
- denormalized outputs: `[-0.00603306,  0.00768098, -0.00643997, -0.0188184, -8.56105e+06,  1.91362e+06,  3.65642e+07, -5.13326e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 9

- run ID: `3`
- source sample index: `871`
- current raw Extension_1..4: `[ 0.208998,  0.650455,  0.661596,  0.482052]`
- predicted raw DeltaExtension_1..4: `[-0.00605348,  0.00768839, -0.00491056, -0.018647]`
- updated raw Extension_1..4: `[ 0.202945,  0.658143,  0.656685,  0.463405]`
- current raw dP_1..4: `[-1.16599e+08,  1.09927e+08,  1.5513e+08, -2.62585e+08]`
- predicted raw DeltaDP_1..4: `[-4.74192e+06,  2.23175e+06,  7.53281e+07, -2.08727e+07]`
- updated raw dP_1..4: `[-1.21341e+08,  1.12158e+08,  2.30458e+08, -2.83458e+08]`
- normalized Extension values: `[-0.417423,  1.26636,  1.65267, -0.769126]`
- normalized dP values: `[-1.22203,  1.48697,  3.06843, -3.85214]`
- model normalized outputs: `[-0.884041,  1.00007, -0.194778, -1.0759, -0.248367,  0.120437,  3.0927, -0.777204]`
- denormalized outputs: `[-0.00605348,  0.00768839, -0.00491056, -0.018647, -4.74192e+06,  2.23175e+06,  7.53281e+07, -2.08727e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 10

- run ID: `3`
- source sample index: `872`
- current raw Extension_1..4: `[ 0.202945,  0.658143,  0.656685,  0.463405]`
- predicted raw DeltaExtension_1..4: `[-0.00606059,  0.00767111, -0.00355577, -0.0187986]`
- updated raw Extension_1..4: `[ 0.196884,  0.665814,  0.653129,  0.444606]`
- current raw dP_1..4: `[-1.21341e+08,  1.12158e+08,  2.30458e+08, -2.83458e+08]`
- predicted raw DeltaDP_1..4: `[-4.90656e+06,  1.35795e+06, -3.79827e+07, -1.19781e+08]`
- updated raw dP_1..4: `[-1.26247e+08,  1.13516e+08,  1.92475e+08, -4.03239e+08]`
- normalized Extension values: `[-0.488809,  1.3564,  1.58678, -1.1072]`
- normalized dP values: `[-1.29177,  1.52123,  4.56583, -4.15131]`
- model normalized outputs: `[-0.885109,  0.997824, -0.13971, -1.08472, -0.257086,  0.0732918, -1.55616, -4.49219]`
- denormalized outputs: `[-0.00606059,  0.00767111, -0.00355577, -0.0187986, -4.90656e+06,  1.35795e+06, -3.79827e+07, -1.19781e+08]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 11

- run ID: `3`
- source sample index: `873`
- current raw Extension_1..4: `[ 0.196884,  0.665814,  0.653129,  0.444606]`
- predicted raw DeltaExtension_1..4: `[-0.0060187,  0.0075499, -0.00537685, -0.0200763]`
- updated raw Extension_1..4: `[ 0.190866,  0.673364,  0.647752,  0.42453]`
- current raw dP_1..4: `[-1.26247e+08,  1.13516e+08,  1.92475e+08, -4.03239e+08]`
- predicted raw DeltaDP_1..4: `[-6.10823e+06, -1.48641e+07, -3.73164e+07, -9.50567e+07]`
- updated raw dP_1..4: `[-1.32355e+08,  9.86522e+07,  1.55159e+08, -4.98296e+08]`
- normalized Extension values: `[-0.560279,  1.44623,  1.53907, -1.44802]`
- normalized dP values: `[-1.36392,  1.54207,  3.8108, -5.86817]`
- model normalized outputs: `[-0.878808,  0.982083, -0.213731, -1.15907, -0.32072, -0.801958, -1.52882, -3.56354]`
- denormalized outputs: `[-0.0060187,  0.0075499, -0.00537685, -0.0200763, -6.10823e+06, -1.48641e+07, -3.73164e+07, -9.50567e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 12

- run ID: `3`
- source sample index: `874`
- current raw Extension_1..4: `[ 0.190866,  0.673364,  0.647752,  0.42453]`
- predicted raw DeltaExtension_1..4: `[-0.00597388,  0.00746524, -0.00687516, -0.020406]`
- updated raw Extension_1..4: `[ 0.184892,  0.680829,  0.640877,  0.404124]`
- current raw dP_1..4: `[-1.32355e+08,  9.86522e+07,  1.55159e+08, -4.98296e+08]`
- predicted raw DeltaDP_1..4: `[-3.65608e+06,  5.83677e+06, -2.77503e+07, -1.31674e+08]`
- updated raw dP_1..4: `[-1.36011e+08,  1.04489e+08,  1.27409e+08, -6.2997e+08]`
- normalized Extension values: `[-0.631255,  1.53464,  1.46693, -1.812]`
- normalized dP values: `[-1.45375,  1.31388,  3.06901, -7.23065]`
- model normalized outputs: `[-0.872068,  0.971088, -0.274632, -1.17826, -0.190867,  0.314943, -1.13635, -4.93888]`
- denormalized outputs: `[-0.00597388,  0.00746524, -0.00687516, -0.020406, -3.65608e+06,  5.83677e+06, -2.77503e+07, -1.31674e+08]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 13

- run ID: `3`
- source sample index: `875`
- current raw Extension_1..4: `[ 0.184892,  0.680829,  0.640877,  0.404124]`
- predicted raw DeltaExtension_1..4: `[-0.00593291,  0.0073882, -0.00800325, -0.0210259]`
- updated raw Extension_1..4: `[ 0.178959,  0.688217,  0.632874,  0.383098]`
- current raw dP_1..4: `[-1.36011e+08,  1.04489e+08,  1.27409e+08, -6.2997e+08]`
- predicted raw DeltaDP_1..4: `[-5.922e+06, -4.20925e+06, -5.16175e+07, -1.36559e+08]`
- updated raw dP_1..4: `[-1.41933e+08,  1.0028e+08,  7.57913e+07, -7.66529e+08]`
- normalized Extension values: `[-0.701703,  1.62207,  1.37468, -2.18197]`
- normalized dP values: `[-1.50752,  1.40349,  2.51738, -9.11798]`
- model normalized outputs: `[-0.865905,  0.961084, -0.320485, -1.21433, -0.310858, -0.227083, -2.11556, -5.12235]`
- denormalized outputs: `[-0.00593291,  0.0073882, -0.00800325, -0.0210259, -5.922e+06, -4.20925e+06, -5.16175e+07, -1.36559e+08]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 14

- run ID: `3`
- source sample index: `876`
- current raw Extension_1..4: `[ 0.178959,  0.688217,  0.632874,  0.383098]`
- predicted raw DeltaExtension_1..4: `[-0.00588489,  0.00731353, -0.00963646, -0.0217283]`
- updated raw Extension_1..4: `[ 0.173074,  0.695531,  0.623238,  0.36137]`
- current raw dP_1..4: `[-1.41933e+08,  1.0028e+08,  7.57913e+07, -7.66529e+08]`
- predicted raw DeltaDP_1..4: `[-1.95681e+06,  301170, -7.86498e+07, -1.40377e+08]`
- updated raw dP_1..4: `[-1.4389e+08,  1.00581e+08, -2.85858e+06, -9.06905e+08]`
- normalized Extension values: `[-0.771668,  1.70859,  1.2673, -2.56317]`
- normalized dP values: `[-1.59461,  1.33886,  1.49131, -11.0753]`
- model normalized outputs: `[-0.858684,  0.951387, -0.38687, -1.2552, -0.100883,  0.0162741, -3.22463, -5.26576]`
- denormalized outputs: `[-0.00588489,  0.00731353, -0.00963646, -0.0217283, -1.95681e+06,  301170, -7.86498e+07, -1.40377e+08]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 15

- run ID: `3`
- source sample index: `877`
- current raw Extension_1..4: `[ 0.173074,  0.695531,  0.623238,  0.36137]`
- predicted raw DeltaExtension_1..4: `[-0.00581847,  0.00722092, -0.0116277, -0.0226076]`
- updated raw Extension_1..4: `[ 0.167256,  0.702752,  0.61161,  0.338762]`
- current raw dP_1..4: `[-1.4389e+08,  1.00581e+08, -2.85858e+06, -9.06905e+08]`
- predicted raw DeltaDP_1..4: `[-2.5318e+06, -1.89605e+06, -1.02999e+08, -1.27991e+08]`
- updated raw dP_1..4: `[-1.46422e+08,  9.86848e+07, -1.05858e+08, -1.0349e+09]`
- normalized Extension values: `[-0.841066,  1.79423,  1.138, -2.95711]`
- normalized dP values: `[-1.62338,  1.34349, -0.0721236, -13.0874]`
- model normalized outputs: `[-0.848694,  0.93936, -0.467806, -1.30637, -0.131331, -0.102276, -4.22361, -4.80053]`
- denormalized outputs: `[-0.00581847,  0.00722092, -0.0116277, -0.0226076, -2.5318e+06, -1.89605e+06, -1.02999e+08, -1.27991e+08]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 16

- run ID: `3`
- source sample index: `878`
- current raw Extension_1..4: `[ 0.167256,  0.702752,  0.61161,  0.338762]`
- predicted raw DeltaExtension_1..4: `[-0.00574069,  0.00714177, -0.0140008, -0.0234961]`
- updated raw Extension_1..4: `[ 0.161515,  0.709894,  0.597609,  0.315266]`
- current raw dP_1..4: `[-1.46422e+08,  9.86848e+07, -1.05858e+08, -1.0349e+09]`
- predicted raw DeltaDP_1..4: `[ 5.40746e+06,  323660, -1.26527e+08, -1.19396e+08]`
- updated raw dP_1..4: `[-1.41015e+08,  9.90085e+07, -2.32384e+08, -1.15429e+09]`
- normalized Extension values: `[-0.909681,  1.8788,  0.981987, -3.36698]`
- normalized dP values: `[-1.66062,  1.31438, -2.11957, -14.9219]`
- model normalized outputs: `[-0.836995,  0.929081, -0.564264, -1.35807,  0.28909,  0.0174875, -5.1889, -4.47772]`
- denormalized outputs: `[-0.00574069,  0.00714177, -0.0140008, -0.0234961,  5.40746e+06,  323660, -1.26527e+08, -1.19396e+08]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 17

- run ID: `3`
- source sample index: `879`
- current raw Extension_1..4: `[ 0.161515,  0.709894,  0.597609,  0.315266]`
- predicted raw DeltaExtension_1..4: `[-0.00565135,  0.00707013, -0.0166499, -0.0244864]`
- updated raw Extension_1..4: `[ 0.155864,  0.716964,  0.580959,  0.29078]`
- current raw dP_1..4: `[-1.41015e+08,  9.90085e+07, -2.32384e+08, -1.15429e+09]`
- predicted raw DeltaDP_1..4: `[ 338486,  1.16124e+06, -1.41993e+08, -1.36451e+08]`
- updated raw dP_1..4: `[-1.40676e+08,  1.0017e+08, -3.74377e+08, -1.29074e+09]`
- normalized Extension values: `[-0.977378,  1.96243,  0.794132, -3.79297]`
- normalized dP values: `[-1.58109,  1.31935, -4.63472, -16.6333]`
- model normalized outputs: `[-0.823558,  0.919778, -0.671943, -1.4157,  0.0206638,  0.0626783, -5.82344, -5.11829]`
- denormalized outputs: `[-0.00565135,  0.00707013, -0.0166499, -0.0244864,  338486,  1.16124e+06, -1.41993e+08, -1.36451e+08]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 18

- run ID: `3`
- source sample index: `880`
- current raw Extension_1..4: `[ 0.155864,  0.716964,  0.580959,  0.29078]`
- predicted raw DeltaExtension_1..4: `[-0.00556299,  0.00700226, -0.0191693, -0.0254944]`
- updated raw Extension_1..4: `[ 0.150301,  0.723966,  0.56179,  0.265285]`
- current raw dP_1..4: `[-1.40676e+08,  1.0017e+08, -3.74377e+08, -1.29074e+09]`
- predicted raw DeltaDP_1..4: `[-1.84713e+07, -9.84035e+06, -9.3195e+07, -1.29808e+08]`
- updated raw dP_1..4: `[-1.59148e+08,  9.03294e+07, -4.67572e+08, -1.42055e+09]`
- normalized Extension values: `[-1.04402,  2.04523,  0.570734, -4.23691]`
- normalized dP values: `[-1.57612,  1.33718, -7.45731, -18.5891]`
- model normalized outputs: `[-0.810269,  0.910963, -0.774349, -1.47435, -0.975404, -0.530905, -3.82138, -4.86878]`
- denormalized outputs: `[-0.00556299,  0.00700226, -0.0191693, -0.0254944, -1.84713e+07, -9.84035e+06, -9.3195e+07, -1.29808e+08]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 19

- run ID: `3`
- source sample index: `881`
- current raw Extension_1..4: `[ 0.150301,  0.723966,  0.56179,  0.265285]`
- predicted raw DeltaExtension_1..4: `[-0.00550629,  0.00694346, -0.0209398, -0.0259774]`
- updated raw Extension_1..4: `[ 0.144794,  0.73091,  0.54085,  0.239308]`
- current raw dP_1..4: `[-1.59148e+08,  9.03294e+07, -4.67572e+08, -1.42055e+09]`
- predicted raw DeltaDP_1..4: `[ 9.69923e+06,  3.11356e+07,  2.3134e+07,  3.36809e+07]`
- updated raw dP_1..4: `[-1.49448e+08,  1.21465e+08, -4.44438e+08, -1.38687e+09]`
- normalized Extension values: `[-1.10962,  2.12723,  0.313531, -4.69913]`
- normalized dP values: `[-1.84776,  1.1861, -9.30987, -20.4496]`
- model normalized outputs: `[-0.80174,  0.903327, -0.846312, -1.50246,  0.516359,  1.67993,  0.951305,  1.27182]`
- denormalized outputs: `[-0.00550629,  0.00694346, -0.0209398, -0.0259774,  9.69923e+06,  3.11356e+07,  2.3134e+07,  3.36809e+07]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

## Step 20

- run ID: `3`
- source sample index: `882`
- current raw Extension_1..4: `[ 0.144794,  0.73091,  0.54085,  0.239308]`
- predicted raw DeltaExtension_1..4: `[-0.00548865,  0.00706892, -0.0200705, -0.0259509]`
- updated raw Extension_1..4: `[ 0.139306,  0.737978,  0.52078,  0.213357]`
- current raw dP_1..4: `[-1.49448e+08,  1.21465e+08, -4.44438e+08, -1.38687e+09]`
- predicted raw DeltaDP_1..4: `[-8.11242e+06,  5.01196e+06,  1.0766e+07,  1.00838e+08]`
- updated raw dP_1..4: `[-1.57561e+08,  1.26477e+08, -4.33673e+08, -1.28603e+09]`
- normalized Extension values: `[-1.17456,  2.20854,  0.0325734, -5.1701]`
- normalized dP values: `[-1.70512,  1.6641, -8.85001, -19.9669]`
- model normalized outputs: `[-0.799087,  0.91962, -0.810979, -1.50091, -0.426851,  0.270441,  0.443874,  3.79424]`
- denormalized outputs: `[-0.00548865,  0.00706892, -0.0200705, -0.0259509, -8.11242e+06,  5.01196e+06,  1.0766e+07,  1.00838e+08]`
- finite-value flags: `True`
- extension limit flag: `False`
- dP limit flag: `True`

# Summary

- Aggregate audit stats: `{'nan_or_inf': 0, 'extension_limit_violations': 0, 'dp_limit_violations': 38}`
- Rollout full-run summary over audited sequences: `{'run_count': 3, 'extension_trajectory_rmse_mean': 231267136.88779816, 'dp_trajectory_rmse_mean': 1.219139467930094e+17, 'extension_max_abs_error_mean': 4143276032.0, 'fraction_unstable_rollouts': 0.0, 'nan_inf_count': 0, 'limit_violation_count': 2327}`
- Catastrophic rollout classification: **verified model instability or distribution-shift sensitivity under this rollout mode**
