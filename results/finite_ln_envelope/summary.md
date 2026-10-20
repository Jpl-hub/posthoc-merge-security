# Finite O/B compensation with the complete LN norm

Status: the sole source point is feasible and independently checked; finite task probes, fresh confirmation and full-task/real-attack evaluation remain unverified. This is not an effective or original-defense claim.

| Measurement | Value |
|---|---:|
|Source images|512|
|O distance|0.35355303707522573|
|B compensation distance|8.272856708730898e-05|
|Joint distance|0.35355304675413524|
|O-only full relative-gap drift|0.4667996286234999|
|After compensation drift|0.23358894123310753|
|Original/source correct|511/511|
|Changed predictions|0|
|Solver iterations|28|
|Solver seconds|96.09241281123832|

O input-task32 distance is 2.7121571828622746e-09; B input-task32 distance is 2.6255506999199464e-09. These parameter projections do not prove functional invariance. Source class counts, numerical controls and hashes are in results.json. The original incorrect source prediction is retained.

Only original/public parameters selected head6. The solver used512 pre-existing legal training construction images, with no protected historical point, confirmation, peer or test responses. No original-expert updates, TA or AGA calls occurred. Full LN variance and feature norm were retained through two convex endpoint constraints; this is a sufficient source-only inner approximation, not an all-input or FP32 certificate.

Source/config six-path Git freeze01c3a4abc754a488f41f675bfb64388c58aa4e34 was read back before execution. Standard CVXOPT/SVD/norm support tools are prior work. One fixed O direction and one solve; no rank/head/strength/data/backend retry. Models, cone matrices and raw scores remain on3090.
