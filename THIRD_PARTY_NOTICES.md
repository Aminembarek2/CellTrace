# Third party notices

The original CellTrace code is licensed under MIT.
Dependency licenses and third party data terms remain separate. CellTrace does not relicense
dependency code, provider services, published research or measurement archives.

The table records the installed locked direct dependencies used for this release. License
names are declarations from distribution metadata, not a legal compatibility assessment.
Installed distributions retain their own license texts, including transitive dependencies.

| Dependency | Installed version | Declared license | Installed license files |
| --- | --- | --- | --- |
| httpx | 0.28.1 | BSD-3-Clause | httpx-0.28.1.dist-info/licenses/LICENSE.md |
| duckdb | 1.5.6 | MIT License | duckdb-1.5.6.dist-info/licenses/LICENSE |
| langgraph | 1.2.14 | MIT | langgraph-1.2.14.dist-info/licenses/LICENSE |
| langgraph-checkpoint-sqlite | 3.1.1 | MIT | langgraph_checkpoint_sqlite-3.1.1.dist-info/licenses/LICENSE |
| matplotlib | 3.11.2 | Python Software Foundation License | matplotlib-3.11.2.dist-info/LICENSE |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | numpy-2.5.3.dist-info/licenses/LICENSE.txt, numpy-2.5.3.dist-info/licenses/numpy/_core/include/numpy/libdivide/LICENSE.txt, numpy-2.5.3.dist-info/licenses/numpy/_core/src/common/pythoncapi-compat/COPYING, numpy-2.5.3.dist-info/licenses/numpy/_core/src/highway/LICENSE, numpy-2.5.3.dist-info/licenses/numpy/_core/src/multiarray/dragon4_LICENSE.txt, numpy-2.5.3.dist-info/licenses/numpy/_core/src/npysort/x86-simd-sort/LICENSE.md, numpy-2.5.3.dist-info/licenses/numpy/_core/src/umath/svml/LICENSE, numpy-2.5.3.dist-info/licenses/numpy/fft/pocketfft/LICENSE.md, numpy-2.5.3.dist-info/licenses/numpy/linalg/lapack_lite/LICENSE.txt, numpy-2.5.3.dist-info/licenses/numpy/ma/LICENSE, numpy-2.5.3.dist-info/licenses/numpy/random/LICENSE.md, numpy-2.5.3.dist-info/licenses/numpy/random/src/distributions/LICENSE.md, numpy-2.5.3.dist-info/licenses/numpy/random/src/mt19937/LICENSE.md, numpy-2.5.3.dist-info/licenses/numpy/random/src/pcg64/LICENSE.md, numpy-2.5.3.dist-info/licenses/numpy/random/src/philox/LICENSE.md, numpy-2.5.3.dist-info/licenses/numpy/random/src/sfc64/LICENSE.md, numpy-2.5.3.dist-info/licenses/numpy/random/src/splitmix64/LICENSE.md |
| pandas | 3.0.6 | BSD License | pandas-3.0.6.dist-info/LICENSE |
| pydantic | 2.13.5 | MIT | pydantic-2.13.5.dist-info/licenses/LICENSE |
| pyyaml | 6.0.3 | MIT | pyyaml-6.0.3.dist-info/licenses/LICENSE |
| scikit-learn | 1.9.1 | BSD-3-Clause | scikit_learn-1.9.1.dist-info/licenses/COPYING |

## Datasets

The measurement archives are not included in this repository and keep their own terms.

| Dataset | Source | Terms |
| --- | --- | --- |
| Raca Irish 5G production dataset | [github.com/uccmisl/5Gdataset](https://github.com/uccmisl/5Gdataset) | Repository licensed GPL 3.0. Cite D. Raca, D. Leahy, C. J. Sreenan and J. J. Quinlan, "Beyond Throughput, The Next Generation: A 5G Dataset with Channel and Context Metrics", ACM MMSys 2020 |
| Raca 4G LTE dataset, 2017 and 2018 | [University College Cork, MISL](https://www.ucc.ie/en/misl/research/datasets/ivid_4g_lte_dataset/) | No formal license, the authors ask for citation of D. Raca, J. J. Quinlan, A. H. Zahran and C. J. Sreenan, "Beyond Throughput: a 4G LTE Dataset with Channel and Context Metrics", ACM MMSys 2018 |
| PAM 2025 multi carrier dataset, three AT&T downlink drives | [github.com/NUWiNS/pam2025-multi-carrier-dataset](https://github.com/NUWiNS/pam2025-multi-carrier-dataset) | CC0 1.0 |

Only aggregate results derived from these archives are published in `results/`.
