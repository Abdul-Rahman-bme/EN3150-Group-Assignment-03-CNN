# Notebook preservation fixture audit

The fixture recorded notebook version `4773775b3d4a842000e633e7645d7e67c4f99a98`. The intended retained
notebook is committed version `9e36bb4983d9f8659b804a55016a8d9366c16cf1` (execution-count and output-path
update). The working notebook matches that committed version exactly.

Only the 37 affected cell-record hashes were updated. All other fixture
entries, optional cell IDs, metadata and the preservation assertion remain
unchanged. The notebook was not edited or executed to create the new hashes;
hashes came from the independently checked committed version.

The previously reported cell `4251e991` differs only in `execution_count`
(42 to 34). The assertion iterates over a set of IDs, so other runs can report
another affected cell first.

Saved output images, tables, metric values, output counts and notebook metadata
are identical between the two versions. Output fields that changed in the
existing commit are three widget IDs and nine printed report-folder paths;
they were already present before Model C work and are preserved as committed.

| Cell ID | Execution count (fixture → intended) | Additional changed field |
| --- | --- | --- |
| `a351791e` | 9 → 1 | — |
| `56be3a20` | 10 → 2 | — |
| `4fba7c67` | 11 → 3 | — |
| `1757f13d` | 12 → 4 | — |
| `ce03ef06` | 13 → 5 | — |
| `dc6fda09` | 14 → 6 | `outputs[0].data.application/vnd.jupyter.widget-view+json.model_id` |
| `cbee5fe6` | 15 → 7 | `outputs[0].data.application/vnd.jupyter.widget-view+json.model_id` |
| `c145be7c` | 16 → 8 | — |
| `b174a0d6` | 17 → 9 | — |
| `76961dc2` | 18 → 10 | — |
| `c429e85a` | 19 → 11 | — |
| `3afc242b` | 20 → 12 | — |
| `e52351c5` | 21 → 13 | — |
| `11081b96` | 22 → 14 | — |
| `c8f12cf1` | 23 → 15 | — |
| `6c468a03` | 24 → 16 | — |
| `ccd02dad` | 25 → 17 | — |
| `bb9cfbb2` | 26 → 18 | — |
| `adc703b8` | 27 → 19 | — |
| `8b928a05` | 28 → 20 | — |
| `6b7a8944` | 29 → 21 | — |
| `8c294ea0` | 30 → 22 | — |
| `6f91b7ab` | 31 → 23 | `outputs[2].text[0]` |
| `358c07f9` | 32 → 24 | — |
| `b2c22e01` | 33 → 25 | `outputs[2].text[0]` |
| `5b81687e` | 34 → 26 | `outputs[2].text[0]` |
| `64d202d7` | 35 → 27 | — |
| `9a2ce807` | 36 → 28 | `outputs[2].text[0]` |
| `0dd7d495` | 37 → 29 | `outputs[0].data.application/vnd.jupyter.widget-view+json.model_id`, `outputs[2].text[1]` |
| `ce45d6dc` | 38 → 30 | — |
| `9a1c0504` | 39 → 31 | `outputs[2].text[0]` |
| `e65a94c4` | 40 → 32 | — |
| `70cfc86b` | 41 → 33 | `outputs[2].text[0]` |
| `4251e991` | 42 → 34 | — |
| `a7c4e86b` | 43 → 35 | `outputs[2].text[0]` |
| `7fd68929` | 44 → 36 | `outputs[2].text[0]` |
| `8fbf1bbb` | 45 → 37 | — |

Exact changed output values:

- Cell `dc6fda09`, `outputs[0].data.application/vnd.jupyter.widget-view+json.model_id`:
  - Before: `b29ade98f00244758935c9409007e650`
  - Intended: `23115744603d4420a1330afaefd7d261`
- Cell `cbee5fe6`, `outputs[0].data.application/vnd.jupyter.widget-view+json.model_id`:
  - Before: `37184cf5fcdd490692f2bc642fc67c19`
  - Intended: `9eead19dc09b4a4bae2ff023c25f9846`
- Cell `6f91b7ab`, `outputs[2].text[0]`:
  - Before: `Plot saved: D:\2_ML PROJECTS\38. CNN_PR\EN3150-Assignment03-CNN\outputs\reports\notebook_model_a_adam_lr0.001_ens5_rm0\learning_curves.png`
  - Intended: `Plot saved: D:\2_ML PROJECTS\38. CNN_PR\EN3150-Assignment03-CNN\outputs\reports\notebook_model_a_adam_lr0.001_nahccos4\learning_curves.png`
- Cell `b2c22e01`, `outputs[2].text[0]`:
  - Before: `Plot saved: D:\2_ML PROJECTS\38. CNN_PR\EN3150-Assignment03-CNN\outputs\reports\notebook_model_b_adam_lr0.001_rj5ya2t_\learning_curves.png`
  - Intended: `Plot saved: D:\2_ML PROJECTS\38. CNN_PR\EN3150-Assignment03-CNN\outputs\reports\notebook_model_b_adam_lr0.001_8lmfg6ow\learning_curves.png`
- Cell `5b81687e`, `outputs[2].text[0]`:
  - Before: `Comparison saved: D:\2_ML PROJECTS\38. CNN_PR\EN3150-Assignment03-CNN\outputs\reports\notebook_adam_comparison_xky2pl7p`
  - Intended: `Comparison saved: D:\2_ML PROJECTS\38. CNN_PR\EN3150-Assignment03-CNN\outputs\reports\notebook_adam_comparison__swc8c_i`
- Cell `9a2ce807`, `outputs[2].text[0]`:
  - Before: `Plot saved: D:\2_ML PROJECTS\38. CNN_PR\EN3150-Assignment03-CNN\outputs\reports\notebook_model_a_sgd_lr0.01_d50ewpkj\learning_curves.png`
  - Intended: `Plot saved: D:\2_ML PROJECTS\38. CNN_PR\EN3150-Assignment03-CNN\outputs\reports\notebook_model_a_sgd_lr0.01_z9vk34c2\learning_curves.png`
- Cell `0dd7d495`, `outputs[0].data.application/vnd.jupyter.widget-view+json.model_id`:
  - Before: `0b81f8f0e5cc40e2b902879b2b959a46`
  - Intended: `8bab2e06e52c42c6892646648a0c4141`
- Cell `0dd7d495`, `outputs[2].text[1]`:
  - Before: `Validation report: D:\2_ML PROJECTS\38. CNN_PR\EN3150-Assignment03-CNN\outputs\reports\notebook_validation_7sh5re66\validation_check`
  - Intended: `Validation report: D:\2_ML PROJECTS\38. CNN_PR\EN3150-Assignment03-CNN\outputs\reports\notebook_validation_m7o4zsbo\validation_check`
- Cell `9a1c0504`, `outputs[2].text[0]`:
  - Before: `Plot saved: D:\2_ML PROJECTS\38. CNN_PR\EN3150-Assignment03-CNN\outputs\reports\notebook_model_a_sgd_momentum_lr0.01_qe90h8h8\learning_curves.png`
  - Intended: `Plot saved: D:\2_ML PROJECTS\38. CNN_PR\EN3150-Assignment03-CNN\outputs\reports\notebook_model_a_sgd_momentum_lr0.01_js8jk7ho\learning_curves.png`
- Cell `70cfc86b`, `outputs[2].text[0]`:
  - Before: `Plot saved: D:\2_ML PROJECTS\38. CNN_PR\EN3150-Assignment03-CNN\outputs\reports\notebook_model_b_sgd_lr0.01_38443dng\learning_curves.png`
  - Intended: `Plot saved: D:\2_ML PROJECTS\38. CNN_PR\EN3150-Assignment03-CNN\outputs\reports\notebook_model_b_sgd_lr0.01_vu0vwmo6\learning_curves.png`
- Cell `a7c4e86b`, `outputs[2].text[0]`:
  - Before: `Plot saved: D:\2_ML PROJECTS\38. CNN_PR\EN3150-Assignment03-CNN\outputs\reports\notebook_model_b_sgd_momentum_lr0.01_bb64akpk\learning_curves.png`
  - Intended: `Plot saved: D:\2_ML PROJECTS\38. CNN_PR\EN3150-Assignment03-CNN\outputs\reports\notebook_model_b_sgd_momentum_lr0.01_powxb543\learning_curves.png`
- Cell `7fd68929`, `outputs[2].text[0]`:
  - Before: `Comparison saved: D:\2_ML PROJECTS\38. CNN_PR\EN3150-Assignment03-CNN\outputs\reports\notebook_optimizer_comparison_4jplb68j`
  - Intended: `Comparison saved: D:\2_ML PROJECTS\38. CNN_PR\EN3150-Assignment03-CNN\outputs\reports\notebook_optimizer_comparison_4ggf6q3x`

Notebook SHA-256 before and after the fixture update: `b7af4ee1c5f56d212522e3e85b8ca316173f0e72039c2a89a89b3973dcce5a2d`.

Validation: `python -m unittest discover -s tests -v` in `ml_env_fixed` passed
all 18 tests, including preservation and both fresh-kernel notebook checks.
No training or test-set evaluation was performed.
