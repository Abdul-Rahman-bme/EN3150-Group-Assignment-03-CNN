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

## Final-results consolidation baseline (2026-10-09)

The supplied notebook contained 113 cells. The prior preservation fixture differed for 43 cells before any edit in this task. Comparing the pre-edit snapshot with the final notebook confirms that every supplied non-source cell record is unchanged. The fixture is refreshed from that supplied snapshot, including the new final-analysis outputs. One source-only helper repair and one appended Markdown cell do not change historical outputs.

| Cell ID | Prior fixture record SHA-256 | Supplied pre-edit record SHA-256 |
| --- | --- | --- |
| `0dd7d495` | `78e733b62ac9a52e1f1b871f0da0656168e75b43596f9219926f5a4b79921e3e` | `ba80a4a86de6178bc861a606a15ee49755ddff8d914be6ad681f850a24c1312c` |
| `11081b96` | `f7b77c8612c01a206856e80de1e4dd954c2cfc9efc9aa17a98ca27147096f6de` | `eef35e92bded4447e180375d0a7de31e0b4f07e149c31b8fbb01d23a572bcde6` |
| `1757f13d` | `22f57119ecd44b55a471a1167fb14437fc2ab6b1c5bda090fc3b403fefcd4c6a` | `0ec5eb968ce908524f2b169f3201cfcf685bdc20187caacad154f1ddc4922c85` |
| `358c07f9` | `6fb8edf875ecef2608e4869df82747b099ec65d040d172f0de40640f39eff3ff` | `e5b9c1aaa3924d553299d6f5b246ab1b96c486f9704365ca91724ea8aa8c3726` |
| `3901d8bb` | `805723c8799a47e97ffde8edd077fa1962162411ed633bea65ea69a2c9bed490` | `c674731f4d358da0d2eeb850c393ac506a99f9ca1884d187aaa79afbf55510e4` |
| `3afc242b` | `32271f868c4b9b7b1971a9134214ab2a57eb25596669759cab8d3d13c5c293be` | `4d91563d95883b9b199ecd525d6a47fe4c6fcc36c0006e20f131a71cb10375f3` |
| `4251e991` | `9eedd6b5a20083ea04ca7cdb5268bbf56f12724a615f10ceb5b474970902ddd3` | `c941135c5a61dc617194214c8b8436f2627083bb778ab29862a7e51bc4bf8dd4` |
| `4dcd2a0b` | `aaa4c1ce55730168ad8b1ca79739abab7db1fc0ec2749320b56a341f660dfee5` | `92e56ed0edb4e2fbc8408bcf0ae54f6f7f04427e8e6c42c80aae4d20a226292c` |
| `4fba7c67` | `4132e83b3831d998e6392d9a12cd23cf0b0a25107fbbd6924b7d6f53e1a7302a` | `be3086a53874dc91f9f01e59618be21af6e233c651d188cf06f892e90c3baca3` |
| `56be3a20` | `f067eae1d8f64ccec537100431415a159d35c7171c2a4fe2a3385840f8ad3809` | `457b6b0b5b3642d3bf4d6bd479d144dd85c0355f4d42579e2c5d89e441baffa8` |
| `5b81687e` | `dd1dc6a7b3a639b9a5fb72932c2c1d72344ece2f202db92d252bb04653173d78` | `e6ad0ecb654573d724786d7e89d5c01f1864024e65f0d4e61151c8756e2ffa79` |
| `64d202d7` | `f0748e17002a7278bcfbb434902de91aabde1b2b9ab250cae7bc715fcc033777` | `bb55663c248f458e5ec9b1b79534d6aaf2c4d7f3ddb006225fe46dceb8c74778` |
| `6b7a8944` | `44250db6de67f6631efc8f3252d5727eedeab0b982049940c166bc61efa23b7a` | `546243a0da6c4d8c3a233e544ca2eac4f4d5cdac2406913633d08587e921e73c` |
| `6c468a03` | `1591a0d503e469fcace5098620c7a280b97acb1a7f90fe3390f53418dab90338` | `36d491909b538838b7f8c3206d025b4816606a5b74dd4bfed094a4f61d8a86ef` |
| `6e992b3c` | `885325e3f04574f712efc36e0d494f469fc476e4cd317d12679b959c29fc1e22` | `febaa9a958dcfcb337a8c5d3598a5e3ab1a66bb4d08eb7af500e168b1c3a490e` |
| `6f91b7ab` | `8a3d607f7af1b74eae22807ea4d0c5d19143be8e99da2c3fff5f6735c93f8368` | `f015b73333ecb727872afd1b15034ca361a9003e5bbd3d6e738f4d0437483358` |
| `70cfc86b` | `32a146ca4fba6ba7c1befda51c5ed271dee1b3c5cd027ed279e3e8c598163305` | `0cb78c27d267eae1592c507c7f3ee8a047b624bd4d194d185c130b434c6a973e` |
| `76961dc2` | `6625e7fe876605dc0a186b8cf0fde172562c57e22c2c427d382bf0ca3f880101` | `5d685424793cd381094cc7193cefa5c33b4a4e1a90c8575a663704fc4be75086` |
| `7fd68929` | `2135596dd7c6c99cc0549e3ed6b9fdbd419c722ce5897898b8cbbf8f0863ddc5` | `0a5c428417ebc90c86de152562a2ee72b2a9312c7520bcd00d589ab91c3737a9` |
| `8b928a05` | `6fe85f49a15e6056e459bda030d79f65a38c5509636ad0000b7b6eccc80cef23` | `0e0f4a75217f3ef36fb1db74ed1b0f6ee8bb8a6a412fc25cb0169d2c3ec3240e` |
| `8c294ea0` | `fa2f5d1fbaabda16aed2eb9b53a11b7de144b962a75ae56d6211ea8ccd0569eb` | `062f16a6dd219d78c16e989cff8dfbc043a138d18f2261a8714ed6dd2a7c24d9` |
| `8fbf1bbb` | `e23ef08a9869a571207733fbe231f7e6f018a73029d632d731624255f392ddba` | `b4f81d58b12f0a702ed171ab1fb9b8ef8c4052b749ecc7905aae940aac93749f` |
| `9a1c0504` | `48bfeb42c73fab9d5cdac8695bed882915fdef95cb13bb92b4c89a43c3650287` | `82dce6152fc554c052ef0dc0886c8e9f722538f9cee584fea7e0295f64041f1f` |
| `9a2ce807` | `7981d5d86e269ae89b53043bacaef5a5a1f462e79ef7e2359f7582bda0fb9217` | `7d34d7276587fed6634a61b21fbf6858f7edd010d86a6abb3bb83a0acee4e8d9` |
| `a351791e` | `caf0cba50f7baf092d148d7b1ea729a648e49c5f7d6af7cd19398bfb5f95a3ff` | `6c6bcbb1d4758627978a3931c7fe541f7dd28ccc2f69ccba16755dd1613c7a7d` |
| `a7c4e86b` | `e0262ffce9d810ffd496b717ec3e530789dd41a22b9a2b01fa311fc024770254` | `d5c21a0deccb8b3c8903afb401fac319b9bf3fa5b16aef8d53b40cb971c0ccf8` |
| `adc703b8` | `2764ea74275e5fa38a9ab1bb4d494c714fb5b7dd4e151d8164c8ebe576d80b0e` | `421af954bb85f86a67edadf9be3df997138db357e05eb7d8f329190e1e07224e` |
| `b174a0d6` | `a0e396ca3504e2201315810bdf3c7c810160fb918ef744d4b076ea56606f6bc1` | `a3e3ad0b44ef49daebbe054c4c9ee02070732277118fe7703cfa79805f7e830d` |
| `b2c22e01` | `8e1302661aa2ca09e48240ba02c364cb3c99eebca346778ecaf9acaa7c854961` | `f43309c96ed50d53ea532b621c9a505d4017061ddb37f6201e317e38f3450cdc` |
| `bb9cfbb2` | `6b5905e7574d00cecbe008b0182cac49e81a040f919ff3535277232143a49e40` | `10c541288af2af34bfc305b35381def25c35af46d22a378badd8ecb30a0c1e1f` |
| `c145be7c` | `c7e3ad3c890c41fd3033b42e515276671f5516deba186549f27e094878cc39ec` | `eb5679fe89166406175bd904c35fa15e07950a8cc402bc137f0eda191e2ae3c0` |
| `c429e85a` | `ce00b7c69404f35e105a62f08321e4bc761a6f5e2e498489626742802e837cee` | `a34542400124f07080774e30417f0d60fc092c80780e4978aaf21a8c2089dc2c` |
| `c5ef18d5` | `bb3aac670ac4d9423580d303b5bb61a796edee657e666ea7d1ea604666dd7d78` | `94f28758b72fee2ab8dcf1dea534002965386ff1d06b8e59f3e409b56ffc6bfc` |
| `c8f12cf1` | `e3699ca99da1dd3f186432583cd1d648270dccb49842a7fd7d301aee051fea97` | `8da4d42501e9de60f83c3734a6e84eb763210cf3195cd9359151c175e68e21dd` |
| `cbee5fe6` | `70efbeb0f8a4bc5bc10c40a34123617c326d51cfc1bbcd1f0ad830b1054bb91c` | `53113808ce286d90e2c876fdbaeda1c5272bd2f7827858ec2fc4b9529e2d7114` |
| `ccd02dad` | `0193a1ba7e80750cca4c0ac4d4c3a2b2cccb7a613a1394ae0cb6d4eca9ba2fa4` | `0331a02a76fe3cad19873648cfb4a50c953dd900d795f1187c37ed376ba57f70` |
| `ce03ef06` | `85695863b5de6dfbc7218dcfbb7a037666f1ec88948f3cb3e7b4db9cfecdb5cb` | `16cac98350dab4b9339038c77f7993788eb2e45260739c6e1bf18aaf4f9970cc` |
| `ce45d6dc` | `c6e24eebe8e1adc29880ad339a8b0bde02b055ac79bef055290eef63190095fd` | `7a038edff2670f931bd25e795cfed345e34c59c32be44aff1fe570289706b054` |
| `d73db9fc` | `b9d82d3615cd4cd1b26adab7d590d8741f4dd4f205d40cd53493663bf422a43b` | `f2f076ed9377fc43dab587d156ee678c46d76e0e790e1dd39266d696b91c23bb` |
| `dc6fda09` | `7dc0e9e83751361a828575a0254709c461de4d4dd9641530f7b9ff753885c3ac` | `5d013755c9ab9f6c8567db6f1ec935439abb2f1b472b46732ee84359f8628aba` |
| `e52351c5` | `835cec64bbc71ad2c914c5de8740823faabde0f8a05e75e4e01c722f61366855` | `663b684229073973e69e968a174d3ebf129335a3827aa1fff3a3dcf43e4c093f` |
| `e65a94c4` | `cc0734455a6254ff762969ae94a7e1d4dfc32de5bca047816ab52b3a82c5c40c` | `a43dfca8e9371cfdc43c7b3b998957cfe39dc7b553c7dcb67622fd32eec01335` |
| `fdae80a8` | `b344f192725ddc173f91db5f9881fa6f294a78cc66d713daf29fa53159b175d4` | `7e1a3d680cbbeaee6640ef3a52ea8beb12a5c6031a14a1e526a2d558faf45195` |
