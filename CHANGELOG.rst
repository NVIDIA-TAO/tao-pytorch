Changelog
=========

Version 3.21.08
---------------
:Date: August 12, 2021
Soul: Update TAO release for 3.21.08

- Rename Jarvis to Riva
- Rename TLT to TAO
- Bug fixes from 3.21.08
- Integrating superres base model to ToT
- Update CI from gitlab runner to Blossom Jenkins pipeline
- Add automated publishing of Dockers to the CI

Version 3.0.17
--------------
:Date: April 10, 2021
Soul: Schema-based validation of exported .ejrvs archives with EFF 0.3.1

- Made -r command line arg mandatory to all TLT scripts for all workflows (!)
- Schema-based validation
  - Implemented validation schemes for .ejvrs files
  - Added schema based validation to export
  - Refactored unit tests to rely on schema validation
- Workflows:
  - ASR speech-to-text (Jasper/Quartznet)
  - ASR speech-to-text (CitriNet)
  - NLP text-classification (BERT)
  - NLP text-classification (Megatron)
  - NLP question-answering (BERT)
  - NLP question-answering (Megatron)
  - NLP intent and slot classification (BERT)
  - NLP intent and slot classification (Megatron)
  - NLP token classification (BERT)
  - NLP punctuation and capitalization (BERT)
- Text Classification with Megatron encoder (MR 104)
- Joint Intent Classification and Slot Filling with Megatron encoder (MR 104)
- Built and pushed a new docker image
- Updated docker ID/sha in TLT configuration
- Updated/optimized unit test for mockup models
  - Test assuring all models have CRC sums assigned
  - Made (un)encryption check lighter - retrieve only manifest
- Bumped EFF dependency to 0.3.2
- Added CHANGELOG file
   
Version 3.0.16
--------------
:Date: March 18, 2021
Soul: QA Megatron on pytorch:21.02-py3

- Updated BASE_IMAGE to pytorch:21.02-py3
- Cleaned up/modified dependencies (removed onnx-graphsurgeon, PTL pinned to 1.1.3)
- Built and pushed a new docker image
- Updated docker ID/sha in TLT configuration
- Bugfix: set encryption key for Megatron export
- Speech to Text with CitriNet (MR 101)
- QA with Megatron encoder (MR 103)
  - QA BERT and Megatron training operational and covered with tests
  - QA Megatron fine-tuning operational and covered with tests
  - QA Megatron evaluation operational and covered with tests
  - QA Megatron export operational and covered with tests
  - QA Megatron inference operational and covered with tests
  - QA Megatron ONNX inference operational and covered with tests


Version 3.0.15
--------------
:Date: Feb 25, 2021
Soul: Encryption key obfuscation
  
- Removed cmd log file (train, finetune, evaluate) in 6 workflows
- Fixed download_specs
- Reverted sys.args["encryption_key"] obfuscation
- Removed generation of cmd log file from exp_minimal_manager
- bumped version to 3.0.15
- Pulling Vahid's NeMo fix pushed to 1.0.0b4 into this release (https://github.com/NVIDIA/NeMo/pull/1762)
- Rebuilt and pushed the docker - updated docker sha!
