
## Project Overview

This repository is a modernized and refactored version of a Transformer-based machine translation project(hyunwoongko/transformer) originally implemented in 2019. The original code provided a sequence-to-sequence (Seq2Seq) model using the Transformer architecture for English-to-German translation. However, due to the age of the codebase, several compatibility issues arose with newer library versions.

### Key Improvements

* **Updated Data Processing**: Fixed compatibility issues related to data loading and preprocessing caused by updates in libraries like `Dataloader`.
* **Added Prediction Functionality**: Introduced a dedicated prediction (predict) module, enabling easy testing and translation after training.
* **Preserved Core Architecture**: Maintained the core Transformer implementation (encoder and decoder), ensuring model performance and educational value.

This project serves as a practical starting point for learning about the Transformer architecture and building lightweight translation systems using PyTorch.

