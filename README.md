# 📦 HierarchicalMetric.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://your-docs-link)
[![CI](https://github.com/kudlate1/HierarchicalMetric.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/kudlate1/HierarchicalMetric.jl/actions)

🚀 **HierarchicalMetric.jl** is a Julia package providing functions and scripts for the metric learning. 

## 🔍 **Introduction**

The metric learning is based on the distance between data points in a dataset. It can be used for many approaches in machine learning such as classification, clustering, or information retrieval. The distances for vector spaces are well known and relatively easy to define (e. g. euclidean, manhattan) in contrast to structured data. In this article, we introduce the theoretical foundations of metric learning in vector spaces and also examine learning with the Hierarchically-structured Tree Distance (HTD).

## 📚 **Approaches**

We are given fully labeled datasets, where we want to learn a sparse metric to separate
two different classes using the triplet loss method. The first dataset is artificial and trivial, its main purpose is to graphically explain the theoretical foundations of the experiment, the second dataset called mutagenesis is significantly larger and its data points have a tree structure and, accordingly, more parameters to learn. The metric in these datasets is called sparse, which means that not every parameter (or dimension) affects the computation of distances; therefore, some of the parameters can converges toward zero).

## 📌 **Task**

The task is to implement a training algorithm for learning a sparse metric. This process includes the iterative examination of the model parameters within a training loop. Parameter sparsity is achieved through the application of the Lasso regularization method.
