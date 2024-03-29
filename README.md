# mesh-neural-networks

This is the repository for the paper [Formal derivation of Mesh Neural Networks with their Forward-Only gradient Propagation](https://arxiv.org/abs/1905.06684)


| ![](examples/moons_training.gif) | ![](examples/circles_training.gif) |
| --- | --- |
| ![](examples/blobs_training.gif)   | ![](examples/blobs2_training.gif) |

_All the above examples uses 5 hidden neurons_

![](examples/spirals_training.gif)

_Spirals solved with 15 neurons_

As preliminary experiments we have tested our model against some toy datasets:
* [2D Notebook](https://nbviewer.jupyter.org/github/galatolofederico/mesh-neural-networks/blob/master/examples/2D%20Mesh%20Neural%20Network.ipynb)
* [Iris Notebook](https://nbviewer.jupyter.org/github/galatolofederico/mesh-neural-networks/blob/master/examples/Iris%20Mesh%20Neural%20Network.ipynb)

We provide these additional notebooks:
* [Numerical Pytorch Test](https://github.com/galatolofederico/mesh-neural-networks/blob/master/examples/Numerical%20Test.ipynb)

We also provide the experiments for the chapter 4.2 "Real-world problems"

* [Real World Problems](https://github.com/galatolofederico/mesh-neural-networks/tree/master/real-world-problems)

The code is intentionally left unoptimized in order to be coherent with the mathematical framework presented on the paper.

An optimized version is currently under development.

For any further question you can find me at [federico.galatolo@ing.unipi.it](mailto:federico.galatolo@ing.unipi.it) or on Telegram at [@galatolo](https://t.me/galatolo)


### Citing

If you want to cite use you can use this BibTeX

```
@article{galatolo_mnn_,
  title={Formal derivation of Mesh Neural Networks with their Forward-Only gradient Propagation},
  author={Galatolo, Federico A and Cimino, Mario GCA and Vaglini, Gigliola},
  journal={Neural Processing Letters},
  pages={1--16},
  year={2021},
  publisher={Springer}
}
```

