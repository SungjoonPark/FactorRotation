# Rotated Word Vector Representations and their Interpretability


## Overview
We open a TensorFlow Python source file for Rotated Word Vector Representations and their Interpretability.

The paper is published in EMNLP 2017. [<a href="http://aclweb.org/anthology/D17-1041">paper</a>, <a href="https://sungjoonpark.github.io./assets/emnlp2017_poster.pdf">poster</a>]

These files are originated in a Python version of the gradient projection rotation
algorithms (GPA) developed by Bernaards, C.A. and Jennrich, R.I.

If you need more extended version of factor rotation algorithms, you can visit:
https://github.com/mvds314/factor_rotation


## Requirements
- Python 3.6.3
- TensorFlow 1.4
- NumPy 1.13.3
- gensim 3.1.0


## How to Run
First, you should bring input word vector representation.

Input of the program is word vector representation.
'text8_word2vec_50_5_100.csv' file is example input file.
Output of the program is rotated word vector representation saved as numpy matrix file.


### Run the program
To run the program, you just run the test python file.

```sh
python example.py
```


## Reference

Robert I Jennrich. 2001. A simple general procedurefor orthogonal rotation.Psychometrika66(2):289–306
