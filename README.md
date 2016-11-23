===============================
scikit-feature
===============================
## Note 
* This fork adds Python 3 support using Python-Future's [futurize](https://python-mfuture.org/futurize.html). (This should allow continued Python 2 compatibility, but this has not been verified.)
* I intend to continue incorporating any additional changes and updates from [source project](https://github.com/jundongl/scikit-feature), though this will only realistically happen when I make modifications or see an issue.
* Parameterized functions with keywords arguments rather than having to constantly parse `kwargs`.

## About
Feature selection repository scikit-feature in Python (DMML Lab@ASU). 

scikit-feature is an open-source feature selection repository in Python developed by Data Mining and Machine Learning Lab at Arizona State University. It is built upon one widely used machine learning package scikit-learn and two scientific computing packages Numpy and Scipy. scikit-feature contains around 40 popular feature selection algorithms, including traditional feature selection algorithms and some structural and streaming feature selection algorithms. 

It serves as a platform for facilitating feature selection application, research and comparative study. It is designed to share widely used feature selection algorithms developed in the feature selection research, and offer convenience for researchers and practitioners to perform empirical evaluation in developing new feature selection algorithms.

##Installing scikit-feature
###Prerequisites:
Python 3.3+ (should still work with Python 2.7)

NumPy

SciPy

Scikit-learn

###Steps:
After you download scikit-feature-1.0.0.zip from the project website (http://featureselection.asu.edu/), unzip the file.

For Linux users, you can install the repository by the following command:

    python setup.py install

For Windows users, you can also install the repository by the following command:

    setup.py install

##Project website
Instructions of using this repository can be found in our project webpage at http://featureselection.asu.edu/

##Citation

If you find scikit-feature feature selection reposoitory useful in your research, please consider citing the following paper::

    @article{li2016feature,
       title={Feature Selection: A Data Perspective},
       author={Li, Jundong and Cheng, Kewei and Wang, Suhang and Morstatter, Fred and Trevino, Robert P and Tang, Jiliang and Liu, Huan},
       journal={arXiv preprint arXiv:1601.07996},
       year={2016}
    }
    
##Contact
Jundong Li
E-mail: jundong.li@asu.edu

Kewei Cheng
E-mail: kcheng18@asu.edu
