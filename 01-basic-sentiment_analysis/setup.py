from setuptools import setup, find_packages

setup(name="urdu_sentiment",
      version="0.1",
      description="Helper functions for the project",
      install_requires=['scikit-learn',
                        'numpy',
                        'pandas',
                        'nltk',
                        'gensim'],
     packages=['urdu_sentiment'])
