# BegINNER Con: Machine Learning Workshop
#### By: Dante Razo, TTS CSIRT SU20 Intern
Code, resources, and data from Dante's machine learning workshop. Presented at Target Corporation's 2020 BegINNER Con on August 6, 2020.

## Proposal / Summary
This workshop will serve as a crash course for those who want to learn about one of the hottest fields in tech. The goal is to introduce basic machine learning, statistics, and natural language processing concepts. The workshop will end with a live sentiment analysis demo on a Twitter-sourced corpus.

## Contents

| **Directory** / *Filename* | Notes |
| :--: | :--: |
| **code/** | Contains all scripts |
| **data/** | Contains Waseem's datasets |
| **paper/** | Contains Waseem paper (source of data) |
| *demo.py* | Script as seen in the live demo |
| *demo_extended.py* | Object-oriented version of the demo script. Same output but easier to tinker with parameters and settings. |



## Setup

### Git

First, clone the repo:
```
> git clone https://github.com/danterazo/target_ml-workshop.git
```

Then, navigate to the repo folder:
```
> cd target_ml-workshop
```

### Python
You can use either vanilla Python or Conda.

#### Vanilla
If Python3 is installed on your system, run the following to install dependencies:
```
> pip3 install -r requirements.txt
```

#### Conda
If you're using [Conda](https://docs.conda.io/en/latest/), you can either create a new environment or use an existing one. Consider the following options:

##### 1. Create New Environment
Create a new Conda environment like so:
```
> conda create --name ml-workshop --file requirements.txt --yes
> conda activate ml-workshop
```

##### 2. Use Existing Environment
You can also install the required packages in an existing Conda environment:
```
> conda activate <conda_env_name>
> conda install --file requirements.txt
```

##### 3. Use Base Environment
Alternatively, you can install the required packages in Conda's base environment:
```
> conda install --file requirements.txt
```

### Usage
To run the script:
```
> python3 demo.py
```

Enjoy!
