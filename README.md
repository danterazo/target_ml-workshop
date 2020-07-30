# Dante Razo's ML Workshop for Target BegINNER Con 2020
Presented on 08/06/2020

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
#### Vanilla
If Python3 is installed on your system, run the following to install dependencies:
```
> pip3 install -r requirements.txt
```

#### Conda
If you're using [Conda](https://docs.conda.io/en/latest/), consider one of the following options:

##### 1. New Environment
Create a new Conda environment like so: 
```
> conda create --name ml-workshop --file requirements.txt --yes
> conda activate ml-workshop
```

##### 2. Base Environment
Or, you can install the required packages to Conda's base environment:
```
> conda install --file requirements.txt
```

### Usage
To run the script:
```
> python3 demo.py
```

Enjoy!