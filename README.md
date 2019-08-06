# Deception-Detection
A Machine learning model to detect deceptive(fake) Hotel and Electronic reviews

### Dataset Used:
[Boulder Lies and Truth dataset](https://catalog.ldc.upenn.edu/LDC2014T24)

### Model Architecture:

![Model Architecture](https://raw.githubusercontent.com/nikhilsu/Deception-Detection/blob/master/model.png)

### Project Dependencies:
- The project dependencies(python libraries) can be installed by running the following command:-
```bash
$ pip install -r requirements.txt
```


### Train model:
- Run the below commands to start training and evaluating the network.
    - You will need to provide the path to the dataset, and
    - A flag(*treat_F_as_deceptive*) that tell the program whether to treat the 'F' label in the dataset as *deceptive* or to treat it as a unique class while training.
        - More information - [Paper](https://pdfs.semanticscholar.org/2020/69b7beb1069fa653953867ef4c4b78663499.pdf?_ga=2.256976139.144500798.1565130137-276775829.1564163481).

```bash
$ python main.py --path_to_dataset "<path to the BLT dataset>" --treat_F_as_deceptive <True/False>
```
