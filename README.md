# Robot Arm Environment for Reinforcement Learning 

Multi arms setup for training an Reinforcement Learning algorithm.


![demo1](demo/demo1.png)
![demo2](demo/demo2.png)


## Install 

```shell script
git clone https://github.com/MorvanZhou/RLarm
cd RLarm 
pip3 install -r requirments.txt
```

## Training

```shell script
python3 main.py -n 3 
```

-n: number of arms

## Testing
After training, run following command to test the last stored model.

```shell script
python3 main.py -n 3 --human
```