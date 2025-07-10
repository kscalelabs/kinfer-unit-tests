# kinfer-unit-tests
Create kinfer files to test kscale robot functionality

https://github.com/user-attachments/assets/0e03a622-3d84-4495-81f8-4f67c7a7e673

## Install 

```bash
pip install -r kinfer_unit_tests/requirements.txt
```

## Usage

This repo lets you make rule based kinfer files to test the basic functionality of the robot.

There are currently 3 recipes:
1) Go to zero positions
2) Go to joint biases (aka starting pose) 
3) Do a basic sinusoidal pattern

Make the kinfer files with 
```bash
# cd to repo root
python kinfer_unit_tests/make_test_kinfers.py 
```

then run any of the files with
```bash
kinfer-sim kinfer_unit_tests/assets/kbot_sine_motion.kinfer kbot --use-keyboard --suspend
```
