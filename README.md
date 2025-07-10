# kinfer-unit-tests
Create kinfer files to test kscale robot functionality

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
# cd to kinfer-sim root
python tests/simple_kinfer_tests/make_test_kinfers.py 
```

then run any of the files with
```bash
kinfer-sim tests/simple_kinfer_tests/outputs/kbot_sine_motion.kinfer kbot --use-keyboard --suspend
```