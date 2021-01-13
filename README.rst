"Training Normalizing Flows with the Information Bottleneck for Competitive Generative Classification" (2020)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

https://arxiv.org/abs/2001.06448

USAGE
^^^^^^^^^^^^^^

* All configuration files for all experiments in the paper are contained
  in the directory 'experiments_configs'.

* Training (for one example configuration):

      `python main.py train experiments_configs/cifar10/beta_ramp/beta_1.0000.ini`

* Testing:

      `python main.py test experiments_configs/cifar10/full_model.ini`

* The cifar/mnist datasets should be downloaded automatically the first time
  it is run. For the OoD evaluation, tiny imagenet and quickdraw have to be downloaded
  separately.

REQUIREMENTS
^^^^^^^^^^^^^^

To implement the INNs, we use of the FrEIA library
(github.com/VLL-HD/FrEIA)

Requires Python >= 3.6

.. code:: sh
    pip install -r requirements.txt
