Extension of my [micrograd-clone](https://github.com/ckkissane/micrograd-clone). The main additions are:
* Use Vector rather than Value in engine.py
* Rewrite nn.py and functional.py to support these new Vector operations
* Add cross_entropy loss function to functional.py

It's much faster than the previous micrograd-clone implementation. 
Check out the example notebook to see it in action on a full 10 digit MNIST classification task.
