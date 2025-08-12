Requirements:
gcc/g++ 9.4 or higher
cmake 3.10 or higher
OpenMP support
CUDA 11.8 or higher (optional, for mixed CUDA and OpenMP support)
Python 2.7 or 3.6+ (for Python bindings)

Installing Python Packages:
```bash
pip install -r requirements.txt

cmake -Bbuild && cmake --build build

pip install mae

export PYTHONPATH=.:$PYTHONPATH
```

TO-DO list:
- Implement VG-RAM training phase using batches and copy each sample test to shared memory, 
	compute hamming distance and store the canditates to find the nearest neighbour in CPU in order to avoid using atomic instructions.
	This can be an alternative to current memory restrictions.