set -x

#training
python examples/data_preprocess/lighteval_math.py

python examples/data_preprocess/openr1_math_default.py

#testing
python examples/data_preprocess/test_math500.py
python examples/data_preprocess/test_amc.py
python examples/data_preprocess/test_aime2024.py
python examples/data_preprocess/test_gaokao.py
python examples/data_preprocess/test_minervamath.py
python examples/data_preprocess/test_olympiad.py
