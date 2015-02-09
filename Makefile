
code_dir = ./kcbo
test_dir = ./tests/

autopep8:
	autopep8 -r $(code_dir) $(test_dir) --in-place \
		--aggressive --aggressive \
		-v \
		--max-line-length 79 \
		--indent-size 4

lint:
	flake8 $(code_dir) --exclude=__init__.py
	flake8 $(test_dir)

autolint: lint autopep8

clean:
	find $(code_dir) $(test_dir) -name "*.pyc" -delete
	find $(code_dir) $(test_dir) -name "__pycache__" -delete

build:
	python setup.py sdist

install: build
	python setup.py install

test:
	py.test $(test_dir) -v
