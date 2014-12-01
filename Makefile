
code_dir = ./kcbo

autopep8:
	autopep8 -r $(code_dir) --aggressive --aggressive --in-place -v \
		--max-line-length 79 --indent-size 4

autolint: autopep8
	flake8 $(code_dir) --exclude=__init__.py

clean:
	find $(code_dir) -name "*.pyc" -delete

build:
	python setup.py sdist

install: build
	python setup.py install
