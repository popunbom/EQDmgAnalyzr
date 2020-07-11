.PHONY : show-docs
.PHONY : docs
.PHONY : clean

SOURCE_DIR = ./_source
BUILD_DIR = ./_build

show-docs: docs
	open $(BUILD_DIR)/index.html

docs: clean
	sphinx-apidoc -f --separate -o $(SOURCE_DIR) ../
	sphinx-build -a $(SOURCE_DIR) $(BUILD_DIR)

clean:
	rm -vrf $(BUILD_DIR)
	find $(SOURCE_DIR) ! -name "conf.py" ! -name "index.rst" ! -path "$(SOURCE_DIR)" -exec rm -vf {} +
