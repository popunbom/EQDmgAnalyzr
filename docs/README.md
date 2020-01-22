# Docs

## How to build documentation using Sphinx

- PWD: `./EQDmgAnalyzr`

```shell script
$ sphinx-apidoc -f -o ./docs/source . --separate
$ sphinx-build -b html ./docs/source ./docs/build
```

