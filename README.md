## configure

```
ccmake -S . -B build
```

## start over

```
rm -rf build && ccmake -S . -B build
```

## build

```
cmake --build build
```

## build and test

```
cmake --build build && cd build && (ctest || true) && cd ..
```

## build and run a lesson

```
cmake --build build && ./build/lesson04
```
