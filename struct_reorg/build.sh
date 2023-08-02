#! /bin/bash

/path/to/gcc-11.3.0/install/bin/g++  -I`/path/to/gcc-11.3.0/install/bin/g++ -print-file-name=plugin`/include  -I/path/to/gcc-11.3.0/source/include -std=c++11 -Wall -fno-rtti -Wno-literal-suffix -fPIC -c -o ipa-struct-reorg.o ipa-struct-reorg.c
/path/to/gcc-11.3.0/install/bin/g++ -shared -o ipa-struct-reorg.so ipa-struct-reorg.o

