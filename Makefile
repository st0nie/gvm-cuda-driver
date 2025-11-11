all: custom_cuda

custom_cuda:
	gcc -fPIC -shared -g -Wall -Wshadow -Wno-error -Wno-format -static-libgcc -ldl gvm.c loader.c autofwd.c -o libcuda.so
	gcc -fPIC -shared -g -Wall -Wshadow -Wno-error -Wno-format -static-libgcc -ldl gvm.c loader.c autofwd.c -o libcuda.so.1
