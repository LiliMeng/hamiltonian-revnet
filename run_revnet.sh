#!/usr/bin/env bash
python run_cifar_train.py --dataset=cifar-100 --model=hamiltonian-38
python run_cifar_train.py --dataset=cifar-100 --model=hamiltonian-50
python run_cifar_train.py --dataset=cifar-100 --model=hamiltonian-110

