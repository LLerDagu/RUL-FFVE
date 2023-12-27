python main.py --save_dir ./checkpoint/FD001/FFVE_b1/op_4 --dataset FD001 --itr 5 --lr 0.005 --e_layers 1 --fac_C 1 --short_res 1 --long_res 0 --device cuda:0

python main.py --save_dir ./checkpoint/FD002/FFVE_b2/op_4 --dataset FD002 --itr 5 --lr 0.001 --e_layers 2 --fac_C 1 --short_res 1 --long_res 0 --device cuda:0

python main.py --save_dir ./checkpoint/FD003/FFVE_b1/op_4 --dataset FD003 --itr 5 --lr 0.001 --e_layers 1 --fac_C 1 --short_res 1 --long_res 0 --device cuda:0

python main.py --save_dir ./checkpoint/FD004/FFVE_b2/op_1 --dataset FD004 --itr 5 --lr 0.001 --e_layers 2 --fac_C 1 --short_res 1 --long_res 1 --device cuda:0