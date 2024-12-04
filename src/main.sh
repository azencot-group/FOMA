# airfoil
python main.py --dataset Airfoil --estimate_id 0 --rho 0.975 --batch_selection knn --lr 1e-2 --foma_input 0 --foma_latent 1 --overlap 1 --alpha 1.4 --batch_size 32 --num_epochs 100 --small_singular 0

# NO2
python main.py --dataset NO2 --estimate_id 0 --rho 0.95 --batch_selection knnp --lr 1e-3 --foma_input 0 --foma_latent 1 --alpha 0.3 --batch_size 8 --num_epochs 250 --small_singular 1

# exchange_rate
python main.py --dataset TimeSeries --data_dir ./data/exchange_rate/exchange_rate.txt --ts_name exchange_rate --estimate_id 0 --rho 0.95 --batch_selection knn --lr 1e-3 --foma_input 1 --foma_latent 0 --alpha 1 --batch_size 8 --num_epochs 150 --small_singular 0

# electricity
python main.py --dataset TimeSeries --data_dir ./data/electricity/electricity.txt --ts_name electricity --estimate_id 0 --rho 0.875 --batch_selection knn --lr 1e-3 --foma_input 1 --foma_latent 0 --alpha 1 --batch_size 8 --num_epochs 200 --small_singular 0

#RCFMNIST
python main.py --dataset RCF_MNIST --data_dir ./data/RCF_MNIST --estimate_id 0 --rho 0.85 --batch_selection knnp --lr 1e-4 --foma_input 1 --foma_latent 1 --alpha 0.4 --batch_size 128 --num_epochs 50 --small_singular 1

#crime
python main.py --dataset CommunitiesAndCrime --estimate_id 0 --rho 0.875 --batch_selection knn --lr 1e-3 --foma_input 1 --foma_latent 1 --alpha 0.6 --batch_size 64 --num_epochs 100 --small_singular 1

#dti_dg
python main.py --dataset Dti_dg --data_dir ../../dti_dg/domainbed/data/ --estimate_id 0 --rho 0.825 --batch_selection knn --lr 1e-2 --foma_input 0 --foma_latent 1 --alpha 0.6 --batch_size 64 --num_epochs 250 --small_singular 0

