python vqvae_train.py --embedding_dim 16 --num_embeddings 512 --cuda_num $1
# python vqvae_train.py --embedding_dim 16 --num_embeddings 1024 --cuda_num $1
# python vqvae_train.py --embedding_dim 1 --num_embeddings 8 --cuda_num $1
python vqvae_eval.py  --embedding_dim 16 --num_embeddings 512 --cuda_num $1
# python vqvae_eval.py  --embedding_dim 1 --num_embeddings 4 --cuda_num $1
# python vqvae_eval.py  --embedding_dim 1 --num_embeddings 8 --cuda_num $1
