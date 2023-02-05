# for the calculation of impact scores
import os
import numpy as np
from keras.models import load_model

from modules import TransformerBlock, TokenAndPositionEmbedding
from config import Config

def main():
        
    # model parameters
    model_config = Config()

    embedding_layer = TokenAndPositionEmbedding(model_config.wsize, 
                                                model_config.vocab_size, 
                                                model_config.embed_dim)
    transformer_block1 = TransformerBlock(model_config.embed_dim, 
                                          model_config.num_heads, 
                                          model_config.mlp_dim)
    transformer_block2 = TransformerBlock(model_config.embed_dim, 
                                          model_config.num_heads, 
                                          model_config.mlp_dim)
    
    print("--------------------------------------------------")
    print("Data loading")
    parpath = os.path.dirname(os.getcwd())
    inputpath = parpath + '/datasets/GSE139036 disome/'

    model = load_model(parpath + '/models/yeast_disome.h5', 
                       custom_objects={'TransformerBlock': transformer_block1,
                                       'TransformerBlock': transformer_block2,
                                       'TokenAndPositionEmbedding': embedding_layer,
                                       },
                       )

    x_c = np.loadtxt(inputpath + 'disome_xc.txt', delimiter="\t")

    x_c[:, :40] = x_c[:,0:40]/100 - 5
    x_c[:, 40] = x_c[:,40]/100


    # load ribosome pausing sites
    indices = np.loadtxt(inputpath + 'disome_pause_indices.txt', delimiter='\t').astype('int')[:, 0]
    print(len(indices), "ribosome pausing sites.")

    y_pred = model.predict([x_c[:, -40:], x_c[:, :40]])
    
    print("--------------------------------------------------")
    print("Calculating sequence impact score")

    N_plot = len(indices)
    pos_start, pos_end = 0, 30
    window_size = 10
    N_rand = 50
    y_rand_mean = np.ones([N_plot, pos_end - pos_start])

    codon_list = np.append(np.arange(58), [60, 61, 63])
    x1 = np.zeros([int(N_plot*N_rand * (pos_end - pos_start)), 40])
    x2 = np.zeros([int(N_plot*N_rand * (pos_end - pos_start)), 40])
    y1 = np.zeros([int(N_plot*N_rand * (pos_end - pos_start)), 2])

    # calculate the sequence impact scores
    for i, idx in enumerate(indices):
        for j in range(pos_start, pos_end):
            
            # generate the random codons
            codon_rand = np.random.randint(low=0, high=61, size=[N_rand, window_size])
            switch_index = lambda t: codon_list[int(t)]
            vfunc = np.vectorize(switch_index)
            codon_rand = vfunc(codon_rand)
            
            # make predictions and record results
            x_codon_rand = np.tile(x_c[idx, -40:], (N_rand, 1))
            x_codon_rand[:, j:j + window_size] = codon_rand
            x_level =  np.tile(x_c[idx, :40], (N_rand, 1))

            idx1 = i*N_rand * (pos_end - pos_start) + (j-pos_start)*N_rand
            idx2 = i*N_rand * (pos_end - pos_start) + (j-pos_start)*N_rand + N_rand

            #print(idx1, idx2)
            x1[idx1:idx2] = x_codon_rand
            x2[idx1:idx2] = x_level
            y1[idx1:idx2, 0] = y_pred[idx]
            y1[idx1:idx2, 1] = x_c[idx, 20]

        if i % 1000 == 0:
            print(f"Finished {i} pause sites.")
    
    print("--------------------------------------------------")
    print("Model prediction.")
    # record the absolute changes
    y_rand2 = model.predict([x1, x2]).reshape(y1[:, 0].shape) - y1[:, 0]
    y_rand2 = np.array([np.mean(y_rand2[q:q+N_rand]) for q in range(0, len(y_rand2), N_rand)])

    y_rand_mean[0:N_plot] = y_rand2.reshape(N_plot, pos_end - pos_start)
    
    print("--------------------------------------------------")
    print("Finishing calculation and saving results.")
    np.savetxt(inputpath + 'disome_SIS.txt', y_rand_mean, delimiter='\t')
    

if __name__ == "__main__":
    main()