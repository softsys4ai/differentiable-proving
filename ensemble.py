import numpy as np

lst1 = np.load('predictions/ar_en_prim_ibp_1k.npy')
lst2 = np.load('predictions/en_ar_prim_ibp_1k.npy')
lst3 = np.load('predictions/en_es_prim_ibp_1k.npy')
lst4 = np.load('predictions/en_fr_prim_ibp_1k.npy')
lst5 = np.load('predictions/en_grk_prim_ibp_1k.npy')
lst6 = np.load('predictions/en_ro_prim_ibp_1k.npy')
lst7 = np.load('predictions/es_en_prim_ibp_1k.npy')
lst8 = np.load('predictions/fr_en_prim_ibp_1k.npy')
lst9 = np.load('predictions/grk_en_prim_ibp_1k.npy')
len = lst2.shape[0]
# take the or of the numpy arrays
or_lst = np.array([False]*len)
for i in range(len):
    number_of_true = 0
    number_of_false = 0
    for item in [lst1[i], lst2[i], lst3[i], lst4[i], lst5[i], lst6[i], lst7[i], lst8[i], lst9[i]]:
        if item:
            number_of_true += 1
        else:
            number_of_false +=1
    if number_of_true > number_of_false:
        or_lst[i] = True
        
    # or_lst[i] = lst2[i] or lst3[i] or lst4[i] or lst5[i] or lst6[i] or lst7[i] or lst8[i]
    
print('Final Ensemble Acc: {}'.format(100 * sum(or_lst) / 1000))

