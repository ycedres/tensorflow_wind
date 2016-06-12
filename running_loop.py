import graph
import numpy as np

number_of_iterations = 10
perf_matrix = np.zeros([31,31])

for h1 in range(10, 41): # 10 to 40
    for h2 in range(10,41): # 10 to 40
        print(h1)
        print(h2)
        perf_acum = 0
        for i in range(number_of_iterations): # 0 to 9
            perf_acum += graph.run_training(h1,h2)
        perf_matrix[h1-10,h2-10] = perf_acum/number_of_iterations

np.save('/home/ycedres/Documentos/Doctorado/salida_tensor2.npy',perf_matrix)
