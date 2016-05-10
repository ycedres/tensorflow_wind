import graph

#print('Resultado: %f' % graph.run_training(10,10))
perf = 0
for _ in range(1,11):
    tmp_perf = graph.run_training(10,10)
    print('Resultado: %f' % tmp_perf)
    perf += tmp_perf

perf = perf/10
print(perf)