from placement import Placer
import numpy as np

all_benchmarks = [
    # 'C880',
    # 'alu2',
    'apex1',
    'apex4',
    'cm138a',
    'cm150a',
    'cm151a',
    'cm162a',
    'cps',
    'e64',
    'paira',
    'pairb'
]

init_temperatures = [1, 10, 100, 1000, 10000]
cooling_periods = [1, 10, 100, 1000]
betas = [0.8, 0.9, 0.95]


def hyperparameterSearch(benchmark, stop_condition):
    best_cost = 999999
    for init_temperature in init_temperatures:
        for cooling_period in cooling_periods:
            for beta in betas:
                placer = Placer('benchmarks/' + benchmark + '.txt')
                placer.simulatedAnnealer(init_temperature=init_temperature,
                                         cooling_period=cooling_period,
                                         early_stop_iter=stop_condition,
                                         beta=beta)
                cost = np.sum(placer.cost)

                print('benchmark {bm} get searched cost {cost} with init T {t}, cooling period {period}, and beta {b}'.format(bm=benchmark,
                                                                                                                              cost=np.sum(placer.cost),
                                                                                                                              t=init_temperature,
                                                                                                                              period=cooling_period,
                                                                                                                              b=beta))
                if cost < best_cost:
                    # record cost and hyperparameter
                    print('find better cost, update!')
                    result = {}
                    result['cost'] = cost
                    result['init_temperature'] = init_temperature
                    result['cooling_period'] = cooling_period
                    result['beta'] = beta
                    best_cost = cost
    return result


if __name__ == '__main__':
    best = {}
    print('Start hyperparameter search')
    for benchmark in all_benchmarks:
        best[benchmark] = hyperparameterSearch(benchmark, stop_condition=2e3)

    # evaluate all searched best hyperparameter with longer stop condition
    print('Evaluate best schedule cost')
    for benchmark in all_benchmarks:
        placer = Placer('benchmarks/' + benchmark + '.txt')
        schedule = best[benchmark]
        placer.simulatedAnnealer(init_temperature=schedule['init_temperature'],
                                 cooling_period=schedule['cooling_period'],
                                 early_stop_iter=1e5,
                                 beta=schedule['beta'])
        print('benchmark {bm} get final cost {cost} with init T {t}, cooling period {period}, and beta {b}'.format(bm=benchmark,
                                                                                                                   cost=np.sum(placer.cost),
                                                                                                                   t=schedule['init_temperature'],
                                                                                                                   period=schedule['cooling_period'],
                                                                                                                   b=schedule['beta']))
