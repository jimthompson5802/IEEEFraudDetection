#
# Statistical tests
#
import numpy as np
import scipy.stats as stats


def calc_chi_score(c_tab, expected):
    return np.sum((c_tab - expected) ** 2 / expected)


def simulate_chi_square(ct, trials=1000):
    outcomes = ct.sum(axis=1)
    cats = ct.sum(axis=0)
    cats_slice = np.cumsum(cats).tolist()
    overall_rate = outcomes[1] / outcomes.sum()
    expected_ones = outcomes[1] * (cats / cats.sum())
    expected_zeros = cats - expected_ones
    expected = np.vstack([expected_zeros, expected_ones])
    chi_score = calc_chi_score(ct, expected)

    exceed_threshold = 0
    number_true = outcomes[1]
    total_number = outcomes.sum()
    for t in range(trials):
        true_indicators = np.random.choice(range(total_number), number_true, replace=False)
        trial = np.zeros(total_number)
        trial[true_indicators] = 1

        trial = np.split(trial, cats_slice[:ct.shape[1] - 1])
        new_trial = np.array([(x.size - x.sum(), x.sum()) for x in trial]).T
        exceed_threshold += 1 if calc_chi_score(new_trial, expected) > chi_score else 0

    return chi_score, float(exceed_threshold) / trials, expected

# unit test
if __name__ == '__main__':
    from datetime import datetime as dt
    ct = np.array([[986, 992, 988],
                   [14, 8, 12]])
    print(ct)

    start_time = dt.now()
    ans1 = stats.chi2_contingency(ct)
    print("\nscipy function:", dt.now() - start_time)
    print(ans1)

    np.random.seed(13)
    start_time = dt.now()
    ans2 = simulate_chi_square(ct)
    print("\nsimulated:", dt.now() - start_time)
    print(ans2)
    assert np.round(ans1[0],8) == np.round(ans2[0],8)
    assert ans2[1] == 0.45
    np.testing.assert_allclose(ans2[2], ans1[3])



    ct = np.array([[2161, 1733, 5759],
                   [52, 68, 227]])
    print('\n',ct)

    start_time = dt.now()
    ans1 = stats.chi2_contingency(ct)
    print("\nscipy function:", dt.now() - start_time)
    print(ans1)

    np.random.seed(13)
    start_time = dt.now()
    ans2 = simulate_chi_square(ct)
    print("\nsimulated:", dt.now() - start_time)
    print(ans2)
    assert np.round(ans1[0],8) == np.round(ans2[0],8)
    assert ans2[1] == 0.007
    np.testing.assert_allclose(ans2[2], ans1[3])