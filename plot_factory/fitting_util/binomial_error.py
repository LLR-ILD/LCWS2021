import scipy.stats.distributions as dist


def binomialProportionMeanAndCL(total_h, pass_h, cl=0.683, a=1, b=1):
    """a,b: Parameters of the prior (Beta function B(a, b)).
    B(1, 1) is the uniform distribution U(0,1).
    Jeffreys prior: B(.5, .5)

    Expected value is NOT nSelected/nEvents.
    While that is the most probable value / mode,the expected value / mean is
    given by (nSelected+a)/(nEvents+a+b).

    From: https://arxiv.org/pdf/0908.0130.pdf
    Unused source: https://indico.cern.ch/event/66256/contributions/2071577/attachments/1017176/1447814/EfficiencyErrors.pdf
    """
    mean = (pass_h+a) / (total_h+a+b)
    p_lower = dist.beta.ppf(  (1-cl)/2., pass_h+a, total_h-pass_h+b)
    p_upper = dist.beta.ppf(1-(1-cl)/2., pass_h+a, total_h-pass_h+b)
    err_lower = mean - p_lower
    err_upper = p_upper - mean
    return mean, err_lower, err_upper


def get_binomial_1sigma_simplified(x):
    mean, err_lower, err_upper = binomialProportionMeanAndCL(x.sum(), x)
    return (err_lower + err_upper) / 2.


if __name__ == "__main__":
    import numpy as np
    Y = np.array([0.027281, 0.000604, 0.034257, 0.266154, 0.044241,
                  0.405130, 0.088262, 0.014300, 0.014797, 0.104973])
    print(get_binomial_1sigma_simplified(Y))
    print(get_binomial_1sigma_simplified(Y*1000))